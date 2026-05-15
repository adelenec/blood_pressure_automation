#!/usr/bin/env python3
"""
Record iPhone Mirroring (Quartz window) + Vernier LabQuest cuff pressure together.

Writes:
  <stem>_mirroring.mp4          — video at nominal --video-fps
  <stem>_pressure.csv           — columns compatible with video_pressure_viewer.py
  <stem>_video_frames.csv       — per-frame sync times (wall + session-relative)
  <stem>_collapse_events.csv    — collapse type, t_sync_s, cuff_pressure_mm_hg (interpolated)

Pressure is read on the **main thread** each video frame (Vernier `labquest` is not reliable
when `read()` runs from a worker thread — same pattern as read_vernier_cuff_labquest.py).
With the figure focused: press **v** = venous collapse (blue dashed line), **a** = arterial collapse (red dashed line);
each event prints **session time and interpolated cuff pressure (mmHg)** when LabQuest samples exist, and is saved to CSV + session JSON.

Live window: video frame + pressure vs time; close the figure to stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from read_ultrasound import DEFAULT_OWNER, find_window_id, grab_bgr

# video_pressure_viewer.py expects these headers
CSV_TIME_COL = "Latest: Time (s)"
CSV_PRESSURE_COL = "Latest: Cuff Pressure (mm Hg)"


def _poll_labquest_pressure(
    lq: object,
    t0: float,
    times: list[float],
    values: list[float],
    lock: threading.Lock,
    *,
    max_reads: int,
) -> int:
    """
    Read up to max_reads samples from ch1 on the calling thread.

    The Vernier labquest library expects open/start/read on the same thread; a
    background reader often gets None immediately (0 samples in CSV).
    """
    added = 0
    for _ in range(max(1, max_reads)):
        v = lq.read("ch1")
        if v is None:
            break
        t = time.perf_counter() - t0
        with lock:
            times.append(t)
            values.append(float(v))
        added += 1
    return added


def _interp_pressure_mmhg(
    ts: float,
    times: list[float],
    values: list[float],
    lock: threading.Lock,
) -> float | None:
    """Linearly interpolate cuff pressure at session time ts, or None if no samples."""
    with lock:
        if not times:
            return None
        pt = np.asarray(times, dtype=float)
        pv = np.asarray(values, dtype=float)
    if pt.size == 0:
        return None
    i = int(np.searchsorted(pt, ts))
    if i <= 0:
        return float(pv[0])
    if i >= pt.size:
        return float(pv[-1])
    t_lo, t_hi = float(pt[i - 1]), float(pt[i])
    v_lo, v_hi = float(pv[i - 1]), float(pv[i])
    if t_hi == t_lo:
        return v_lo
    return float(v_lo + (ts - t_lo) / (t_hi - t_lo) * (v_hi - v_lo))


def main() -> None:
    if sys.platform != "darwin":
        sys.exit("macOS only (Quartz window capture).")

    try:
        from labquest import LabQuest  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("pip install labquest (Vernier Go Direct / LabQuest driver)")

    ap = argparse.ArgumentParser(
        description="Synchronized mirroring MP4 + LabQuest pressure CSV + timing metadata."
    )
    ap.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="path prefix without suffix (default: data/session_<timestamp>)",
    )
    ap.add_argument("--video-fps", type=float, default=30.0, help="nominal MP4 frame rate")
    ap.add_argument("--labquest-hz", type=int, default=100, help="LabQuest.start() rate")
    ap.add_argument("--owner", default=DEFAULT_OWNER, help="Quartz window owner substring")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    args = ap.parse_args()

    stem = args.output_stem
    if stem is None:
        stem = Path("data") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    path_video = stem.parent / f"{stem.name}_mirroring.mp4"
    path_pressure = stem.parent / f"{stem.name}_pressure.csv"
    path_frames = stem.parent / f"{stem.name}_video_frames.csv"
    path_collapses = stem.parent / f"{stem.name}_collapse_events.csv"
    path_meta = stem.parent / f"{stem.name}_session.json"

    wid, wlabel = find_window_id(args.owner)
    frame0 = grab_bgr(wid)
    if frame0 is None:
        sys.exit("window capture failed (Screen Recording permission?)")
    h, w = frame0.shape[:2]

    lq = LabQuest()
    lq.open()
    lq.select_sensors(ch1="lq_sensor")

    lq.start(int(args.labquest_hz))
    t0 = time.perf_counter()
    t0_epoch = time.time()
    t0_iso = datetime.now(timezone.utc).astimezone().isoformat()

    pressure_t: list[float] = []
    pressure_v: list[float] = []
    plock = threading.Lock()
    # ~ one frame interval worth of 100 Hz samples (+ slack); polled on main thread only
    reads_per_frame = max(8, int(round(float(args.labquest_hz) / max(float(args.video_fps), 1.0))) + 5)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path_video), fourcc, float(args.video_fps), (w, h))
    if not vw.isOpened():
        lq.stop()
        lq.close()
        sys.exit(f"cannot open VideoWriter: {path_video}")

    frame_indices: list[int] = []
    frame_t_sync: list[float] = []
    frame_t_wall: list[float] = []

    dt = 1.0 / float(args.video_fps) if args.video_fps > 0 else 1.0 / 30.0
    next_t = t0
    n = 0

    plt.ion()
    fig, (ax_im, ax_pt) = plt.subplots(1, 2, figsize=(12, 5))
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "set_window_title"):
        mgr.set_window_title("mirroring + cuff (close window to stop)")
    rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    im_artist = ax_im.imshow(rgb0)
    ax_im.set_title(wlabel)
    ax_im.axis("off")
    (line_p,) = ax_pt.plot([0.0], [0.0], color="tab:blue", lw=1.0)
    (vline,) = ax_pt.plot([0.0, 0.0], [0.0, 1.0], color="tab:green", alpha=0.85, lw=1.5)
    ax_pt.set_xlabel("session time (s)")
    ax_pt.set_ylabel("mm Hg")
    ax_pt.set_title("cuff (LabQuest) — v venous / a arterial collapse")
    ax_pt.grid(True, alpha=0.3)
    ax_pt.set_ylim(0, 200)
    fig.tight_layout()

    run = True

    def on_close(_evt) -> None:
        nonlocal run
        run = False

    fig.canvas.mpl_connect("close_event", on_close)

    collapse_records: list[tuple[str, float, float | None]] = []

    def on_key(event) -> None:
        if not getattr(event, "key", None):
            return
        key = event.key.lower()
        if key not in (" "):
            return
        ts = time.perf_counter() - t0
        p_mm = _interp_pressure_mmhg(ts, pressure_t, pressure_v, plock)
        kind = "venous" if key == " " else "arterial"
        if key == " ":
            ax_pt.axvline(
                ts,
                color="tab:blue",
                ls="--",
                lw=1.8,
                alpha=0.9,
                zorder=5,
            )
        else:
            ax_pt.axvline(
                ts,
                color="tab:red",
                ls="--",
                lw=1.8,
                alpha=0.9,
                zorder=5,
            )
        if p_mm is None:
            print(
                f"[{kind} collapse] t_sync = {ts:.6f} s  cuff = (no LabQuest samples yet)",
                flush=True,
            )
        else:
            print(
                f"[{kind} collapse] t_sync = {ts:.6f} s  cuff = {p_mm:.2f} mmHg",
                flush=True,
            )
        collapse_records.append((kind, ts, p_mm))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def append_frame(img: np.ndarray, t_sync: float) -> None:
        nonlocal n
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        vw.write(img)
        frame_indices.append(n)
        frame_t_sync.append(t_sync)
        frame_t_wall.append(time.time())
        n += 1

    try:
        append_frame(frame0, time.perf_counter() - t0)
        _poll_labquest_pressure(
            lq, t0, pressure_t, pressure_v, plock, max_reads=reads_per_frame * 2
        )
        while run:
            if args.duration is not None and (time.perf_counter() - t0) >= args.duration:
                break
            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_sync = time.perf_counter() - t0
            frame = grab_bgr(wid)
            if frame is None:
                break
            append_frame(frame, t_sync)

            _poll_labquest_pressure(
                lq, t0, pressure_t, pressure_v, plock, max_reads=reads_per_frame
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_artist.set_data(rgb)
            with plock:
                pt = np.asarray(pressure_t, dtype=float)
                pv = np.asarray(pressure_v, dtype=float)
            if pt.size:
                line_p.set_data(pt, pv)
                ax_pt.relim()
                ax_pt.autoscale_view(scaley=False)
            ax_pt.set_ylim(0, 200)
            ylo, yhi = ax_pt.get_ylim()
            vline.set_data([t_sync, t_sync], [ylo, yhi])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        run = False
        # Avoid UI freeze: do not call thousands of blocking read() before stop().
        try:
            vw.release()
        except Exception:
            pass
        try:
            plt.close(fig)
        except Exception:
            pass
        try:
            lq.stop()
        except Exception:
            pass
        deadline = time.perf_counter() + 0.25
        n_drain = 0
        while time.perf_counter() < deadline and n_drain < 4000:
            v = lq.read("ch1")
            if v is None:
                break
            t = time.perf_counter() - t0
            with plock:
                pressure_t.append(t)
                pressure_v.append(float(v))
            n_drain += 1
        try:
            lq.close()
        except Exception:
            pass

    t_end = time.perf_counter()
    session_s = t_end - t0

    # --- write pressure CSV (viewer-compatible headers)
    with open(path_pressure, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=[CSV_TIME_COL, CSV_PRESSURE_COL])
        writer.writeheader()
        with plock:
            for ti, vi in zip(pressure_t, pressure_v):
                writer.writerow({CSV_TIME_COL: f"{ti:.9f}", CSV_PRESSURE_COL: f"{vi:.6f}"})

    with open(path_frames, "w", newline="", encoding="utf-8") as fp:
        fw = csv.writer(fp)
        fw.writerow(["frame_index", "t_sync_s", "wall_epoch_s"])
        for i, ts, tw in zip(frame_indices, frame_t_sync, frame_t_wall):
            fw.writerow([i, f"{ts:.9f}", f"{tw:.6f}"])

    collapse_sorted = sorted(collapse_records, key=lambda r: r[1])
    with open(path_collapses, "w", newline="", encoding="utf-8") as fp:
        cw = csv.DictWriter(
            fp,
            fieldnames=["event_type", "t_sync_s", "cuff_pressure_mm_hg"],
        )
        cw.writeheader()
        for kind, ts, p_mm in collapse_sorted:
            row = {
                "event_type": kind,
                "t_sync_s": f"{ts:.9f}",
                "cuff_pressure_mm_hg": "" if p_mm is None else f"{p_mm:.6f}",
            }
            cw.writerow(row)

    # observed rates from timestamps
    vts = np.array(frame_t_sync, dtype=float)
    vdt = np.diff(vts) if vts.size > 1 else np.array([])
    pts = np.asarray(pressure_t, dtype=float)
    pdt = np.diff(pts) if pts.size > 1 else np.array([])

    meta = {
        "t0_epoch_s": t0_epoch,
        "t0_local_iso": t0_iso,
        "session_wall_s": float(session_s),
        "video": {
            "nominal_fps": float(args.video_fps),
            "frame_count": int(n),
            "mean_inter_frame_s": float(vdt.mean()) if vdt.size else None,
            "std_inter_frame_s": float(vdt.std()) if vdt.size else None,
            "implied_mean_fps": float(1.0 / vdt.mean()) if vdt.size and vdt.mean() > 0 else None,
        },
        "pressure": {
            "nominal_hz": int(args.labquest_hz),
            "sample_count": len(pressure_t),
            "mean_inter_sample_s": float(pdt.mean()) if pdt.size else None,
            "std_inter_sample_s": float(pdt.std()) if pdt.size else None,
            "implied_mean_hz": float(1.0 / pdt.mean()) if pdt.size and pdt.mean() > 0 else None,
        },
        "collapse_events": [
            {
                "type": kind,
                "t_sync_s": float(ts),
                "cuff_pressure_mm_hg": p_mm,
            }
            for kind, ts, p_mm in collapse_sorted
        ],
        "files": {
            "video": str(path_video.resolve()),
            "pressure_csv": str(path_pressure.resolve()),
            "video_frames_csv": str(path_frames.resolve()),
            "collapse_events_csv": str(path_collapses.resolve()),
        },
        "quartz_window_owner": wlabel,
        "quartz_window_id": int(wid),
    }
    path_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {n} video frames -> {path_video}", file=sys.stderr)
    print(f"wrote {len(pressure_t)} pressure samples -> {path_pressure}", file=sys.stderr)
    if not pressure_t:
        print(
            "warning: zero pressure samples — check USB/LabQuest app, ch1 sensor, "
            "and that this script is the only program using the device.",
            file=sys.stderr,
        )
    print(f"wrote frame timestamps -> {path_frames}", file=sys.stderr)
    print(
        f"wrote {len(collapse_sorted)} collapse marks -> {path_collapses}",
        file=sys.stderr,
    )
    print(f"wrote session meta -> {path_meta}", file=sys.stderr)


if __name__ == "__main__":
    main()
