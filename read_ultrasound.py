#!/usr/bin/env python3
"""
Capture the macOS iPhone Mirroring (or any) window via Quartz — BGR frames for OpenCV.

Used by record_mirroring_labquest.py (find_window_id, grab_bgr). Run this script directly
to record a standalone MP4 from the matched window.

Requires: macOS, Screen Recording permission for the terminal/Python host, pyobjc Quartz, opencv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Substring matched against kCGWindowOwnerName / kCGWindowName (case-insensitive).
DEFAULT_OWNER = "iPhone Mirroring"


def _cgimage_to_bgr(cgimage) -> np.ndarray:
    """Convert a Quartz CGImageRef to HxWx3 uint8 BGR (OpenCV)."""
    from Quartz import (
        CGDataProviderCopyData,
        CGImageGetBitsPerPixel,
        CGImageGetBytesPerRow,
        CGImageGetDataProvider,
        CGImageGetHeight,
        CGImageGetWidth,
    )

    w = int(CGImageGetWidth(cgimage))
    h = int(CGImageGetHeight(cgimage))
    bpp = int(CGImageGetBitsPerPixel(cgimage))
    bpr = int(CGImageGetBytesPerRow(cgimage))
    if bpp != 32:
        raise ValueError(f"expected 32 bpp CGImage from window capture, got {bpp}")
    prov = CGImageGetDataProvider(cgimage)
    data = CGDataProviderCopyData(prov)
    raw = np.frombuffer(data, dtype=np.uint8, count=bpr * h)
    rgba = raw.reshape((h, bpr // 4, 4))[:, :w, :].copy()
    # macOS window images are typically premultiplied BGRA in buffer order OpenCV expects as BGRA.
    return cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)


def find_window_id(owner_substring: str) -> tuple[int, str]:
    """
    Pick the largest on-screen window whose owner or title contains owner_substring.

    Returns:
        (window_id, short_label for UI)
    """
    from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionOnScreenOnly

    needle = owner_substring.strip().lower()
    if not needle:
        raise ValueError("owner_substring must be non-empty")

    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    candidates: list[tuple[float, int, str]] = []
    for w in windows:
        owner = (w.get("kCGWindowOwnerName") or "").lower()
        name = (w.get("kCGWindowName") or "").lower()
        if needle not in owner and needle not in name:
            continue
        wid = int(w["kCGWindowNumber"])
        bounds = w.get("kCGWindowBounds") or {}
        try:
            area = float(bounds.get("Width", 0)) * float(bounds.get("Height", 0))
        except (TypeError, ValueError):
            area = 0.0
        label = f"{w.get('kCGWindowOwnerName', '')} — {w.get('kCGWindowName', '')}".strip(
            " —"
        )
        candidates.append((area, wid, label or owner_substring))

    if not candidates:
        raise RuntimeError(
            f"No on-screen window matched {owner_substring!r}. "
            "Open iPhone Mirroring (or adjust --owner), and grant Screen Recording to this app."
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    _area, wid, label = candidates[0]
    return wid, label


def grab_bgr(window_id: int) -> np.ndarray | None:
    """
    Grab one frame from the given CGWindowID as BGR uint8, or None if capture fails.
    """
    from Quartz import (
        CGWindowListCreateImage,
        CGRectInfinite,
        kCGWindowImageBoundsIgnoreFraming,
        kCGWindowImageNominalResolution,
        kCGWindowListOptionIncludingWindow,
    )

    img = CGWindowListCreateImage(
        CGRectInfinite,
        kCGWindowListOptionIncludingWindow,
        window_id,
        kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution,
    )
    if img is None:
        return None
    return _cgimage_to_bgr(img)


def record_mirroring_mp4(
    *,
    output: Path,
    owner: str,
    fps: float,
    duration_s: float | None,
) -> int:
    """Record window to MP4; returns frame count."""
    wid, wlabel = find_window_id(owner)
    frame0 = grab_bgr(wid)
    if frame0 is None:
        raise RuntimeError("grab_bgr returned None (Screen Recording permission?)")
    h0, w0 = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(output), fourcc, float(fps), (w0, h0))
    if not vw.isOpened():
        raise RuntimeError(f"cannot open VideoWriter: {output}")
    n = 0
    t0 = time.perf_counter()
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    next_t = t0
    try:
        while True:
            if duration_s is not None and (time.perf_counter() - t0) >= duration_s:
                break
            frame = grab_bgr(wid)
            if frame is None:
                break
            if frame.shape[0] != h0 or frame.shape[1] != w0:
                frame = cv2.resize(frame, (w0, h0))
            vw.write(frame)
            n += 1
            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        vw.release()
    print(f"wrote {n} frames -> {output.resolve()} ({wlabel})", file=sys.stderr)
    return n


def main() -> None:
    if sys.platform != "darwin":
        sys.exit("macOS only (Quartz window capture).")

    ap = argparse.ArgumentParser(
        description="Record iPhone Mirroring (or matching window) to MP4 via Quartz."
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("mirroring_capture.mp4"),
        help="output MP4 path",
    )
    ap.add_argument("--owner", default=DEFAULT_OWNER, help="substring for Quartz window owner/title")
    ap.add_argument("--fps", type=float, default=30.0, help="nominal capture frame rate")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds (default: until Ctrl+C)")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        record_mirroring_mp4(
            output=args.output,
            owner=args.owner,
            fps=float(args.fps),
            duration_s=args.duration,
        )
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
