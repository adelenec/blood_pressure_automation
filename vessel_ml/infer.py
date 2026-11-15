"""Run the trained U-Net on a video and turn the masks into collapse events.

Usage: python vessel_ml/infer.py --ckpt vessel_ml/checkpoints/unet_x.pt VIDEO_ID [...]

Writes to data/vessel_dataset/pred_<tag>/:
  <vid>.csv   frame, t_s, artery_px, vein_px, cuff_mm_hg   (areas at 256x256)
  <vid>.png   area + pressure plot with detected and manually logged events
  <vid>.mp4   overlay video
  <vid>.json  detected events

Event rules (from Arthur, 2026-11-15), applied to the deflation phase:
  systolic  = cuff pressure when the artery first reappears after the collapse
              (the longest stretch with no artery).
  diastolic = cuff pressure after which the artery never disappears again.
  venous    = cuff pressure where the vein reappears and stays.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_unet import DS, ROOT, SIZE, SmallUNet

HERE = Path(__file__).parent
BEAT_S = 1.0  # assumed heartbeat period. UNSURE: should come from the pulse, not a constant.


def load_session(vid, n_frames, fps=30.0):
    sessions = json.loads((HERE / "sessions.json").read_text())
    s = sessions.get(vid, {})
    t = np.arange(n_frames) / fps
    pressure = events = None
    if s.get("frames_csv"):
        t = pd.read_csv(ROOT / s["frames_csv"])["t_sync_s"].to_numpy()[:n_frames]
    if s.get("pressure_csv"):
        p = pd.read_csv(ROOT / s["pressure_csv"])
        pressure = (p.iloc[:, 0].to_numpy(), p.iloc[:, 1].to_numpy())
    if s.get("events_csv"):
        events = pd.read_csv(ROOT / s["events_csv"])
    return t, pressure, events


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def detect_events(t, artery, vein, cuff, valid):
    """All inputs are per-frame arrays. Returns dict of events (or None)."""
    out = {"systolic": None, "diastolic": None, "venous_reopen": None, "notes": []}
    if cuff is None or valid.sum() < 10:
        out["notes"].append("no pressure trace or too few usable frames")
        return out
    idx = np.flatnonzero(valid)
    peak = idx[np.argmax(cuff[idx])]
    defl = idx[idx >= peak]  # deflation phase, usable frames only

    def presence(area):
        ref = np.percentile(area[idx], 95)
        if ref < 30:
            return None
        return area >= max(20, 0.15 * ref)

    pa = presence(artery)
    if pa is None:
        out["notes"].append("artery never clearly visible")
    else:
        absent = ~pa[defl]
        rs = runs(absent)
        if not rs:
            out["notes"].append("artery never absent during deflation")
        else:
            # the collapse is the longest stretch without the artery; what follows is the
            # pulsating phase (artery visible on some frames only), then continuously open
            a, b = max(rs, key=lambda r: t[defl[min(r[1], len(defl) - 1)]] - t[defl[r[0]]])
            if b < len(defl):
                i = defl[b]
                out["systolic"] = {"frame": int(i), "t_s": float(t[i]), "cuff_mm_hg": float(cuff[i])}
                later_abs = np.flatnonzero(absent[b:])
                j = defl[b + later_abs[-1] + 1] if len(later_abs) and b + later_abs[-1] + 1 < len(defl) else i
                out["diastolic"] = {"frame": int(j), "t_s": float(t[j]), "cuff_mm_hg": float(cuff[j])}
            else:
                out["notes"].append("artery never reappears")
    pv = presence(vein)
    if pv is None:
        out["notes"].append("vein never clearly visible")
    else:
        absent = ~pv[defl]
        ab = np.flatnonzero(absent)
        if len(ab) and ab[-1] + 1 < len(defl):
            k = defl[ab[-1] + 1]
            out["venous_reopen"] = {"frame": int(k), "t_s": float(t[k]), "cuff_mm_hg": float(cuff[k])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("vids", nargs="+")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = SmallUNet().to(device).eval()
    model.load_state_dict(ck["model"])
    out_dir = DS / ("pred_" + Path(args.ckpt).stem.replace("unet_", ""))
    out_dir.mkdir(exist_ok=True)

    for vid in args.vids:
        files = sorted((DS / "frames" / vid).glob("*.jpg"))
        t, pressure, events = load_session(vid, len(files))
        cuff = np.interp(t, *pressure) if pressure else None
        h, w = cv2.imread(str(files[0]), 0).shape
        writer = cv2.VideoWriter(str(out_dir / f"{vid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        art, vein, valid = [], [], []
        for i in range(0, len(files), 32):
            imgs = [cv2.imread(str(f), 0) for f in files[i : i + 32]]
            x = np.stack([cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for im in imgs])
            with torch.no_grad():
                pred = model(torch.from_numpy(x).float().div(255)[:, None].to(device)).argmax(1).cpu().numpy()
            for k, (im, p) in enumerate(zip(imgs, pred)):
                art.append(int((p == 1).sum()))
                vein.append(int((p == 2).sum()))
                valid.append(bool(im[h // 3 :].mean() >= 12))  # not a frozen / no-contact frame
                vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                pm = cv2.resize(p.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                for c, col in ((1, (0, 0, 255)), (2, (255, 80, 0))):
                    cn, _ = cv2.findContours((pm == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(vis, cn, -1, col, 1)
                label = f"f{i + k}" + (f"  {cuff[i + k]:.0f} mmHg" if cuff is not None else "")
                cv2.putText(vis, label, (4, h - 6), 0, 0.45, (0, 255, 0), 1)
                writer.write(vis)
        writer.release()
        art, vein, valid = np.array(art), np.array(vein), np.array(valid)
        t = t[: len(art)]
        det = detect_events(t, art, vein, cuff, valid)
        (out_dir / f"{vid}.json").write_text(json.dumps(det, indent=1) + "\n")
        pd.DataFrame({"frame": np.arange(len(art)), "t_s": t, "artery_px": art, "vein_px": vein,
                      "usable": valid, "cuff_mm_hg": cuff if cuff is not None else np.nan}
                     ).to_csv(out_dir / f"{vid}.csv", index=False)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, np.where(valid, art, np.nan), color="tab:red", lw=1, marker=".", ms=3, label="artery area")
        ax.plot(t, np.where(valid, vein, np.nan), color="tab:blue", lw=1, marker=".", ms=3, label="vein area")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("area (px at 256x256)")
        if cuff is not None:
            ax2 = ax.twinx()
            ax2.plot(t, cuff, color="0.45", lw=1, label="cuff pressure")
            ax2.set_ylabel("cuff pressure (mm Hg)")
        if events is not None:
            for _, e in events.iterrows():
                ax.axvline(e["t_sync_s"], color="0.2", ls=":", lw=1)
                ax.text(e["t_sync_s"], ax.get_ylim()[1], f"manual {e['cuff_pressure_mm_hg']:.0f}",
                        rotation=90, va="top", ha="right", fontsize=7)
        for name, col in (("systolic", "tab:red"), ("diastolic", "tab:orange"), ("venous_reopen", "tab:blue")):
            if det[name]:
                ax.axvline(det[name]["t_s"], color=col, ls="--", lw=1)
                ax.text(det[name]["t_s"], ax.get_ylim()[1] * 0.55, f"{name} {det[name]['cuff_mm_hg']:.0f}",
                        rotation=90, va="top", ha="left", fontsize=7, color=col)
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title(vid)
        fig.tight_layout()
        fig.savefig(out_dir / f"{vid}.png", dpi=110)
        plt.close(fig)
        print(vid, json.dumps({k: (v["cuff_mm_hg"] if isinstance(v, dict) else v) for k, v in det.items()}))


if __name__ == "__main__":
    main()
