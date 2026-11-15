"""Render a review video of the U-Net's output.

    cd vessel_ml && ../.venv-linux/bin/python visualize.py --ckpt checkpoints/unet_x.pt VIDEO_ID [...]

Each frame shows: raw image | U-Net prediction | SAM / human label (nearest labelled
frame), with the predicted artery and vein area over the whole video underneath and
a cursor at the current frame. Output: data/vessel_dataset/viz_<tag>/<vid>.mp4 (H.264).
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from train_unet import DS, SIZE, SmallUNet

COL = {1: (60, 60, 255), 2: (255, 140, 40)}  # BGR: artery red, vein blue
PANEL_H = 440
FPS = {"butterfly_no_color": 70 / 3, "phillips_no_color": 30.0}  # others: slow capture, play at 5 fps


def paint(img, mask):
    out = img.copy()
    for c, col in COL.items():
        m = mask == c
        if m.any():
            out[m] = (0.65 * out[m] + 0.35 * np.array(col)).astype(np.uint8)
            cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cn, -1, col, 2)
    return out


def title(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("vids", nargs="+")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = SmallUNet().to(device).eval()
    model.load_state_dict(ck["model"])
    out_dir = DS / ("viz_" + Path(args.ckpt).stem.replace("unet_", ""))
    out_dir.mkdir(exist_ok=True)

    for vid in args.vids:
        files = sorted((DS / "frames" / vid).glob("*.jpg"))
        held_out = vid in ck.get("val_vids", [])
        h0, w0 = cv2.imread(str(files[0]), 0).shape
        pw = int(round(w0 * PANEL_H / h0 / 2)) * 2
        preds = []
        for i in range(0, len(files), 32):
            x = np.stack([cv2.resize(cv2.imread(str(f), 0), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                          for f in files[i : i + 32]])
            with torch.no_grad():
                preds += list(model(torch.from_numpy(x).float().div(255)[:, None].to(device)).argmax(1).cpu().numpy().astype(np.uint8))
        area = np.array([[(p == 1).sum(), (p == 2).sum()] for p in preds], float)
        labelled = sorted(int(m.stem) for m in (DS / "masks" / vid).glob("*.png")) if (DS / "masks" / vid).exists() else []
        lab_arr = np.array(labelled)

        W, strip_h = 3 * pw, 130
        H = PANEL_H + strip_h
        # static timeline: predicted areas over the whole video
        strip = np.full((strip_h, W, 3), 18, np.uint8)
        mx = max(1.0, area.max())
        xs = (np.arange(len(files)) / max(1, len(files) - 1) * (W - 1)).astype(int)
        for k, c in ((0, 1), (1, 2)):
            pts = np.stack([xs, strip_h - 22 - (area[:, k] / mx * (strip_h - 46)).astype(int)], 1)
            cv2.polylines(strip, [pts.reshape(-1, 1, 2)], False, COL[c], 1, cv2.LINE_AA)
        cv2.putText(strip, "U-Net predicted area over the video:  red = artery   blue = vein   (yellow line = now)",
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        fps = FPS.get(vid, 5.0)
        ff = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:.3f}",
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_dir / f"{vid}.mp4")],
            stdin=subprocess.PIPE)
        for i, f in enumerate(files):
            raw = cv2.resize(cv2.imread(str(f)), (pw, PANEL_H), interpolation=cv2.INTER_CUBIC)
            pm = cv2.resize(preds[i], (pw, PANEL_H), interpolation=cv2.INTER_NEAREST)
            p_panel = paint(raw, pm)
            l_panel = raw.copy()
            l_text = "label: none for this video"
            if len(lab_arr):
                j = int(lab_arr[np.argmin(np.abs(lab_arr - i))])
                if abs(j - i) <= 2:
                    lm = cv2.resize(cv2.imread(str(DS / "masks" / vid / f"{j:05d}.png"), 0), (pw, PANEL_H),
                                    interpolation=cv2.INTER_NEAREST)
                    l_panel = paint(raw, lm)
                    l_text = f"label (SAM from clicks), frame {j}"
                else:
                    l_panel = (raw * 0.4).astype(np.uint8)
                    l_text = "label: frame not labelled"
            a, v = int(area[i, 0]), int(area[i, 1])
            top = np.hstack([
                title(raw.copy(), f"{vid}  frame {i}/{len(files) - 1}"),
                title(p_panel, f"U-Net{' (held out)' if held_out else ' (TRAINING video)'}  artery {'yes' if a >= 20 else 'no '}  vein {'yes' if v >= 20 else 'no'}"),
                title(l_panel, l_text),
            ])
            s = strip.copy()
            cv2.line(s, (xs[i], 18), (xs[i], strip_h - 1), (0, 220, 255), 1)
            ff.stdin.write(np.vstack([top, s]).tobytes())
        ff.stdin.close()
        ff.wait()
        print(vid, "->", out_dir / f"{vid}.mp4", f"({len(files)} frames at {fps:.1f} fps, held_out={held_out})")


if __name__ == "__main__":
    main()
