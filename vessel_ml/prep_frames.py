"""Crop every ultrasound video to the moving image region and dump frames.

The scanner / phone UI is static while the ultrasound image changes, so the
bounding box of the high temporal-variance region isolates the image without
per-video hand tuning. Boxes are written to crops.json so they can be
inspected and overridden by hand (set "manual": true to stop re-detection).

Usage: python vessel_ml/prep_frames.py [--only NAME_SUBSTRING]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vessel_dataset"
CROPS_JSON = Path(__file__).with_name("crops.json")

VIDEOS = sorted(
    p for p in (ROOT / "data").rglob("*.mp4") if "vessel_dataset" not in p.parts
)


def video_id(path: Path) -> str:
    return path.stem


def detect_crop(path: Path, n_samples: int = 150):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(np.linspace(0, total - 1, min(n_samples, total)).astype(int).tolist())
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    stack = np.stack(frames).astype(np.float32)
    std = stack.std(axis=0)
    mask = (std > 12).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise RuntimeError(f"no moving region found in {path}")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[k, :4])
    return {"x": x, "y": y, "w": w, "h": h, "frames": i, "manual": False}


def dump_frames(path: Path, crop: dict):
    out_dir = OUT / "frames" / video_id(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(path))
    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        # SAM 2's video loader wants files named <int>.jpg
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    cap.release()
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    crops = json.loads(CROPS_JSON.read_text()) if CROPS_JSON.exists() else {}
    for path in VIDEOS:
        vid = video_id(path)
        if args.only and args.only not in vid:
            continue
        if not crops.get(vid, {}).get("manual"):
            crops[vid] = detect_crop(path)
        crops[vid]["source"] = str(path.relative_to(ROOT))
        n = dump_frames(path, crops[vid])
        c = crops[vid]
        print(f"{vid}: {n} frames, crop x={c['x']} y={c['y']} w={c['w']} h={c['h']}")
    CROPS_JSON.write_text(json.dumps(crops, indent=2) + "\n")


if __name__ == "__main__":
    main()
