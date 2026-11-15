"""Classical (non-ML) vessel tracker, as a baseline for the U-Net.

Each vessel is seeded from its first positive click in prompts.json and then
followed frame to frame:
  1. blur the frame, take a search window around the last known centre
  2. Otsu-threshold the window (vessel lumen is darker than its surroundings)
  3. keep the dark blob closest to the last centre that passes area / shape /
     contrast gates; otherwise report "absent" and keep the last position
The only per-video input is the seed click; all gates are relative to the
seed blob's size, so there are no per-video hand-tuned thresholds.

Usage: python vessel_ml/classical_baseline.py [VIDEO_ID ...]
Writes masks to data/vessel_dataset/classical/<vid>/ in the same format as the
SAM labels and prints agreement with those labels.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "vessel_dataset"
HERE = Path(__file__).parent
OBJ = {"artery": 1, "vein": 2}


def dark_blob(gray, centre, radius, level, exclude=None, lost=0):
    """Best dark blob near `centre`. `level` is the expected lumen brightness.
    Returns (mask, centre, area, level) or None."""
    h, w = gray.shape
    # Growing the search area while the vessel is lost (reach = 1 + 0.25 * lost, with a
    # stricter size gate) was tried on 2026-11-15 and made every video worse: the track
    # jumped onto dark muscle. Left disabled.
    reach, min_frac = 1.0, 0.08
    win = int(max(2.5 * radius, 24) * reach)
    x0, x1 = max(0, int(centre[0]) - win), min(w, int(centre[0]) + win)
    y0, y1 = max(0, int(centre[1]) - win), min(h, int(centre[1]) + win)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    # Otsu on the window does not work: muscle is nearly as dark as the lumen and
    # lands in the same class. Threshold part-way between the lumen and the
    # window's typical brightness instead.
    bg = float(np.median(roi))
    if bg - level < 10:
        return None
    dark = (roi < level + 0.45 * (bg - level)).astype(np.uint8)
    if exclude is not None:
        dark[exclude[y0:y1, x0:x1] > 0] = 0
    k = max(3, int(radius / 4) | 1)  # break thin bridges to neighbouring dark tissue
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dark)
    ref_area = np.pi * radius**2
    best = None
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        if not (min_frac * ref_area <= area <= 2.5 * ref_area):
            continue
        if area / (bw * bh) < 0.45 or max(bw, bh) / max(1, min(bw, bh)) > 3.5:
            continue  # not compact: a streak or a ragged region, not a lumen
        # touching the window border means the dark region continues outside (muscle, shadow)
        if stats[k, 0] == 0 or stats[k, 1] == 0 or stats[k, 0] + bw == roi.shape[1] or stats[k, 1] + bh == roi.shape[0]:
            continue
        comp = lab == k
        c = cents[k] + (x0, y0)
        d = np.hypot(*(c - centre))
        if d > (1.5 * radius + 6) * reach:
            continue
        if best is None or d < best[0]:
            best = (d, comp, c, area)
    if best is None:
        return None
    full = np.zeros_like(gray, np.uint8)
    full[y0:y1, x0:x1][best[1]] = 1
    return full, best[2], best[3], float(np.median(roi[best[1]]))


def track(vid, cfg):
    files = sorted((DS / "frames" / vid).glob("*.jpg"))
    seeds = {}
    for p in cfg["prompts"]:
        pos = [pt for pt, lab in zip(p["points"], p.get("labels", [1] * len(p["points"]))) if lab == 1]
        if pos and (p["obj"] not in seeds or p["frame"] < seeds[p["obj"]][0]):
            seeds[p["obj"]] = (p["frame"], np.array(pos[0], float))
    out = {i: np.zeros(cv2.imread(str(files[0]), 0).shape, np.uint8) for i in range(len(files))}
    grays = {}

    def gray(i):
        if i not in grays:
            grays[i] = cv2.GaussianBlur(cv2.imread(str(files[i]), 0), (0, 0), 2.0)
        return grays[i]

    for name in ("artery", "vein"):  # artery first, so the vein cannot take its pixels
        if name not in seeds:
            continue
        f0, c0 = seeds[name]
        g0 = gray(f0)
        level0 = float(np.median(g0[int(c0[1]) - 2 : int(c0[1]) + 3, int(c0[0]) - 2 : int(c0[0]) + 3]))
        first = None
        for frac in (0.04, 0.06, 0.09, 0.13):  # seed size is unknown: try a few scales
            cand = dark_blob(g0, c0, frac * g0.shape[1], level0)
            if cand is not None and cand[0][int(c0[1]), int(c0[0])]:
                first = cand
                break
        if first is None:
            print(f"  {vid}: could not find a dark blob at the {name} seed")
            continue
        radius0 = np.sqrt(first[2] / np.pi)
        for order in (range(f0, len(files)), range(f0 - 1, -1, -1)):
            centre, level, lost = first[1].copy(), first[3], 0
            for i in order:
                taken = (out[i] > 0).astype(np.uint8)
                hit = dark_blob(gray(i), centre, radius0, level, exclude=taken, lost=lost)
                lost += 1
                if hit is not None:
                    lost = 0
                    m, c, area, lv = hit
                    out[i][m > 0] = OBJ[name]
                    centre = 0.5 * centre + 0.5 * c  # light smoothing of the track
                    level = 0.9 * level + 0.1 * lv  # follow slow gain changes
    return out


def compare(vid, pred, min_px_frac=0.0006):
    res = {}
    labelled = sorted((DS / "masks" / vid).glob("*.png"))
    for name, c in OBJ.items():
        dice, conf = [], np.zeros((2, 2), int)
        for mp in labelled:
            t = cv2.imread(str(mp), 0)
            p = pred[int(mp.stem)]
            thr = min_px_frac * t.size
            ta, pa = (t == c).sum(), (p == c).sum()
            conf[int(ta >= thr), int(pa >= thr)] += 1
            if ta >= thr:
                dice.append(2 * ((t == c) & (p == c)).sum() / (ta + pa))
        res[name] = {"dice_when_present": round(float(np.mean(dice)), 3) if dice else None,
                     "n_present": int(conf[1].sum()), "n_absent": int(conf[0].sum()),
                     "missed": int(conf[1, 0]), "false_alarm": int(conf[0, 1])}
    return res


def main():
    prompts = json.loads((HERE / "prompts.json").read_text())
    for vid in sys.argv[1:] or [v for v in prompts if (DS / "masks" / v).exists()]:
        pred = track(vid, prompts[vid])
        d = DS / "classical" / vid
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for i, m in pred.items():
            cv2.imwrite(str(d / f"{i:05d}.png"), m)
        print(vid, json.dumps(compare(vid, pred)))


if __name__ == "__main__":
    main()
