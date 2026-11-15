"""Pseudo-label artery / vein masks with SAM 2 video propagation.

Reads click prompts from prompts.json and writes, per video:
  data/vessel_dataset/masks/<vid>/<frame>.png   (0 background, 1 artery, 2 vein)
  data/vessel_dataset/overlays/<vid>.mp4        (for human review)
  data/vessel_dataset/areas/<vid>.csv           (frame, artery_px, vein_px)

prompts.json format, coordinates in cropped-frame pixels:
  "<vid>": {
    "stride": 3,                       # label every Nth frame (memory / redundancy)
    "prompts": [{"frame": 0, "obj": "artery", "points": [[x, y]], "labels": [1]}],
    "absent":  [{"obj": "vein", "from": 500, "to": 900}],   # force empty (inclusive)
    "skip":    [[0, 20]]               # frames excluded from the dataset entirely
  }

Usage: python vessel_ml/sam2_label.py [VIDEO_ID ...]
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "vessel_dataset"
HERE = Path(__file__).parent
OBJ = {"artery": 1, "vein": 2}
COLOR = {1: (0, 0, 255), 2: (255, 0, 0)}  # BGR: artery red, vein blue
MIN_AREA_PX = 12  # smaller blobs are treated as "not visible"
MIN_FRAC, MAX_FRAC = 0.0004, 0.06  # plausible vessel size as a fraction of the image


def in_ranges(i, ranges):
    return any(a <= i <= b for a, b in ranges)


def is_blank(path):
    """Frozen / no-contact frames: the lower two thirds of the image is black."""
    img = cv2.imread(str(path), 0)
    return img[img.shape[0] // 3 :].mean() < 12


def label_video(predictor, image_predictor, vid, cfg):
    failed = []
    frame_dir = DS / "frames" / vid
    all_frames = sorted(frame_dir.glob("*.jpg"))
    stride = cfg.get("stride", 1)
    skip = cfg.get("skip", [])
    prompt_frames = {p["frame"] for p in cfg["prompts"]}
    keep = [
        i
        for i in range(len(all_frames))
        if (i % stride == 0 or i in prompt_frames)
        and not in_ranges(i, skip)
        and not is_blank(all_frames[i])
    ]
    local = {orig: k for k, orig in enumerate(keep)}

    tmp = Path(tempfile.mkdtemp(prefix="sam2_"))
    try:
        for k, orig in enumerate(keep):
            (tmp / f"{k:05d}.jpg").symlink_to(all_frames[orig])
        state = predictor.init_state(
            video_path=str(tmp), offload_video_to_cpu=True, offload_state_to_cpu=True
        )
        grouped = {}
        for p in cfg["prompts"]:
            grouped.setdefault((p["frame"], p["obj"]), {"points": [], "labels": []})
            g = grouped[(p["frame"], p["obj"])]
            g["points"] += p["points"]
            g["labels"] += p.get("labels", [1] * len(p["points"]))
        # A single click on a faint vessel is ambiguous: SAM often answers with the whole
        # ultrasound fan, and that then spreads to neighbouring frames. So each click is
        # first resolved on its own frame, keeping the best candidate that is vessel-sized,
        # and that mask (not the raw click) is what seeds the video propagation.
        stats = {"ok": 0, "no_candidate": 0, "marked_absent": 0, "dropped": 0}
        for (frame, obj), g in sorted(grouped.items()):
            if frame not in local:
                stats["dropped"] += 1  # click on a skipped or blank frame
                continue
            pts = np.array(g["points"], dtype=np.float32)
            labs = np.array(g["labels"], dtype=np.int32)
            img = cv2.cvtColor(cv2.imread(str(all_frames[frame])), cv2.COLOR_BGR2RGB)
            if not labs.any():
                # only "not the vessel" clicks: the user is saying it is absent on this frame
                seed = np.zeros(img.shape[:2], bool)
                stats["marked_absent"] += 1
            else:
                image_predictor.set_image(img)
                masks, scores, _ = image_predictor.predict(point_coords=pts, point_labels=labs, multimask_output=True)
                frac = masks.reshape(len(masks), -1).mean(1)
                pos = pts[labs == 1].astype(int)
                hit = np.array([all(m[y, x] for x, y in pos) for m in masks.astype(bool)])
                good = (frac >= MIN_FRAC) & (frac <= MAX_FRAC) & hit
                if not good.any():
                    stats["no_candidate"] += 1
                    failed.append((frame, obj))
                    continue
                seed = masks[int(np.argmax(np.where(good, scores, -1)))].astype(bool)
                stats["ok"] += 1
            predictor.add_new_mask(state, frame_idx=local[frame], obj_id=OBJ[obj], mask=seed)
        print(f"{vid}: click frames -> {stats}; no vessel-sized mask at: {failed}", flush=True)
        logits = {}
        for reverse in (False, True):
            for k, obj_ids, out in predictor.propagate_in_video(state, reverse=reverse):
                logits[k] = {o: out[j, 0].cpu().numpy() for j, o in enumerate(obj_ids)}
        predictor.reset_state(state)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    mask_dir = DS / "masks" / vid
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    mask_dir.mkdir(parents=True)
    (DS / "overlays").mkdir(exist_ok=True)
    (DS / "areas").mkdir(exist_ok=True)
    h, w = cv2.imread(str(all_frames[0]), 0).shape
    writer = cv2.VideoWriter(
        str(DS / "overlays" / f"{vid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w * 2, h)
    )
    rows = ["frame,artery_px,vein_px"]
    absent = cfg.get("absent", [])
    rejected = 0
    for k, orig in enumerate(keep):
        lg = logits[k]
        stack = np.stack([lg.get(1, np.full((h, w), -99.0)), lg.get(2, np.full((h, w), -99.0))])
        for a in absent:  # suppress before the argmax so the other vessel keeps its pixels
            if a["from"] <= orig <= a["to"]:
                stack[OBJ[a["obj"]] - 1] = -99.0
        mask = np.where(stack.max(0) > 0, stack.argmax(0) + 1, 0).astype(np.uint8)
        for o in (1, 2):
            # a vessel is one compact blob: keep the largest piece, and drop the mask if
            # that piece is tiny, larger than any vessel, or ragged (SAM leaking into tissue)
            n_cc, lab, st, _ = cv2.connectedComponentsWithStats((mask == o).astype(np.uint8))
            if n_cc < 2:
                continue
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            piece = (lab == big).astype(np.uint8)
            n_px = int(piece.sum())
            cnt = max(cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
            solidity = n_px / max(1.0, cv2.contourArea(cv2.convexHull(cnt)))
            bad = n_px < MIN_AREA_PX or n_px > MAX_FRAC * mask.size or (solidity < 0.8 and orig not in prompt_frames)
            rejected += int(bad and n_px >= MIN_AREA_PX)
            mask[mask == o] = 0
            if not bad:
                mask[piece > 0] = o
        cv2.imwrite(str(mask_dir / f"{orig:05d}.png"), mask)
        rows.append(f"{orig},{int((mask == 1).sum())},{int((mask == 2).sum())}")

        img = cv2.imread(str(all_frames[orig]))
        over = img.copy()
        for o in (1, 2):
            cnts, _ = cv2.findContours((mask == o).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(over, cnts, -1, COLOR[o], 1)
        cv2.putText(over, f"f{orig}", (4, h - 6), 0, 0.45, (0, 255, 0), 1)
        if orig in prompt_frames:
            for p in cfg["prompts"]:
                if p["frame"] == orig:
                    for (x, y), lab in zip(p["points"], p.get("labels", [1] * len(p["points"]))):
                        cv2.drawMarker(over, (int(x), int(y)), (0, 255, 0) if lab else (0, 255, 255),
                                       cv2.MARKER_CROSS, 8, 1)
        writer.write(np.hstack([img, over]))
    writer.release()
    (DS / "areas" / f"{vid}.csv").write_text("\n".join(rows) + "\n")
    print(f"{vid}: labelled {len(keep)} frames, {rejected} oversized or ragged masks rejected", flush=True)


def main():
    prompts = json.loads((HERE / "prompts.json").read_text())
    vids = sys.argv[1:] or list(prompts)
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(HERE / "checkpoints" / "sam2.1_hiera_small.pt"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # SAM 2 stores memory features as bfloat16 and expects to run under autocast.
    # We run in float32 (GTX 16xx cards misbehave in half precision), so cast the
    # memory back up before attention; without this some prompt layouts crash.
    attn = predictor.memory_attention.forward

    def attn_f32(**kw):
        return attn(**{k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()})

    predictor.memory_attention.forward = attn_f32
    # By default every clicked frame stays in attention memory. With dozens of clicked
    # frames that fills the 6 GB GPU and slows to ~20 s per frame, so keep only the
    # clicked frames nearest in time to the one being segmented.
    predictor.max_cond_frames_in_attn = 6
    image_predictor = SAM2ImagePredictor(predictor)
    with torch.inference_mode():
        for vid in vids:
            label_video(predictor, image_predictor, vid, prompts[vid])


if __name__ == "__main__":
    main()
