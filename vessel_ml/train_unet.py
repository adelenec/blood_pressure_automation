"""Train a small U-Net (background / artery / vein) on the SAM 2 pseudo-labels.

Usage:
  python vessel_ml/train_unet.py --val 051426_Lee3 --tag holdout_lee3
  python vessel_ml/train_unet.py --tag all            # no hold-out, for deployment

Whole videos are held out, never individual frames: neighbouring frames are
near-duplicates, so a per-frame split would overstate accuracy.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "vessel_dataset"
SIZE = 256  # network input is SIZE x SIZE, aspect ratio is squashed


# ----------------------------------------------------------------------------- model
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    def __init__(self, n_classes=3, width=(16, 32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        cin = 1
        for w in width:
            self.downs.append(block(cin, w))
            cin = w
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for w in reversed(width[:-1]):
            self.ups.append(nn.ConvTranspose2d(cin, w, 2, stride=2))
            self.up_blocks.append(block(2 * w, w))
            cin = w
        self.head = nn.Conv2d(cin, n_classes, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, blk in zip(self.ups, self.up_blocks):
            x = up(x)
            x = blk(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


# ----------------------------------------------------------------------------- data
def list_samples(vids):
    """Labelled frames, minus the "exclude" ranges in prompts.json (frames whose
    pseudo-label is not trusted, e.g. the vessel is faint and SAM dropped it)."""
    prompts = json.loads((Path(__file__).parent / "prompts.json").read_text())
    out = []
    for v in vids:
        excl = prompts.get(v, {}).get("exclude", [])
        for m in sorted((DS / "masks" / v).glob("*.png")):
            if not any(a <= int(m.stem) <= b for a, b in excl):
                out.append((v, DS / "frames" / v / (m.stem + ".jpg"), m))
    return out


def augment(img, mask, rng):
    h, w = img.shape
    # geometry: scale / rotate / shift / flip
    ang = rng.uniform(-15, 15)
    sc = rng.uniform(0.75, 1.3)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += (rng.uniform(-0.12, 0.12) * w, rng.uniform(-0.12, 0.12) * h)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    img = img.astype(np.float32) / 255.0
    # appearance: gain, gamma, blur, speckle, depth-dependent gain, shadow band
    img = img ** rng.uniform(0.6, 1.6)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.4:
        k = rng.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.6:
        noise = np.random.rayleigh(1.0, (h // 2, w // 2)).astype(np.float32)
        noise = cv2.resize(noise, (w, h)) / 1.2533
        a = rng.uniform(0.1, 0.45)
        img = img * ((1 - a) + a * noise)
    if rng.random() < 0.4:
        ramp = np.linspace(rng.uniform(0.7, 1.2), rng.uniform(0.7, 1.2), h, dtype=np.float32)[:, None]
        img = img * ramp
    if rng.random() < 0.15:
        x0 = rng.randint(0, w - 1)
        bw = rng.randint(w // 12, w // 4)
        img[:, x0 : x0 + bw] *= rng.uniform(0.2, 0.6)
    return np.clip(img, 0, 1), np.ascontiguousarray(mask)


class VesselDS(Dataset):
    def __init__(self, samples, train):
        self.samples, self.train = samples, train
        self.rng = random.Random()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        _, fp, mp = self.samples[i]
        img = cv2.resize(cv2.imread(str(fp), 0), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(cv2.imread(str(mp), 0), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train:
            img, mask = augment(img, mask, self.rng)
        else:
            img = img.astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(img))[None], torch.from_numpy(mask).long()


# ----------------------------------------------------------------------------- train
def dice_loss(logits, target, eps=1.0):
    prob = logits.softmax(1)
    onehot = F.one_hot(target, 3).permute(0, 3, 1, 2).float()
    inter = (prob * onehot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps))[1:].mean()


@torch.no_grad()
def evaluate(model, loader, device, min_px=20):
    """Per-class Dice over frames where the class is labelled present, plus
    present / absent agreement with the label (what collapse detection needs)."""
    model.eval()
    dices = {1: [], 2: []}
    conf = {1: np.zeros((2, 2), int), 2: np.zeros((2, 2), int)}  # [label present][pred present]
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for p, t in zip(pred, y):
            for c in (1, 2):
                pa, ta = int((p == c).sum()), int((t == c).sum())
                conf[c][int(ta >= min_px), int(pa >= min_px)] += 1
                if ta >= min_px:
                    dices[c].append(2 * int(((p == c) & (t == c)).sum()) / (pa + ta))
    res = {}
    for c, name in ((1, "artery"), (2, "vein")):
        m = conf[c]
        res[name] = {
            "dice_when_present": float(np.mean(dices[c])) if dices[c] else None,
            "n_present": int(m[1].sum()), "n_absent": int(m[0].sum()),
            "missed": int(m[1, 0]), "false_alarm": int(m[0, 1]),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--iters_per_epoch", type=int, default=250)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    vids = sorted(p.name for p in (DS / "masks").iterdir() if p.is_dir())
    train_vids = [v for v in vids if v not in args.val]
    print("train:", train_vids, "\nval:", args.val)
    tr = list_samples(train_vids)
    # every video contributes equally regardless of its frame count
    counts = {v: sum(1 for s in tr if s[0] == v) for v in train_vids}
    weights = [1.0 / counts[s[0]] for s in tr]
    sampler = WeightedRandomSampler(weights, args.iters_per_epoch * args.bs, replacement=True)
    tl = DataLoader(VesselDS(tr, True), batch_size=args.bs, sampler=sampler, num_workers=8, drop_last=True)
    vl = DataLoader(VesselDS(list_samples(args.val), False), batch_size=32, num_workers=4) if args.val else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallUNet().to(device)
    print("params: %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * args.iters_per_epoch)
    ce_w = torch.tensor([1.0, 3.0, 3.0], device=device)

    out = Path(__file__).parent / "checkpoints" / f"unet_{args.tag}.pt"
    history = []
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=ce_w) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / len(tl)}
        if vl is not None and (ep % 5 == 4 or ep == args.epochs - 1):
            rec["val"] = evaluate(model, vl, device)
        history.append(rec)
        print(json.dumps(rec), flush=True)
    torch.save({"model": model.state_dict(), "size": SIZE, "train_vids": train_vids,
                "val_vids": args.val, "history": history}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
