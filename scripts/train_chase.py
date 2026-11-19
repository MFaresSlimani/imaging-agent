"""Train U-Net (ResNet-34, ImageNet weights) on CHASE_DB1, as in the A2I-26 study.

    python scripts/train_chase.py --data data/CHASE_DB1 --steps 3000
Reports Dice/F1, sensitivity, specificity and accuracy on the test split (inside the FOV).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_agent.data import load_chase  # noqa: E402
from imaging_agent.model import Segmenter, SmpUNet  # noqa: E402


def batch(samples, size, n, rng):
    xs, ys = [], []
    for _ in range(n):
        s = samples[rng.integers(len(samples))]
        h, w = s["gt"].shape
        y0, x0 = rng.integers(0, h - size), rng.integers(0, w - size)
        x = s["image"][y0:y0 + size, x0:x0 + size].transpose(2, 0, 1)
        y = s["gt"][y0:y0 + size, x0:x0 + size][None].astype(np.float32)
        k = rng.integers(4)
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        x = np.clip(x * rng.uniform(0.85, 1.15) + rng.uniform(-0.06, 0.06), 0, 1)  # brightness/contrast
        xs.append(x.copy())
        ys.append(y.copy())
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


def metrics(prob, gt, fov, thr=0.5):
    p, g = (prob > thr)[fov], gt[fov]
    tp, fp, fn, tn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum(), (~p & ~g).sum()
    return {"dice": 2 * tp / (2 * tp + fp + fn), "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp), "accuracy": (tp + tn) / p.size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CHASE_DB1")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--out", default="checkpoints/unet_r34_chase.pt")
    a = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = load_chase(a.data, "train")
    test = load_chase(a.data, "test")
    print(f"device={dev} train={len(train)} test={len(test)}", flush=True)

    model = SmpUNet(pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.steps)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        x, y = batch(train, a.crop, a.batch, rng)
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(dev, enabled=dev == "cuda"):
            logits = model(x)
            p = torch.sigmoid(logits)
            dice = 1 - (2 * (p * y).sum((2, 3)) + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice.mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, a.out)

    seg = Segmenter(a.out, arch="smp-unet")
    rows = []
    for s in test:
        prob, _, _ = seg.predict(s["image"], tta="none")
        rows.append(metrics(prob, s["gt"], s["fov"]))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("TEST (CHASE_DB1 subjects 11-14, inside FOV):", json.dumps({k: round(v, 4) for k, v in mean.items()}))
    Path("runs").mkdir(exist_ok=True)
    Path("runs/chase_test_metrics.json").write_text(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
