"""Train the small U-Net on procedurally generated fundus images.

Only clean acquisitions are used, so degraded images are out of distribution
at test time. That is what lets the uncertainty signal flag them.

    python scripts/train_synthetic.py --steps 1500 --out checkpoints/small_unet.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_agent.model import SmallUNet  # noqa: E402
from imaging_agent.synthetic import make_tree, render  # noqa: E402


def make_bank(n, size, seed0):
    imgs, masks = [], []
    for i in range(n):
        tree = make_tree(seed0 + i, size)
        img, mask, _ = render(tree, seed0 + i, noise=np.random.default_rng(i).uniform(0.01, 0.03))
        imgs.append(img.transpose(2, 0, 1))
        masks.append(mask[None].astype(np.float32))
    return torch.tensor(np.stack(imgs)), torch.tensor(np.stack(masks))


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    return 1 - (2 * (p * y).sum((2, 3)) + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bank", type=int, default=400)
    ap.add_argument("--out", default="checkpoints/small_unet.pt")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}; building {a.bank} training images...")
    X, Y = make_bank(a.bank, a.size, seed0=10_000)  # demo cases use seeds < 10_000
    Xv, Yv = make_bank(32, a.size, seed0=50_000)
    model = SmallUNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.steps)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(X), (a.batch,))
        x, y = X[idx].to(dev), Y[idx].to(dev)
        if torch.rand(1) < 0.5:
            x, y = x.flip(3), y.flip(3)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xv.to(dev))) > 0.5
                yv = Yv.to(dev) > 0.5
                dice = (2 * (p & yv).sum() / (p.sum() + yv.sum())).item()
            model.train()
            print(f"step {step:5d}  loss {loss.item():.4f}  val dice {dice:.4f}  ({time.time() - t0:.0f}s)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "arch": "small-unet", "val_dice": dice}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
