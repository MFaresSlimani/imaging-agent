"""One CHASE_DB1 test eye under the three follow-up scenarios: image, prediction, uncertainty.

    python scripts/figure_chase.py --checkpoint checkpoints/unet_r34_chase.pt
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_chase import make_followup  # noqa: E402
from imaging_agent.data import load_chase  # noqa: E402
from imaging_agent.model import Segmenter  # noqa: E402
from imaging_agent.tools import _uncertainty  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CHASE_DB1")
    ap.add_argument("--checkpoint", default="checkpoints/unet_r34_chase.pt")
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    seg = Segmenter(a.checkpoint, arch="smp-unet")
    s = load_chase(a.data, "test")[a.index]
    rows = [("baseline", s["image"])] + [(sc, make_followup(s, sc, 1000 + a.index)[0])
                                         for sc in ("progression", "poor_quality")]
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for r, (name, img) in enumerate(rows):
        prob, std, _ = seg.predict(img, tta="basic")
        u = _uncertainty(prob, std, s["fov"])
        ax[r, 0].imshow(img)
        ax[r, 0].set_title(f"{s['name']}: {name}", fontsize=9)
        ax[r, 1].imshow(prob, cmap="gray", vmin=0, vmax=1)
        ax[r, 1].set_title("vessel probability", fontsize=9)
        ax[r, 2].imshow(std, cmap="magma", vmin=0, vmax=0.35)
        ax[r, 2].set_title(f"uncertainty, mean std {u['mean_std']:.3f}", fontsize=9)
    for x in ax.flat:
        x.axis("off")
    fig.tight_layout()
    Path("runs/chase").mkdir(parents=True, exist_ok=True)
    fig.savefig("runs/chase/case_figure.png", dpi=90)
    print("saved runs/chase/case_figure.png")


if __name__ == "__main__":
    main()
