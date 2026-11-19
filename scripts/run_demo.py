"""Run the agent on three synthetic cases and save traces + figures.

    python scripts/run_demo.py --checkpoint checkpoints/small_unet.pt
    python scripts/run_demo.py --planner claude        # needs ANTHROPIC_API_KEY
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_agent.agent import Agent  # noqa: E402
from imaging_agent.model import Segmenter  # noqa: E402
from imaging_agent.synthetic import make_case  # noqa: E402
from imaging_agent.tools import Case  # noqa: E402

SCENARIOS = [("stable", 7), ("progression", 11), ("poor_quality", 11)]


def figure(case, data, decision, path):
    fig, ax = plt.subplots(2, 3, figsize=(10, 6.8))
    for row, v in enumerate(("baseline", "followup")):
        ax[row, 0].imshow(case.images[v])
        ax[row, 0].set_title(f"{v} image")
        seg = case.seg.get(v)
        if seg:
            ax[row, 1].imshow(seg["prob"], cmap="gray", vmin=0, vmax=1)
            ax[row, 1].set_title("vessel probability")
            ax[row, 2].imshow(seg["std"], cmap="magma", vmin=0, vmax=0.3)
            ax[row, 2].set_title(f"uncertainty (mean std {seg['mean_std']:.3f})")
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"[{data['scenario']}] {decision['decision'].upper()}: {decision['summary']}", fontsize=9, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/small_unet.pt")
    ap.add_argument("--arch", default="small-unet")
    ap.add_argument("--planner", default="auto", choices=["auto", "rules", "claude"])
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    seg = Segmenter(a.checkpoint, a.arch)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for scenario, seed in SCENARIOS:
        data = make_case(seed, scenario)
        case = Case({"baseline": data["baseline"], "followup": data["followup"]}, data["fov"])
        res = Agent(case, seg, planner=a.planner).run()
        res["ground_truth"] = {"scenario": scenario, "true_shift": data["true_shift"],
                               "true_density": {v: float(data["gt"][v].sum() / data["fov"].sum())
                                                for v in ("baseline", "followup")}}
        (out / f"trace_{scenario}.json").write_text(json.dumps(res, indent=2))
        figure(case, data, res["decision"], out / f"case_{scenario}.png")
        steps = " -> ".join(t["tool"] + (f"({t['args'].get('visit', '')},{t['args'].get('tta', '')})"
                                          if t["tool"] == "segment" else "") for t in res["trace"])
        print(f"\n[{scenario}] planner={res['planner']}\n  steps: {steps}\n"
              f"  {res['decision']['decision'].upper()}: {res['decision']['summary']}")


if __name__ == "__main__":
    main()
