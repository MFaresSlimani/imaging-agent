"""Real-data evaluation on CHASE_DB1.

Part 1  Does uncertainty predict segmentation error?
        Each calib/test image is segmented clean and at 3 degradation strengths.
        Thresholds are fitted on the calib subjects only, then applied to test.
Part 2  Agent evaluation: every test image x 3 scenarios
        (stable, simulated rarefaction, rarefaction + poor-quality follow-up).

    python scripts/evaluate_chase.py --checkpoint checkpoints/unet_r34_chase.pt
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_agent import tools  # noqa: E402
from imaging_agent.agent import Agent  # noqa: E402
from imaging_agent.data import load_chase, remove_thin_vessels  # noqa: E402
from imaging_agent.model import Segmenter  # noqa: E402
from imaging_agent.synthetic import degrade, shift  # noqa: E402
from imaging_agent.tools import Case, _uncertainty  # noqa: E402

STRENGTHS = [0.0, 0.4, 0.7, 1.0]
BAD_DICE = 0.70  # a segmentation below this is not usable for measurement


def dice(prob, gt, fov):
    p, g = (prob > 0.5) & fov, gt & fov
    return float(2 * (p & g).sum() / max(p.sum() + g.sum(), 1))


def score_images(seg, samples, seed0):
    rows = []
    for i, s in enumerate(samples):
        for st in STRENGTHS:
            img = s["image"] if st == 0 else degrade(s["image"], seed0 + i, st)
            prob, std, _ = seg.predict(img, tta="basic")
            u = _uncertainty(prob, std, s["fov"])
            rows.append({"name": s["name"], "strength": st, "dice": dice(prob, s["gt"], s["fov"]), **u})
    return rows


def best_threshold(values, bad):
    """Youden's J: threshold that best separates bad from good segmentations."""
    best = (-1, None)
    for t in sorted(set(values)):
        flag = np.array(values) > t
        tpr = (flag & bad).sum() / max(bad.sum(), 1)
        fpr = (flag & ~bad).sum() / max((~bad).sum(), 1)
        best = max(best, (tpr - fpr, t))
    return best[1]


def auroc(score, bad):
    pos, neg = np.array(score)[bad], np.array(score)[~bad]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def make_followup(s, scenario, seed):
    img, gt = s["image"], s["gt"]
    if scenario != "stable":
        img, gt = remove_thin_vessels(img, gt, s["fov"], seed)
    else:
        img = np.clip(img * 1.05 + 0.01, 0, 1).astype(np.float32)  # different exposure
    true_gt = gt  # the retina's state at follow-up, before the camera shift
    dy, dx = np.random.default_rng(seed).uniform(-15, 15, 2).round(1)
    img = shift(img, dy, dx)
    if scenario == "poor_quality":
        img = degrade(img, seed, 1.0)
    return img, true_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CHASE_DB1")
    ap.add_argument("--checkpoint", default="checkpoints/unet_r34_chase.pt")
    ap.add_argument("--out", default="runs/chase")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seg = Segmenter(a.checkpoint, arch="smp-unet")
    calib, test = load_chase(a.data, "calib"), load_chase(a.data, "test")

    # ---------------- Part 1: uncertainty vs error ----------------
    rc, rt = score_images(seg, calib, 100), score_images(seg, test, 200)
    bad_c = np.array([r["dice"] < BAD_DICE for r in rc])
    t_std = best_threshold([r["mean_std"] for r in rc], bad_c)
    t_amb = best_threshold([r["ambiguous_ratio"] for r in rc], bad_c)
    tools.POLICY.update({"max_uncertainty": round(float(t_std), 4), "max_ambiguous_ratio": round(float(t_amb), 4)})

    bad_t = np.array([r["dice"] < BAD_DICE for r in rt])
    flag_t = np.array([r["mean_std"] > t_std or r["ambiguous_ratio"] > t_amb for r in rt])
    dices = np.array([r["dice"] for r in rt])
    part1 = {
        "n_calib": len(rc), "n_test": len(rt),
        "fitted_policy": dict(tools.POLICY),
        "test_spearman_std_vs_error": float(spearmanr([r["mean_std"] for r in rt], 1 - dices)[0]),
        "test_auroc_std_detects_bad": auroc([r["mean_std"] for r in rt], bad_t),
        "test_bad_caught": f"{int((flag_t & bad_t).sum())}/{int(bad_t.sum())}",
        "test_good_wrongly_deferred": f"{int((flag_t & ~bad_t).sum())}/{int((~bad_t).sum())}",
        "test_mean_dice_accepted": float(dices[~flag_t].mean()) if (~flag_t).any() else None,
        "test_mean_dice_deferred": float(dices[flag_t].mean()) if flag_t.any() else None,
        "per_strength": {str(st): {"dice": float(np.mean([r["dice"] for r in rt if r["strength"] == st])),
                                   "mean_std": float(np.mean([r["mean_std"] for r in rt if r["strength"] == st]))}
                         for st in STRENGTHS},
    }
    print(json.dumps(part1, indent=2), flush=True)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for st, c in zip(STRENGTHS, ["#2b6cb0", "#68a0d8", "#e08a3c", "#c0392b"]):
        rr = [r for r in rt if r["strength"] == st]
        ax.scatter([r["mean_std"] for r in rr], [r["dice"] for r in rr], c=c, label=f"degradation {st}", s=28)
    ax.axvline(t_std, ls="--", c="k", lw=1)
    ax.axhline(BAD_DICE, ls=":", c="gray", lw=1)
    ax.text(t_std, 0.02, " defer ->", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_xlabel("mean predictive std (2x TTA)")
    ax.set_ylabel("Dice vs manual annotation")
    ax.set_title("CHASE_DB1 test: uncertainty tracks error", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "uncertainty_vs_dice.png", dpi=130)
    plt.close(fig)

    # ---------------- Part 1b: repeatability on stable calib pairs ----------------
    drift = []
    for i, s in enumerate(calib):
        img1, _ = make_followup(s, "stable", 500 + i)
        case = Case({"baseline": s["image"], "followup": img1}, s["fov"])
        for v in ("baseline", "followup"):
            tools.segment(case, seg, v)
        tools.register(case)
        tools.measure(case, "baseline")
        tools.measure(case, "followup")
        drift.append(abs(tools.compare(case)["changes"]["density"]["relative_change"]))
    tools.POLICY["min_detectable_change"] = round(float(1.5 * max(drift)), 4)
    part1["stable_calib_abs_drift"] = drift
    part1["fitted_policy"] = dict(tools.POLICY)
    print("repeatability limit:", tools.POLICY["min_detectable_change"], "from drifts", drift, flush=True)

    # ---------------- Part 2: agent on longitudinal pairs ----------------
    rows = []
    for i, s in enumerate(test):
        for scenario in ("stable", "progression", "poor_quality"):
            img1, gt1 = make_followup(s, scenario, 1000 + i)
            case = Case({"baseline": s["image"], "followup": img1}, s["fov"])
            res = Agent(case, seg, planner="rules").run()
            d = res["decision"]
            true_change = gt1.sum() / max(s["gt"].sum(), 1) - 1
            cmp = next((t["result"] for t in res["trace"] if t["tool"] == "compare"), None)
            rows.append({"image": s["name"], "scenario": scenario, "decision": d["decision"],
                         "significant": cmp["change_beyond_uncertainty"] if cmp else None,
                         "measured_density_change": cmp["changes"]["density"]["relative_change"] if cmp else None,
                         "true_vessel_area_change": round(float(true_change), 4)})
            if i == 0:
                (out / f"trace_{s['name']}_{scenario}.json").write_text(json.dumps(res, indent=2, default=str))
            print(rows[-1], flush=True)

    def correct(r):
        if r["scenario"] == "stable":
            return r["decision"] == "report" and not r["significant"]
        if r["scenario"] == "progression":
            return r["decision"] == "report" and r["significant"] and r["measured_density_change"] < 0
        return r["decision"] == "defer"

    summary = {sc: f"{sum(correct(r) for r in rows if r['scenario'] == sc)}/{sum(r['scenario'] == sc for r in rows)}"
               for sc in ("stable", "progression", "poor_quality")}
    prog = [r for r in rows if r["scenario"] == "progression" and r["measured_density_change"] is not None]
    summary["progression_mean_abs_error_pct_points"] = round(float(np.mean(
        [abs(r["measured_density_change"] - r["true_vessel_area_change"]) for r in prog])) * 100, 2) if prog else None
    print("AGENT:", json.dumps(summary), flush=True)
    (out / "results.json").write_text(json.dumps({"part1": part1, "agent_summary": summary, "agent_rows": rows},
                                                 indent=2, default=float))


if __name__ == "__main__":
    main()
