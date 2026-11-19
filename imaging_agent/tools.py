"""Tools the agent can call. Each returns a small JSON-serializable summary;
heavy arrays stay in the Case object so the LLM never sees raw pixels."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.registration import phase_cross_correlation

# Thresholds live here so the trace can record exactly which policy was applied.
POLICY = {
    "max_uncertainty": 0.03,      # mean predictive std on candidate vessel pixels (clean ~0.005-0.016, degraded ~0.044)
    "max_ambiguous_ratio": 0.10,  # ambiguous pixels (0.3<p<0.7) per confident vessel pixel
    "min_registration_ncc": 0.80, # normalized cross-correlation after alignment
}


@dataclass
class Case:
    images: dict[str, np.ndarray]
    fov: np.ndarray
    seg: dict[str, dict] = field(default_factory=dict)
    shift: tuple[float, float] | None = None
    metrics: dict[str, dict] = field(default_factory=dict)


def _uncertainty(prob, std, fov):
    cand = (prob > 0.2) & fov
    confident = ((prob > 0.5) & fov).sum()
    ambiguous = ((prob > 0.3) & (prob < 0.7) & fov).sum()
    return {
        "mean_std": round(float(std[cand].mean()) if cand.any() else 0.0, 4),
        "ambiguous_ratio": round(float(ambiguous / max(confident, 1)), 4),
    }


def segment(case: Case, segmenter, visit: str, tta: str = "basic", mc_passes: int = 1) -> dict:
    prob, std, n = segmenter.predict(case.images[visit], tta=tta, mc_passes=mc_passes)
    unc = _uncertainty(prob, std, case.fov)
    case.seg[visit] = {"prob": prob, "std": std, **unc}
    ok = (unc["mean_std"] <= POLICY["max_uncertainty"]
          and unc["ambiguous_ratio"] <= POLICY["max_ambiguous_ratio"])
    return {"visit": visit, "tta": tta, "mc_passes": mc_passes, "n_samples": n,
            **unc, "within_policy": bool(ok)}


def register(case: Case, ref: str = "baseline", mov: str = "followup") -> dict:
    a, b = case.images[ref][..., 1], case.images[mov][..., 1]
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=10)
    aligned = ndi.shift(b, shift, order=1)
    m = case.fov & (ndi.shift(case.fov.astype(float), shift, order=0) > 0.5)
    x, y = a[m] - a[m].mean(), aligned[m] - aligned[m].mean()
    ncc = float((x * y).sum() / (np.sqrt((x ** 2).sum() * (y ** 2).sum()) + 1e-8))
    case.shift = (float(shift[0]), float(shift[1]))
    return {"shift_px": [round(s, 2) for s in case.shift], "ncc": round(ncc, 4),
            "within_policy": ncc >= POLICY["min_registration_ncc"]}


def _vessel_metrics(mask, region):
    area = mask[region].sum()
    length = skeletonize(mask & region).sum()
    return {"density": float(area / region.sum()), "length_px": int(length),
            "mean_width_px": float(area / max(length, 1))}


def measure(case: Case, visit: str) -> dict:
    """Vessel metrics with an uncertainty band: computed on the mean prediction,
    and on the pessimistic (p - std) and optimistic (p + std) masks."""
    if visit not in case.seg:
        return {"error": f"segment {visit} first"}
    prob, std = case.seg[visit]["prob"], case.seg[visit]["std"]
    if visit != "baseline" and case.shift is not None:  # bring into baseline frame
        prob, std = ndi.shift(prob, case.shift, order=1), ndi.shift(std, case.shift, order=1)
    region = case.fov.copy()
    if case.shift is not None:
        region &= ndi.shift(case.fov.astype(float), case.shift, order=0) > 0.5
    point = _vessel_metrics(prob > 0.5, region)
    low = _vessel_metrics(prob - std > 0.5, region)
    high = _vessel_metrics(prob + std > 0.5, region)
    out = {k: {"value": round(point[k], 5), "band": [round(low[k], 5), round(high[k], 5)]}
           for k in ("density", "length_px")}
    case.metrics[visit] = out
    return {"visit": visit, **out}


def compare(case: Case, ref: str = "baseline", mov: str = "followup") -> dict:
    if ref not in case.metrics or mov not in case.metrics:
        return {"error": "measure both visits first"}
    out = {}
    for k in ("density", "length_px"):
        a, b = case.metrics[ref][k], case.metrics[mov][k]
        rel = (b["value"] - a["value"]) / max(a["value"], 1e-8)
        overlap = not (b["band"][1] < a["band"][0] or b["band"][0] > a["band"][1])
        out[k] = {"relative_change": round(rel, 4), "bands_overlap": overlap}
    significant = not any(v["bands_overlap"] for v in out.values())
    direction = "decrease" if out["density"]["relative_change"] < 0 else "increase"
    return {"changes": out, "change_beyond_uncertainty": significant,
            "summary": f"{direction} in vessel density of {abs(out['density']['relative_change']):.1%}"
                       + ("" if significant else " (within measurement uncertainty)")}
