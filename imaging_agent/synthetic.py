"""Procedural fundus-like images with known vessel masks.

Used for the demo and for training a small model when no real checkpoint is
available. Not a substitute for real data (DRIVE, CHASE_DB1): it only makes
the agent pipeline runnable and testable end to end, with exact ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import disk


@dataclass
class Segment:
    points: np.ndarray  # (N, 2) float, row/col
    radius: float
    depth: int


@dataclass
class VesselTree:
    size: int
    disc: tuple[float, float]
    segments: list[Segment] = field(default_factory=list)


def _grow(tree, rng, start, angle, radius, depth, max_depth=4):
    if radius < 0.6 or depth > max_depth:
        return
    pts = [np.array(start, dtype=float)]
    length = rng.integers(25, 60) if depth else rng.integers(60, 110)
    for _ in range(length):
        angle += rng.normal(0, 0.12)
        nxt = pts[-1] + 1.6 * np.array([np.sin(angle), np.cos(angle)])
        c = tree.size / 2
        if np.hypot(*(nxt - c)) > tree.size * 0.46:
            break
        pts.append(nxt)
    if len(pts) < 4:
        return
    tree.segments.append(Segment(np.array(pts), radius, depth))
    # branch from a few points along the segment
    for idx in rng.choice(len(pts), size=min(2, len(pts) - 1), replace=False):
        if rng.random() < 0.85:
            side = rng.choice([-1, 1])
            _grow(tree, rng, pts[idx], angle + side * rng.uniform(0.4, 1.0),
                  radius * rng.uniform(0.6, 0.8), depth + 1, max_depth)
    # continue the parent vessel past its end
    _grow(tree, rng, pts[-1], angle, radius * 0.85, depth + 1, max_depth)


def make_tree(seed: int, size: int = 256) -> VesselTree:
    rng = np.random.default_rng(seed)
    disc = (size / 2 + rng.uniform(-10, 10), size * 0.7 + rng.uniform(-8, 8))
    tree = VesselTree(size, disc)
    for k in range(rng.integers(4, 7)):
        angle = 2 * np.pi * k / 5 + rng.normal(0, 0.3)
        _grow(tree, rng, disc, angle, rng.uniform(2.4, 3.2), 0)
    return tree


def render(tree: VesselTree, seed: int = 0, noise: float = 0.02):
    """Return (rgb float32 HxWx3 in [0,1], vessel mask bool, fov mask bool)."""
    s = tree.size
    rng = np.random.default_rng(seed)
    mask = np.zeros((s, s), bool)
    for seg in tree.segments:
        for p in seg.points:
            rr, cc = disk(tuple(p), seg.radius, shape=(s, s))
            mask[rr, cc] = True

    yy, xx = np.mgrid[:s, :s]
    r = np.hypot(yy - s / 2, xx - s / 2)
    fov = r < s * 0.47
    vignette = np.clip(1 - (r / (s * 0.5)) ** 2 * 0.5, 0, 1)
    disc_glow = np.exp(-((yy - tree.disc[0]) ** 2 + (xx - tree.disc[1]) ** 2) / (2 * 9 ** 2))

    vessel_soft = ndi.gaussian_filter(mask.astype(float), 0.8)
    texture = ndi.gaussian_filter(rng.normal(0, 1, (s, s)), 6) * 0.04

    red = 0.75 * vignette + 0.25 * disc_glow + texture - 0.20 * vessel_soft
    green = 0.42 * vignette + 0.45 * disc_glow + texture - 0.28 * vessel_soft
    blue = 0.15 * vignette + 0.20 * disc_glow + texture * 0.5 - 0.08 * vessel_soft
    img = np.stack([red, green, blue], -1)
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 1) * fov[..., None]
    return img.astype(np.float32), mask & fov, fov


def progress(tree: VesselTree, seed: int, prune: float = 0.35, thin: float = 0.85) -> VesselTree:
    """Simulate vessel rarefaction at follow-up: drop small branches, thin the rest."""
    rng = np.random.default_rng(seed)
    out = VesselTree(tree.size, tree.disc)
    for seg in tree.segments:
        if seg.depth >= 3 and rng.random() < prune:
            continue
        out.segments.append(Segment(seg.points.copy(), max(seg.radius * thin, 0.6), seg.depth))
    return out


def shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Simulate a different fixation between visits (translation only)."""
    shifts = (dy, dx, 0) if img.ndim == 3 else (dy, dx)
    return ndi.shift(img, shifts, order=1, mode="constant")


def degrade(img: np.ndarray, seed: int, strength: float = 1.0) -> np.ndarray:
    """Simulate a poor-quality acquisition (media opacity / haze + blur + noise)."""
    rng = np.random.default_rng(seed)
    s = img.shape[0]
    yy, xx = np.mgrid[:s, :s]
    cy, cx = rng.uniform(0.35, 0.65, 2) * s
    haze = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (s * 0.28) ** 2)) * strength
    blurred = np.stack([ndi.gaussian_filter(img[..., c], 2.5 * strength) for c in range(3)], -1)
    out = (1 - haze[..., None]) * img + haze[..., None] * (0.6 * blurred + 0.4 * blurred.mean())
    out += rng.normal(0, 0.05 * strength, out.shape)
    return np.clip(out, 0, 1).astype(np.float32) * (img.sum(-1, keepdims=True) > 0)


def make_case(seed: int, scenario: str = "progression", size: int = 256):
    """Two visits of one eye.

    scenario:
      "progression": follow-up has vessel loss, both images good quality.
      "stable":      follow-up is the same eye, only shifted.
      "poor_quality": follow-up has vessel loss AND a degraded acquisition.
    Returns dict with images, ground-truth masks, fov, and the true shift.
    """
    tree = make_tree(seed, size)
    img0, gt0, fov = render(tree, seed)
    tree1 = tree if scenario == "stable" else progress(tree, seed + 1)
    img1, gt1, _ = render(tree1, seed + 2)
    dy, dx = np.random.default_rng(seed + 3).uniform(-8, 8, 2).round(1)
    img1, gt1 = shift(img1, dy, dx), shift(gt1.astype(float), dy, dx) > 0.5
    if scenario == "poor_quality":
        img1 = degrade(img1, seed + 4)
    return {"baseline": img0, "followup": img1, "gt": {"baseline": gt0, "followup": gt1},
            "fov": fov, "true_shift": (float(dy), float(dx)), "scenario": scenario}
