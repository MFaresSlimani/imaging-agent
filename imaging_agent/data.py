"""CHASE_DB1 loading and simulated follow-up visits for real images.

Split (fixed, by subject so both eyes of a child stay together):
  train: subjects 01-08 (16 images)
  calib: subjects 09-10 (4 images)   -> used only to set agent thresholds
  test:  subjects 11-14 (8 images)   -> the usual CHASE_DB1 test split
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects

SPLITS = {"train": range(1, 9), "calib": range(9, 11), "test": range(11, 15)}


def load_chase(root: str | Path, split: str):
    root = Path(root)
    out = []
    for s in SPLITS[split]:
        for eye in "LR":
            name = f"Image_{s:02d}{eye}"
            img = np.asarray(Image.open(root / "Images" / f"{name}.jpg").convert("RGB"), np.float32) / 255
            gt = np.asarray(Image.open(root / "Masks" / f"{name}_1stHO.png")) > 0
            fov = fov_mask(img)
            out.append({"name": name, "image": img, "gt": gt & fov, "fov": fov})
    return out


def fov_mask(img: np.ndarray) -> np.ndarray:
    """CHASE_DB1 ships no FOV masks: threshold the red channel and clean up."""
    m = img[..., 0] > 0.08
    m = ndi.binary_fill_holes(ndi.binary_opening(m, structure=disk(5)))
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return ndi.binary_erosion(m, iterations=8)


def remove_thin_vessels(img, gt, fov, seed, fraction=0.5):
    """Simulated rarefaction on a real image: erase a random half of the thin
    vessel segments by filling them from the surrounding retina.
    Returns (new image, new ground-truth mask)."""
    rng = np.random.default_rng(seed)
    thick = ndi.binary_opening(gt, structure=disk(2))
    thin = remove_small_objects(gt & ~thick, max_size=20)
    lab, n = ndi.label(thin, structure=np.ones((3, 3)))
    drop = np.isin(lab, rng.choice(np.arange(1, n + 1), size=int(n * fraction), replace=False))
    region = ndi.binary_dilation(drop, iterations=3) & fov
    # Fill from the surrounding vessel-free retina (normalized convolution), then
    # add back fine texture so the erased area is not suspiciously smooth.
    bg = (fov & ~ndi.binary_dilation(gt, iterations=2)).astype(np.float32)
    fill = np.stack([ndi.gaussian_filter(img[..., c] * bg, 6) / (ndi.gaussian_filter(bg, 6) + 1e-6)
                     for c in range(3)], -1)
    texture = img - np.stack([ndi.gaussian_filter(img[..., c], 2) for c in range(3)], -1)
    texture = ndi.shift(texture, (0, 25, 0), mode="reflect")  # texture borrowed from elsewhere
    alpha = ndi.gaussian_filter(region.astype(np.float32), 1.0)[..., None]
    out = (1 - alpha) * img + alpha * (fill + 0.5 * texture)
    return np.clip(out, 0, 1).astype(np.float32), gt & ~region
