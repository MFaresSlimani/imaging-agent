"""Segmentation backends with uncertainty (test-time augmentation + MC dropout)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, p=0.0):
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if p:
        layers.append(nn.Dropout2d(p))
    return nn.Sequential(*layers)


class SmallUNet(nn.Module):
    """4-level U-Net with dropout in the deep layers (needed for MC dropout)."""

    def __init__(self, cin=3, base=16, p=0.2):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2 = _block(cin, c[0]), _block(c[0], c[1])
        self.e3, self.b = _block(c[1], c[2], p), _block(c[2], c[3], p)
        self.u3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), _block(c[3], c[2], p)
        self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), _block(c[2], c[1])
        self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), _block(c[1], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def build_model(arch: str = "small-unet") -> nn.Module:
    if arch == "small-unet":
        return SmallUNet()
    if arch == "smp-unet":
        # Same setup as the A2I-26 comparative study: U-Net, ResNet-34 encoder.
        import segmentation_models_pytorch as smp
        return smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    raise ValueError(f"unknown arch {arch}")


# The 8 dihedral transforms; each is (forward, inverse) on NCHW tensors.
_TTA = [
    (lambda x: x, lambda x: x),
    (lambda x: x.flip(3), lambda x: x.flip(3)),
    (lambda x: x.flip(2), lambda x: x.flip(2)),
    (lambda x: x.flip(2).flip(3), lambda x: x.flip(3).flip(2)),
    (lambda x: x.rot90(1, (2, 3)), lambda x: x.rot90(-1, (2, 3))),
    (lambda x: x.rot90(2, (2, 3)), lambda x: x.rot90(-2, (2, 3))),
    (lambda x: x.rot90(3, (2, 3)), lambda x: x.rot90(-3, (2, 3))),
    (lambda x: x.flip(3).rot90(1, (2, 3)), lambda x: x.rot90(-1, (2, 3)).flip(3)),
]
TTA_LEVELS = {"none": 1, "basic": 2, "full": 8}


class Segmenter:
    def __init__(self, checkpoint: str | None = None, arch: str = "small-unet", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch
        self.model = build_model(arch).to(self.device)
        if checkpoint:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state.get("model", state))
        self.model.eval()

    def _enable_mc_dropout(self, on: bool):
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train(on)

    @torch.no_grad()
    def predict(self, img: np.ndarray, tta: str = "basic", mc_passes: int = 1):
        """img HxWx3 float [0,1]. Returns (mean prob HxW, std HxW, n_samples)."""
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(self.device)
        self._enable_mc_dropout(mc_passes > 1)
        samples = []
        for fwd, inv in _TTA[:TTA_LEVELS[tta]]:
            for _ in range(mc_passes):
                samples.append(torch.sigmoid(inv(self.model(fwd(x)))))
        self._enable_mc_dropout(False)
        s = torch.cat(samples, 0)[:, 0]
        return s.mean(0).cpu().numpy(), s.std(0, unbiased=False).cpu().numpy(), len(samples)
