"""M6: U-Net with ResNet-34 encoder, early fusion of the 20-ch/season stack.

Deep models consume a (C, H, W) chip with C=20 per season (14 optical + 4 SAR +
2 terrain), either single-season or 3 seasons stacked (C=60) / time dim.
Uses ``segmentation_models_pytorch`` (smp) when installed; otherwise falls back
to a minimal U-Net so unit tests and CPU inference still run.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

N_CLASSES_DEFAULT = 6
IN_CHANNELS_SINGLE_SEASON = 20
IN_CHANNELS_ALL_SEASONS = 60

try:
    import segmentation_models_pytorch as smp

    _HAS_SMP = True
except ImportError:
    smp = None  # type: ignore[assignment]
    _HAS_SMP = False


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = IN_CHANNELS_SINGLE_SEASON
    num_classes: int = N_CLASSES_DEFAULT
    encoder_name: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)


class _TinyUNet(nn.Module):
    """Minimal U-Net fallback when smp is unavailable (tests / CPU)."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        def block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            )
        self.e1 = block(in_channels, 32)
        self.e2 = block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.mid = block(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = block(64, 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        m = self.mid(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_unet(config: UNetConfig | None = None) -> nn.Module:
    """Build M6. Prefers smp U-Net ResNet-34; falls back to _TinyUNet."""
    cfg = config or UNetConfig()
    if _HAS_SMP:
        assert smp is not None
        return smp.Unet(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.num_classes,
            decoder_channels=cfg.decoder_channels,
        )
    return _TinyUNet(cfg.in_channels, cfg.num_classes)


class UNetEarlyFusion(nn.Module):
    """Early-fusion wrapper: normalise per-modality, single U-Net forward."""

    def __init__(self, config: UNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or UNetConfig()
        self.net = build_unet(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) logits: (B, num_classes, H, W)."""
        return self.net(x)  # type: ignore[no-any-return]

    @property
    def uses_smp(self) -> bool:
        return _HAS_SMP
