"""M7: dual-branch U-Net — separate optical + SAR encoders, fused decoder.

Feature-level fusion: optical branch (14 ch/season) and SAR branch
(4 ch/season, +2 terrain on SAR side by default) encode independently;
bottleneck features are concatenated (+1x1 fuse) and skip connections from
both branches are concatenated at each decoder stage (shared decoder).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

OPTICAL_CHANNELS = 14  # per season
SAR_CHANNELS = 6  # VV,VH,ratio,VH_std + elevation + slope, per season


@dataclass(frozen=True)
class DualBranchConfig:
    optical_channels: int = OPTICAL_CHANNELS
    sar_channels: int = SAR_CHANNELS
    num_classes: int = 6
    base_width: int = 32


def _conv_block(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class _Encoder(nn.Module):
    def __init__(self, in_ch: int, base: int) -> None:
        super().__init__()
        self.e1 = _conv_block(in_ch, base)
        self.e2 = _conv_block(base, base * 2)
        self.e3 = _conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        return s1, s2, s3


class DualBranchUNet(nn.Module):
    """Two encoders -> fused bottleneck + fused skips -> shared decoder."""

    def __init__(self, config: DualBranchConfig | None = None) -> None:
        super().__init__()
        self.config = config or DualBranchConfig()
        b = self.config.base_width
        self.opt_enc = _Encoder(self.config.optical_channels, b)
        self.sar_enc = _Encoder(self.config.sar_channels, b)
        # Bottleneck fuse: concat(4b+4b) -> 4b via 1x1.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(b * 8, b * 4, 1), nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d3 = _conv_block(b * 2 + b * 4, b * 2)  # + concat of both s2 skips
        self.up2 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d2 = _conv_block(b + b * 2, b)  # + concat of both s1 skips
        self.head = nn.Conv2d(b, self.config.num_classes, 1)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        """optical: (B, Co, H, W); sar: (B, Cs, H, W) -> logits (B, K, H/2, W/2)."""
        o1, o2, o3 = self.opt_enc(optical)
        s1, s2, s3 = self.sar_enc(sar)
        m = self.bottleneck(torch.cat([o3, s3], dim=1))
        d3 = self.d3(torch.cat([self.up3(m), o2, s2], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), o1, s1], dim=1))
        return self.head(d2)

    def forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        """Split a stacked early-fusion chip (optical first) then forward."""
        co = self.config.optical_channels
        return self.forward(x[:, :co], x[:, co:co + self.config.sar_channels])
