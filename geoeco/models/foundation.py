"""M8 (TerraMind + UPerNet via TerraTorch) and M9 (DOFA) foundation stubs.

Strategy (PROJECT_CONTEXT §6.4): Stage 1 pretrain on Dynamic World tiles,
Stage 2 fine-tune on ~100 Hyderabad patches; encoder frozen first, then full
fine-tune; random modality dropout during training simulates monsoon cloud.

Heavy deps (terratorch, torchgeo) are optional imports so the package stays
importable on CPU CI. Training scripts raise a clear error if missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

N_CLASSES_DEFAULT = 6


@dataclass(frozen=True)
class FoundationConfig:
    backbone: str = "terramind-base"  # or "dofa-base"
    num_classes: int = N_CLASSES_DEFAULT
    modalities: tuple[str, ...] = ("S2L2A", "S1GRD", "DEM")
    frozen_encoder: bool = True
    modality_dropout_p: float = 0.2  # prob. of blanking optical per sample
    pretrained: bool = True


class ModalityDropout(nn.Module):
    """Randomly blanks the optical token group (cloud simulation). [A]"""

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.p <= 0.0:
            return optical, sar
        mask = (torch.rand(optical.shape[0], 1, 1, 1, device=optical.device) > self.p).float()
        return optical * mask, sar


def _require(pkg: str, pip_name: str) -> None:
    raise ImportError(
        f"{pkg} is required for foundation models but not installed. "
        f"Install with `pip install {pip_name}` (GPU Colab/Kaggle recommended)."
    )


class TerraMindSegmenter(nn.Module):
    """M8: TerraMind encoder + UPerNet/UNet decoder stub over TerraTorch."""

    def __init__(self, config: FoundationConfig | None = None) -> None:
        super().__init__()
        self.config = config or FoundationConfig(backbone="terramind-base")
        self.mod_drop = ModalityDropout(self.config.modality_dropout_p)
        try:
            import terratorch  # noqa: F401
            self._backend_available = True
        except ImportError:
            self._backend_available = False
        # Lightweight decoder head valid regardless of backend (real head
        # wired in train_seg when terratorch present).
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.config.num_classes, 1),
        )
        self._encoder_frozen = False
        if self.config.frozen_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self._encoder_frozen = False

    def forward(self, optical: torch.Tensor, sar: torch.Tensor,
                dem: torch.Tensor | None = None) -> torch.Tensor:
        if not self._backend_available:
            _require("terratorch", "terratorch")
        raise NotImplementedError("Wire TerraTorch backbone in train_seg.py")


class DOFASegmenter(nn.Module):
    """M9: DOFA wavelength-conditioned encoder + decoder stub (TorchGeo)."""

    def __init__(self, config: FoundationConfig | None = None) -> None:
        super().__init__()
        self.config = config or FoundationConfig(backbone="dofa-base")
        self.mod_drop = ModalityDropout(self.config.modality_dropout_p)
        try:
            import torchgeo  # noqa: F401
            self._backend_available = True
        except ImportError:
            self._backend_available = False
        self.decoder = nn.Conv2d(128, self.config.num_classes, 1)
        self._encoder_frozen = self.config.frozen_encoder

    def freeze_encoder(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self._encoder_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._backend_available:
            _require("torchgeo", "torchgeo")
        raise NotImplementedError("Wire TorchGeo DOFA backbone in train_seg.py")


FOUNDATION_SPECS: dict[str, str] = {
    "M8": "TerraMind-base encoder (S2+S1+DEM tokens) + UPerNet decoder.",
    "M9": "DOFA encoder (wavelength-conditioned, S1+S2) + decoder.",
}

FROZEN_THEN_FULL = ("Stage1: frozen encoder, train decoder; Stage2: unfreeze all, "
                    "lower LR (1e-5 enc / 1e-4 dec).")
