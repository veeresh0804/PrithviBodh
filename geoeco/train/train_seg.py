"""Train segmentation models (M6/M7/M8/M9) with PyTorch Lightning.

Loss: weighted cross-entropy + Dice. Augmentations: flips, rot90, brightness
jitter on optical channels ONLY, random modality dropout (blank optical ->
SAR fallback, simulates monsoon cloud). Two stages: Stage 1 pretrain on DW
tiles, Stage 2 fine-tune on Hyderabad patches. Mixed precision via
``Trainer(precision='16-mixed')``. MLflow logging via MLFlowLogger.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import Dataset

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import MLFlowLogger
    _HAS_PL = True
except ImportError:
    pl = None  # type: ignore[assignment]
    MLFlowLogger = None  # type: ignore[assignment,misc]
    _HAS_PL = False

from geoeco.models.dual_branch import DualBranchUNet
from geoeco.models.unet import UNetConfig, UNetEarlyFusion

Stage = Literal["dw_pretrain", "hyd_finetune"]


# ---------------------------------------------------------------- losses
class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)
        tgt = torch.nn.functional.one_hot(target, prob.shape[1]).permute(0, 3, 1, 2).float()
        inter = (prob * tgt).sum(dim=(2, 3))
        denom = prob.sum(dim=(2, 3)) + tgt.sum(dim=(2, 3)) + 1e-6
        return 1.0 - (2 * inter / denom).mean()


class WeightedCEPlusDice(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None, dice_w: float = 0.5) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()
        self.dice_w = dice_w

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (1 - self.dice_w) * self.ce(logits, target) + self.dice_w * self.dice(logits, target)


# --------------------------------------------------------------- dataset
@dataclass
class ChipDataset(Dataset):
    """Loads (image.pt, mask.pt) pairs; images are (C,H,W) float32, masks (H,W) int64."""

    chips: list[tuple[Path, Path]]
    optical_channels: int = 14  # brightness jitter applies to these only
    modality_dropout_p: float = 0.0
    train: bool = True

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_p, mask_p = self.chips[i]
        img = torch.load(img_p).float()
        mask = torch.load(mask_p).long()
        if self.train:
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
            if random.random() < 0.5:
                img = torch.flip(img, dims=[1])
            k = random.choice([0, 1, 2, 3])
            img = torch.rot90(img, k, dims=[1, 2])
            if random.random() < 0.5:  # brightness jitter optical-only
                img[: self.optical_channels] *= random.uniform(0.9, 1.1)
            if random.random() < self.modality_dropout_p:  # cloud simulation
                img[: self.optical_channels] = 0.0
        return img, mask


if _HAS_PL:
    class SegLightningModule(pl.LightningModule):  # type: ignore[no-redef]
        def __init__(self, model_name: str = "M6", num_classes: int = 6,
                     lr_enc: float = 1e-4, lr_dec: float = 1e-3,
                     class_weights: list[float] | None = None) -> None:
            super().__init__()
            self.save_hyperparameters()
            if model_name == "M7":
                self.net: nn.Module = DualBranchUNet()
            else:
                self.net = UNetEarlyFusion(UNetConfig(num_classes=num_classes))
            w = torch.tensor(class_weights) if class_weights else None
            self.loss = WeightedCEPlusDice(class_weights=w)

        def _step(self, batch, stage: str):
            x, y = batch
            if isinstance(self.net, DualBranchUNet):
                logits = self.net.forward_stacked(x)
            else:
                logits = self.net(x)
            # Dual-branch downsamples x1 vs x2: centre-crop target if needed.
            if logits.shape[-2:] != y.shape[-2:]:
                _, _, h, w = logits.shape
                y = y[:, :h, :w]
            loss = self.loss(logits, y)
            self.log(f"{stage}_loss", loss, prog_bar=True)
            return loss

        def training_step(self, batch, batch_idx):  # type: ignore[override]
            return self._step(batch, "train")

        def validation_step(self, batch, batch_idx):  # type: ignore[override]
            self._step(batch, "val")

        def configure_optimizers(self):  # type: ignore[override]
            return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr_dec"])
else:
    class SegLightningModule:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("lightning is required: pip install lightning")


def train(stage: Stage, data_dir: Path, model_name: str = "M6",
          max_epochs: int = 50, batch_size: int = 8,
          ckpt: Path | None = None, experiment: str = "geoeco-seg") -> None:
    """Entry point used by notebooks/CI. Requires lightning + torch."""
    if not _HAS_PL:
        raise ImportError("lightning is required: pip install lightning")
    assert pl is not None
    pairs = [(p, p.with_name(p.name.replace("img_", "mask_"))) for p in data_dir.glob("img_*.pt")]
    ds = ChipDataset(pairs, modality_dropout_p=0.2 if stage == "hyd_finetune" else 0.0)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    module = SegLightningModule(model_name=model_name)
    logger = MLFlowLogger(experiment_name=experiment, run_name=f"{model_name}-{stage}")
    trainer = pl.Trainer(max_epochs=max_epochs, precision="16-mixed", logger=logger)
    trainer.fit(module, loader, ckpt_path=str(ckpt) if ckpt else None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train M6/M7 segmentation models")
    ap.add_argument("--stage", choices=["dw_pretrain", "hyd_finetune"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="M6", choices=["M6", "M7", "M8", "M9"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    train(args.stage, Path(args.data), args.model, args.epochs, args.batch,
          Path(args.ckpt) if args.ckpt else None)


if __name__ == "__main__":
    main()
