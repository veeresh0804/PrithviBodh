"""Tiled inference: 256px tiles, 32px overlap, Gaussian blending.

Streams arbitrarily large rasters through a torch segmentation model,
blending overlapping logits with a 2-D Gaussian kernel (no seams).
Outputs class map (argmax uint8) + confidence map (max prob 0-100 uint8).
Throughput target (PROJECT_CONTEXT §6.5): Hyderabad ~36M px in minutes/T4.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class TilingConfig:
    tile: int = 256
    overlap: int = 32
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def gaussian_weight(tile: int, overlap: int) -> np.ndarray:
    """2-D Gaussian kernel peaking at tile centre, ~0 at edges."""
    ax = np.linspace(-1, 1, tile)
    xx, yy = np.meshgrid(ax, ax)
    sigma = 1.0 - overlap / tile
    w = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)
    return w / w.max()


@torch.no_grad()
def predict_tiled(
    image: np.ndarray,
    model: torch.nn.Module,
    config: TilingConfig | None = None,
    num_classes: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """image: (C,H,W) float32. Returns (labels HxW uint8, confidence HxW uint8)."""
    cfg = config or TilingConfig()
    _, H, W = image.shape
    stride = cfg.tile - cfg.overlap
    device = torch.device(cfg.device)
    model = model.to(device).eval()
    kernel = gaussian_weight(cfg.tile, cfg.overlap)
    logit_acc = np.zeros((num_classes, H, W), dtype=np.float64)
    w_acc = np.zeros((H, W), dtype=np.float64)

    tiles: list[tuple[int, int, np.ndarray]] = []
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            r1, c1 = min(r, H - cfg.tile), min(c, W - cfg.tile)
            r1, c1 = max(r1, 0), max(c1, 0)
            tiles.append((r1, c1, image[:, r1:r1 + cfg.tile, c1:c1 + cfg.tile]))
    # Deduplicate edge-clamped tiles.
    seen, unique = set(), []
    for r1, c1, t in tiles:
        if (r1, c1) not in seen:
            seen.add((r1, c1))
            unique.append((r1, c1, t))

    for i in range(0, len(unique), cfg.batch_size):
        batch = unique[i:i + cfg.batch_size]
        inp = torch.from_numpy(np.stack([t for _, _, t in batch])).to(device)
        logits = model(inp).detach().cpu().numpy()  # (B,K,h,w)
        for (r1, c1, _), lg in zip(batch, logits):
            h, w = lg.shape[1], lg.shape[2]
            k = kernel[:h, :w]
            logit_acc[:, r1:r1 + h, c1:c1 + w] += lg * k
            w_acc[r1:r1 + h, c1:c1 + w] += k

    mean_logits = logit_acc / np.maximum(w_acc, 1e-9)
    prob = torch.softmax(torch.from_numpy(mean_logits), dim=0).numpy()
    labels = prob.argmax(axis=0).astype(np.uint8)
    confidence = (prob.max(axis=0) * 100).round().astype(np.uint8)
    return labels, confidence


def write_maps(labels: np.ndarray, confidence: np.ndarray, out_dir: Path,
               prefix: str = "lc") -> tuple[Path, Path]:
    """Save raw arrays; GeoTIFF + COG conversion handled by export pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lp, cp = out_dir / f"{prefix}_labels.npy", out_dir / f"{prefix}_confidence.npy"
    np.save(lp, labels)
    np.save(cp, confidence)
    return lp, cp
