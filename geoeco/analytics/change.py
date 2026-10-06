"""Post-classification change: 2019 vs 2025, confidence-masked.

- Masks pixels where either year's max-prob confidence < threshold
  (reduces false change, §6.6).
- Transition matrix in hectares (10 m pixel = 0.01 ha).
- Change Vector Analysis (CVA) on fused features as second opinion;
  disagreements with post-classification change are flagged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PIXEL_HA = 0.01  # 10 m x 10 m
NODATA_CHANGE = 255


@dataclass(frozen=True)
class ChangeConfig:
    confidence_threshold: int = 60  # 0-100 scale
    cva_threshold_quantile: float = 0.8


def confidence_masked_change(
    labels_2019: np.ndarray,
    labels_2025: np.ndarray,
    conf_2019: np.ndarray,
    conf_2025: np.ndarray,
    config: ChangeConfig | None = None,
) -> np.ndarray:
    """Per-pixel change code from*6+to; low-confidence pixels -> NODATA_CHANGE."""
    cfg = config or ChangeConfig()
    valid = (conf_2019 >= cfg.confidence_threshold) & (conf_2025 >= cfg.confidence_threshold)
    code = (labels_2019.astype(np.int16) * 6 + labels_2025.astype(np.int16)).astype(np.int16)
    out = np.where(valid, code, NODATA_CHANGE).astype(np.uint8 if code.max() < 256 else np.int16)
    # keep 255 sentinel even for int16 path
    out = np.where(valid, code, NODATA_CHANGE)
    return out.astype(np.int16)


def transition_matrix_ha(change: np.ndarray, num_classes: int = 6) -> pd.DataFrame:
    """Cross-tabulate change codes -> hectares (excludes masked pixels)."""
    valid = change[change != NODATA_CHANGE]
    counts = pd.Series(valid.ravel()).value_counts()
    rows = []
    for code in range(num_classes * num_classes):
        n = int(counts.get(code, 0))
        rows.append({"from_class": code // num_classes, "to_class": code % num_classes,
                     "pixels": n, "area_ha": n * PIXEL_HA})
    return pd.DataFrame(rows)


def cva_magnitude_direction(feats_2019: np.ndarray, feats_2025: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Change Vector Analysis: magnitude + direction (L2) over fused features.

    Args: (C,H,W) float arrays. Returns (magnitude HxW, direction HxW radians
    of first-two-PC projection proxy = arctan2 of first two channels' deltas).
    """
    delta = feats_2025.astype(float) - feats_2019.astype(float)
    magnitude = np.linalg.norm(delta, axis=0)
    direction = np.arctan2(delta[1], delta[0] + 1e-9)
    return magnitude, direction


def cva_second_opinion(change: np.ndarray, magnitude: np.ndarray,
                       quantile: float = 0.8) -> np.ndarray:
    """Flag pixels where post-classif change disagrees with CVA magnitude.

    Returns bool mask: post-classif says change but CVA magnitude is low
    (below quantile), or vice versa.
    """
    n = change.shape[0] * change.shape[1]
    n_classes = 6
    said_change = (change != NODATA_CHANGE) & ((change // n_classes) != (change % n_classes))
    thr = float(np.quantile(magnitude, quantile))
    cva_change = magnitude >= thr
    return said_change != cva_change
