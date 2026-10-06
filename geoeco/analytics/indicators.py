"""Ecosystem indicators: vegetation, water, built-up (§6.6, FR-09).

- Vegetation: seasonal NDVI/EVI mean + delta, computed WITHIN veg classes only.
- Water: fused rule MNDWI > thr OR SAR VV < thr (SAR covers monsoon cloud).
- Built-up: transitions into built-up + NDBI rise + SAR-texture trend proxy
  (VH temporal std increase).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WATER, TREE, CROP, BUILT, BARE, GRASS = 0, 1, 2, 3, 4, 5
VEG_CLASSES = (TREE, CROP, GRASS)


@dataclass(frozen=True)
class IndicatorThresholds:
    mndwi: float = 0.2
    vv_db: float = -15.0
    ndbi_rise: float = 0.05


def veg_means(index: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Mean index value within each vegetation class; NaN if class absent."""
    return {c: float(np.nanmean(np.where(labels == c, index, np.nan))) for c in VEG_CLASSES}


def veg_delta(index_t1: np.ndarray, index_t2: np.ndarray,
              labels_t2: np.ndarray) -> dict[int, float]:
    """Per-veg-class mean delta (t2 - t1) masked to t2 veg pixels."""
    delta = index_t2.astype(float) - index_t1.astype(float)
    return {c: float(np.nanmean(np.where(labels_t2 == c, delta, np.nan))) for c in VEG_CLASSES}


def water_mask(mndwi: np.ndarray, vv_db: np.ndarray,
               thr: IndicatorThresholds | None = None) -> np.ndarray:
    """Fused water rule: MNDWI>thr OR VV<thr. Returns bool array."""
    t = thr or IndicatorThresholds()
    return (mndwi > t.mndwi) | (vv_db < t.vv_db)


def built_up_growth(labels_2019: np.ndarray, labels_2025: np.ndarray,
                    ndbi_2019: np.ndarray, ndbi_2025: np.ndarray,
                    vh_std_2019: np.ndarray, vh_std_2025: np.ndarray,
                    thr: IndicatorThresholds | None = None) -> dict[str, np.ndarray]:
    """Built-up growth evidence layers + consensus mask."""
    t = thr or IndicatorThresholds()
    transition_in = (labels_2019 != BUILT) & (labels_2025 == BUILT)
    ndbi_rise = (ndbi_2025 - ndbi_2019) > t.ndbi_rise
    texture_rise = vh_std_2025 > vh_std_2019
    consensus = transition_in & (ndbi_rise | texture_rise)
    return {"transition_in": transition_in, "ndbi_rise": ndbi_rise,
            "texture_rise": texture_rise, "consensus": consensus}
