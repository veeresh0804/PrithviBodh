"""Texture features (SAR + optical) via focal statistics.

Full GLCM (Haralick) is quota-heavy in EE and slow locally, so the
pipeline uses focal mean/std as texture proxies: they separate smooth
water / granite sheets from rough built-up / rocky outcrops well enough
for the classical baselines, and the Hyd labelling protocol's
granite-vs-rooftop rule keys off these. Window default 5 px (~50 m).
"""

from __future__ import annotations

import numpy as np


def _box_mean(arr: np.ndarray, size: int) -> np.ndarray:
    """Edge-padded box mean (pure NumPy, no scipy needed).

    Args:
        arr: 2-D float array.
        size: Odd window size in pixels.

    Returns:
        Filtered array, same shape.
    """
    if size % 2 == 0:
        raise ValueError(f"window size must be odd, got {size}")
    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones((size, size), dtype=np.float64) / (size * size)
    out = np.zeros_like(arr, dtype=np.float64)
    # Separable-friendly direct loop is fine for 5x5/7x7 chip-scale use.
    for i in range(size):
        for j in range(size):
            out += padded[i : i + arr.shape[0], j : j + arr.shape[1]] * kernel[i, j]
    return out


def focal_std(arr: np.ndarray, size: int = 5) -> np.ndarray:
    """Compute focal (moving-window) standard deviation.

    Args:
        arr: 2-D array (NaN treated as missing -> filled by edge value).
        size: Odd window size in pixels.

    Returns:
        Std-dev array, same shape, NaN where input was NaN.
    """
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"focal_std needs 2-D input, got {data.shape}")
    mask = np.isnan(data)
    filled = np.where(mask, np.nanmean(data) if np.any(~mask) else 0.0, data)
    mean = _box_mean(filled, size)
    mean_sq = _box_mean(filled**2, size)
    var = np.clip(mean_sq - mean**2, 0.0, None)
    out = np.sqrt(var)
    return np.where(mask, np.nan, out)


def sar_texture_stack(vv_db: np.ndarray, vh_db: np.ndarray, size: int = 5) -> dict[str, np.ndarray]:
    """Build SAR texture proxies: focal std of VV and VH.

    Args:
        vv_db: VV dB array.
        vh_db: VH dB array.
        size: Window size in pixels.

    Returns:
        Dict with ``VV_std`` and ``VH_std`` arrays.
    """
    return {"VV_std": focal_std(vv_db, size), "VH_std": focal_std(vh_db, size)}
