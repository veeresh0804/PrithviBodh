"""Spectral + SAR indices (REAL NumPy implementation).

Optical bands are expected as reflectance (0-1, S2 L2A DN * 0.0001).
SAR inputs are in dB. All functions accept scalars or arrays, handle
division-by-zero by emitting 0.0, and clip optical indices to [-1, 1].

Formulas:
  NDVI  = (NIR - Red) / (NIR + Red)
  EVI   = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
  MNDWI = (Green - SWIR1) / (Green + SWIR1)
  NDBI  = (SWIR1 - NIR) / (SWIR1 + NIR)
  VV/VH ratio (dB) = VV_dB - VH_dB  (equivalent to linear power ratio)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayLike = NDArray[np.floating] | float | np.floating


def _as_float(a: ArrayLike) -> NDArray[np.floating]:
    """Coerce input to float64 ndarray (0-d for scalars)."""
    return np.asarray(a, dtype=np.float64)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise division returning 0.0 where denominator is 0."""
    out = np.zeros_like(num, dtype=np.float64)
    valid = den != 0.0
    out[valid] = num[valid] / den[valid]
    return out


def ndvi(nir: ArrayLike, red: ArrayLike) -> np.ndarray:
    """Compute NDVI = (NIR - Red) / (NIR + Red).

    Args:
        nir: Near-infrared reflectance (B8).
        red: Red reflectance (B4).

    Returns:
        NDVI array clipped to [-1, 1], 0 where NIR+Red == 0.
    """
    nir_a, red_a = _as_float(nir), _as_float(red)
    num, den = nir_a - red_a, nir_a + red_a
    return np.clip(_safe_div(num, den), -1.0, 1.0)


def evi(nir: ArrayLike, red: ArrayLike, blue: ArrayLike) -> np.ndarray:
    """Compute EVI = 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1).

    Args:
        nir: NIR reflectance (B8).
        red: Red reflectance (B4).
        blue: Blue reflectance (B2).

    Returns:
        EVI array clipped to [-1, 1], 0 where denominator is 0.
    """
    nir_a, red_a, blue_a = _as_float(nir), _as_float(red), _as_float(blue)
    num = 2.5 * (nir_a - red_a)
    den = nir_a + 6.0 * red_a - 7.5 * blue_a + 1.0
    return np.clip(_safe_div(num, den), -1.0, 1.0)


def mndwi(green: ArrayLike, swir1: ArrayLike) -> np.ndarray:
    """Compute MNDWI = (Green - SWIR1) / (Green + SWIR1).

    Args:
        green: Green reflectance (B3).
        swir1: Short-wave infrared 1 reflectance (B11).

    Returns:
        MNDWI array clipped to [-1, 1], 0 where Green+SWIR1 == 0.
    """
    g, s = _as_float(green), _as_float(swir1)
    return np.clip(_safe_div(g - s, g + s), -1.0, 1.0)


def ndbi(swir1: ArrayLike, nir: ArrayLike) -> np.ndarray:
    """Compute NDBI = (SWIR1 - NIR) / (SWIR1 + NIR).

    Args:
        swir1: SWIR1 reflectance (B11).
        nir: NIR reflectance (B8).

    Returns:
        NDBI array clipped to [-1, 1], 0 where SWIR1+NIR == 0.
    """
    s, n = _as_float(swir1), _as_float(nir)
    return np.clip(_safe_div(s - n, s + n), -1.0, 1.0)


def vv_vh_ratio_db(vv_db: ArrayLike, vh_db: ArrayLike) -> np.ndarray:
    """Compute VV/VH ratio in dB as ``VV_dB - VH_dB``.

    In linear power this equals ``10*log10(P_VV / P_VH)``.

    Args:
        vv_db: VV backscatter in dB.
        vh_db: VH backscatter in dB.

    Returns:
        Ratio array in dB (NaN propagates where inputs are NaN).
    """
    return _as_float(vv_db) - _as_float(vh_db)


def all_optical_indices(
    blue: ArrayLike, green: ArrayLike, red: ArrayLike, nir: ArrayLike, swir1: ArrayLike
) -> dict[str, np.ndarray]:
    """Compute NDVI, EVI, MNDWI and NDBI in one call.

    Args:
        blue: B2 reflectance. green: B3. red: B4. nir: B8. swir1: B11.

    Returns:
        Dict with keys ``NDVI``, ``EVI``, ``MNDWI``, ``NDBI``.
    """
    return {
        "NDVI": ndvi(nir, red),
        "EVI": evi(nir, red, blue),
        "MNDWI": mndwi(green, swir1),
        "NDBI": ndbi(swir1, nir),
    }
