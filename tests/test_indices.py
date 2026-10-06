"""Unit tests: spectral indices vs hand-computed values."""

import numpy as np
import pytest

from geoeco.features.indices import evi, mndwi, ndbi, ndvi, vv_vh_ratio_db


def test_ndvi_hand_computed() -> None:
    # NIR=0.6, Red=0.2 -> (0.4)/(0.8) = 0.5 (float tolerance for binary fp)
    assert float(ndvi(0.6, 0.2)) == pytest.approx(0.5)
    out = ndvi(np.array([0.6, 0.3]), np.array([0.2, 0.3]))
    assert np.allclose(out, [0.5, 0.0])


def test_ndvi_div0_returns_zero() -> None:
    assert float(ndvi(0.0, 0.0)) == 0.0


def test_evi_hand_computed() -> None:
    # NIR=0.5, Red=0.1, Blue=0.05 -> 2.5*0.4/(0.5+0.6-0.375+1) = 1.0/1.725
    assert abs(float(evi(0.5, 0.1, 0.05)) - 1.0 / 1.725) < 1e-9


def test_mndwi_hand_computed() -> None:
    # Green=0.3, SWIR1=0.1 -> 0.2/0.4 = 0.5
    assert float(mndwi(0.3, 0.1)) == pytest.approx(0.5)
    assert float(mndwi(0.0, 0.0)) == 0.0  # div0 guard


def test_ndbi_sign() -> None:
    # SWIR1 > NIR -> positive (built-up-like)
    assert float(ndbi(0.4, 0.2)) > 0.0
    # open water / veg -> negative
    assert float(ndbi(0.05, 0.5)) < 0.0


def test_vv_vh_ratio_db() -> None:
    assert float(vv_vh_ratio_db(-8.0, -14.0)) == -8.0 - (-14.0)
