"""Leakage tests: no spatial block may appear in both train and test."""

import numpy as np
import pandas as pd
import pytest

from geoeco.features.sampling import (
    assert_no_leakage,
    assign_spatial_blocks,
    check_no_leakage,
    group_kfold_splits,
)


def _points(n: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"x_32644": rng.uniform(200_000, 260_000, n), "y_32644": rng.uniform(1_900_000, 1_960_000, n)}
    )


def test_blocks_disjoint_across_folds() -> None:
    df = _points()
    groups = assign_spatial_blocks(df, block_size_m=5000.0)
    assert len(np.unique(groups)) >= 5
    for tr, te in group_kfold_splits(groups, n_splits=5):
        assert check_no_leakage(groups.iloc[tr], groups.iloc[te])
        assert_no_leakage(groups.iloc[tr], groups.iloc[te])


def test_assert_no_leakage_raises_on_overlap() -> None:
    with pytest.raises(ValueError, match="Spatial leakage"):
        assert_no_leakage(["bx1_by1", "bx2_by2"], ["bx2_by2", "bx3_by3"])
