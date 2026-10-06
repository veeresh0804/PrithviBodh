"""Spatial block CV utilities (§6.7, Kattenborn et al. 2022).

Blocks (size from semivariogram range, expect 3-10 km Hyderabad) keep
train/test geographically disjoint. Provides GroupKFold splitter and a
leakage test asserting no block appears on both sides (DoD: 100% pass).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class SpatialCVConfig:
    n_splits: int = 5
    seed: int = 42


def spatial_splits(groups: np.ndarray, config: SpatialCVConfig | None = None):
    """Yield (train_idx, test_idx) with whole blocks held out."""
    cfg = config or SpatialCVConfig()
    gkf = GroupKFold(n_splits=cfg.n_splits)
    dummy = np.zeros(len(groups))
    yield from gkf.split(dummy, dummy, groups)


def assert_no_leakage(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    """Raise AssertionError if any block id is on both sides."""
    overlap = set(np.unique(train_groups)) & set(np.unique(test_groups))
    assert not overlap, f"Spatial leakage: blocks {sorted(overlap)[:5]} in train AND test"


def check_splits(groups: np.ndarray, config: SpatialCVConfig | None = None) -> bool:
    """Verify all folds are leakage-free; returns True or raises."""
    for tr, te in spatial_splits(groups, config):
        assert_no_leakage(np.asarray(groups)[tr], np.asarray(groups)[te])
    return True
