"""Point sampling + spatial-block splitting (leakage-safe).

Kattenborn et al. 2022: random hold-out can overestimate CNN accuracy by
up to 28% vs spatially blocked. All eval therefore uses whole-block
hold-outs via :class:`sklearn.model_selection.GroupKFold`; groups are
regular grid blocks (block size from semivariogram range, expect 3-10 km
in Hyderabad) assigned by :func:`assign_spatial_blocks`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geoeco.utils.config import load_yaml_config, require_keys

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CV_CONFIG = REPO / "configs" / "eval" / "spatial_cv.yaml"


def assign_spatial_blocks(
    df: pd.DataFrame,
    block_size_m: float = 5000.0,
    x_col: str = "x_32644",
    y_col: str = "y_32644",
) -> pd.Series:
    """Assign each point to a regular grid block id.

    Args:
        df: Points table with projected metre coordinates.
        block_size_m: Block edge length in metres.
        x_col: Easting column name.
        y_col: Northing column name.

    Returns:
        Series of block ids like ``"bx123_by456"``.

    Raises:
        KeyError: If coordinate columns are missing.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Missing coordinate columns: {x_col!r}, {y_col!r}")
    bx = np.floor(df[x_col].to_numpy(dtype=np.float64) / block_size_m).astype(int)
    by = np.floor(df[y_col].to_numpy(dtype=np.float64) / block_size_m).astype(int)
    return pd.Series([f"bx{x}_by{y}" for x, y in zip(bx, by)], index=df.index, name="block_id")


def check_no_leakage(train_groups: list[str] | np.ndarray, test_groups: list[str] | np.ndarray) -> bool:
    """Return True iff no group appears in both train and test.

    Args:
        train_groups: Block ids used for training.
        test_groups: Block ids used for testing.

    Returns:
        True when the sets are disjoint.
    """
    return len(set(map(str, train_groups)) & set(map(str, test_groups))) == 0


def assert_no_leakage(train_groups: list[str] | np.ndarray, test_groups: list[str] | np.ndarray) -> None:
    """Raise if any block leaks between train and test.

    Args:
        train_groups: Training block ids.
        test_groups: Testing block ids.

    Raises:
        ValueError: On any overlap (lists the offending ids).
    """
    overlap = set(map(str, train_groups)) & set(map(str, test_groups))
    if overlap:
        raise ValueError(f"Spatial leakage: blocks in both train and test: {sorted(overlap)}")


def group_kfold_splits(
    groups: list[str] | np.ndarray | pd.Series, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) splits with disjoint groups.

    Pure-NumPy GroupKFold with scikit-learn-compatible semantics: each
    distinct group appears in exactly one test fold. Groups are dealt
    round-robin (sorted order) to folds for determinism. No sklearn
    import needed, so leakage tests run on minimal installs.

    Args:
        groups: Block id per sample.
        n_splits: Number of folds.

    Yields:
        (train_indices, test_indices) integer arrays per fold.

    Raises:
        ValueError: If fewer distinct groups than splits.
    """
    g = np.asarray(groups, dtype=str)
    uniq = np.unique(g)
    if len(uniq) < n_splits:
        raise ValueError(f"Need >= {n_splits} distinct blocks, got {len(uniq)}")
    # Deterministic round-robin assignment of sorted groups to folds.
    fold_of = {grp: i % n_splits for i, grp in enumerate(sorted(uniq))}
    fold_ids = np.array([fold_of[x] for x in g])
    idx = np.arange(len(g))
    for fold in range(n_splits):
        te_mask = fold_ids == fold
        yield idx[~te_mask], idx[te_mask]


def sample_points_at_coords(
    stack: np.ndarray, rows: np.ndarray, cols: np.ndarray, nodata: float = np.nan
) -> np.ndarray:
    """Sample a (bands, rows, cols) or (rows, cols) stack at pixel indices.

    Args:
        stack: Raster stack array.
        rows: Row indices (1-D int array).
        cols: Column indices (1-D int array).
        nodata: Value treated as invalid (rows filtered out when any
            band equals it; NaN-aware).

    Returns:
        Sample matrix shaped (n_points, n_bands) — or (n_points,) for
        a single band. Points hitting nodata are returned as NaN rows;
        callers decide to drop or keep them.
    """
    arr = np.asarray(stack)
    rows_i = np.asarray(rows, dtype=int)
    cols_i = np.asarray(cols, dtype=int)
    if arr.ndim == 2:
        return arr[rows_i, cols_i]
    if arr.ndim == 3:
        return arr[:, rows_i, cols_i].T
    raise ValueError(f"stack must be 2-D or 3-D, got shape {arr.shape}")


def load_block_config(path: str | Path = DEFAULT_CV_CONFIG) -> dict[str, Any]:
    """Load and structural-check the spatial-block CV config (no data needed)."""
    cfg = load_yaml_config(path)
    require_keys(
        cfg,
        ["method", "n_splits", "group_by", "block_size_km", "seed"],
        name="spatial_cv.yaml",
    )
    return cfg


def describe_cv_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate block config and return the printable CV plan.

    Raises:
        ValueError: On non-GroupKFold method, < 2 splits, or a
            malformed block_size_km range.
    """
    if cfg["method"] != "GroupKFold":
        raise ValueError(f"method must be 'GroupKFold', got {cfg['method']!r}")
    n_splits = int(cfg["n_splits"])
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    sizes = [float(v) for v in cfg["block_size_km"]]
    if len(sizes) != 2 or not (0 < sizes[0] <= sizes[1]):
        raise ValueError(f"block_size_km must be [lo, hi] with 0 < lo <= hi, got {sizes}")
    return {
        "method": cfg["method"],
        "n_splits": n_splits,
        "group_by": cfg["group_by"],
        "block_size_km": sizes,
        "block_size_m": [s * 1000.0 for s in sizes],
        "seed": int(cfg["seed"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate block config; print the spatial CV plan.")
    ap.add_argument("--config", default=str(DEFAULT_CV_CONFIG))
    args = ap.parse_args(argv)
    try:
        plan = describe_cv_plan(load_block_config(args.config))
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f"ERROR: invalid block config: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"cv_plan": plan}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
