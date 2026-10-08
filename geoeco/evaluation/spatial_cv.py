"""Spatial block CV utilities (§6.7, Kattenborn et al. 2022).

Blocks (size from semivariogram range, expect 3-10 km Hyderabad) keep
train/test geographically disjoint. Provides GroupKFold splitter and a
leakage test asserting no block appears on both sides (DoD: 100% pass).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from geoeco.utils.config import load_yaml_config

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "eval" / "spatial_cv.yaml"
SYNTHETIC_PREFIX = "SYNTHETIC_"


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


def load_spatial_cv_config(path: str | Path = DEFAULT_CONFIG) -> SpatialCVConfig:
    """Load n_splits/seed from the spatial-CV YAML (config-driven, fail loudly)."""
    cfg = load_yaml_config(path)
    return SpatialCVConfig(n_splits=int(cfg["n_splits"]), seed=int(cfg["seed"]))


def synthetic_block_groups(
    n_blocks: int = 10, per_block: int = 6, seed: int = 42
) -> np.ndarray:
    """Shuffled block ids for the smoke test — clearly labelled SYNTHETIC, never real data."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"{SYNTHETIC_PREFIX}b{i:02d}" for i in range(n_blocks)], per_block)
    return rng.permutation(groups)


def run_smoke(config: SpatialCVConfig | None = None) -> dict[str, Any]:
    """GroupKFold smoke on synthetic blocks; per-fold disjointness proof (raises on leak)."""
    cfg = config or SpatialCVConfig()
    groups = synthetic_block_groups(seed=cfg.seed)
    folds = []
    for i, (tr, te) in enumerate(spatial_splits(groups, cfg)):
        assert_no_leakage(groups[tr], groups[te])  # raises AssertionError on leak
        folds.append(
            {
                "fold": i,
                "n_train": len(tr),
                "n_test": len(te),
                "train_blocks": sorted(map(str, np.unique(groups[tr]))),
                "test_blocks": sorted(map(str, np.unique(groups[te]))),
                "disjoint": bool(
                    set(map(str, np.unique(groups[tr]))).isdisjoint(
                        set(map(str, np.unique(groups[te])))
                    )
                ),
            }
        )
    return {
        "synthetic": True,
        "seed": cfg.seed,
        "n_splits": cfg.n_splits,
        "n_samples": len(groups),
        "folds": folds,
        "all_disjoint": all(f["disjoint"] for f in folds),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="GroupKFold smoke on SYNTHETIC blocks; prove disjointness."
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)
    try:
        cfg = load_spatial_cv_config(args.config)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f"ERROR: invalid spatial-CV config: {exc}", file=sys.stderr)
        return 1
    try:
        report = run_smoke(cfg)
    except AssertionError as exc:
        print(f"ERROR: spatial leakage in smoke splits: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
