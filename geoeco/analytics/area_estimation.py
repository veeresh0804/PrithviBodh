"""Olofsson-style error-adjusted area estimation + confidence intervals.

Given a confusion matrix (rows = map class, cols = reference class, counts)
and map class pixel proportions, computes unbiased area proportions, areas
(ha), and 95% CIs per FR-11 / §6.7. Follows Olofsson et al. 2013/2014.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Z_95 = 1.96


def error_adjusted_areas(confusion: np.ndarray, map_proportions: np.ndarray,
                         total_ha: float, z: float = Z_95) -> pd.DataFrame:
    """Args:
      confusion: (K,K) counts, rows=map, cols=reference.
      map_proportions: (K,) fraction of mapped area per class (sums to 1).
      total_ha: total AOI area in ha.
    Returns DataFrame [class_id, map_ha, adj_ha, se_ha, ci_low_ha, ci_high_ha].
    """
    C = np.asarray(confusion, dtype=float)
    W = np.asarray(map_proportions, dtype=float)
    K = C.shape[0]
    assert C.shape == (K, K) and W.shape == (K,)
    n_row = C.sum(axis=1, keepdims=True)
    n_row[n_row == 0] = 1.0
    p = C / n_row  # p[i,j] = P(reference=j | map=i)
    adj_prop = (W[:, None] * p).sum(axis=0)  # unbiased reference proportions
    # Standard error of adjusted proportion (Olofsson eq.):
    # sqrt(sum_i W_i^2 * p_ij(1-p_ij)/(n_i-1)).
    n_i = C.sum(axis=1)
    se = np.zeros(K)
    for j in range(K):
        var = 0.0
        for i in range(K):
            ni = n_i[i]
            if ni > 1:
                var += W[i] ** 2 * p[i, j] * (1 - p[i, j]) / (ni - 1)
        se[j] = float(np.sqrt(var))
    adj_ha = adj_prop * total_ha
    se_ha = se * total_ha
    return pd.DataFrame({
        "class_id": np.arange(K),
        "map_ha": W * total_ha,
        "adj_ha": adj_ha,
        "se_ha": se_ha,
        "ci_low_ha": adj_ha - z * se_ha,
        "ci_high_ha": adj_ha + z * se_ha,
    })


def load_confusion(path: str | Path) -> np.ndarray:
    """Load a confusion matrix from .npy or .csv (rows = map class, cols = reference)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Confusion matrix not found: {path}")
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def parse_proportions(raw: str) -> np.ndarray:
    """Parse proportions from '0.1,0.2,...' or a file path (.npy/.csv/.txt)."""
    candidate = Path(raw)
    if candidate.is_file():
        if candidate.suffix.lower() == ".npy":
            return np.asarray(np.load(candidate), dtype=float)
        return np.asarray(pd.read_csv(candidate, header=None).to_numpy(dtype=float).ravel())
    try:
        return np.array([float(v) for v in raw.split(",")], dtype=float)
    except ValueError as e:
        raise ValueError(f"Cannot parse --proportions {raw!r}: {e}") from e


def run_smoke(total_ha: float = 360000.0, z: float = Z_95) -> int:
    """Olofsson-style CI on a synthetic confusion matrix (seed 42; writes nothing)."""
    rng = np.random.Generator(np.random.PCG64(42))
    diag = np.array([120, 90, 80, 70, 60, 50], dtype=float)
    off = rng.integers(0, 6, (6, 6)).astype(float)
    confusion = off + np.diag(diag)
    map_proportions = np.array([0.10, 0.25, 0.20, 0.15, 0.10, 0.20])
    table = error_adjusted_areas(confusion, map_proportions, total_ha, z=z)
    print("# SMOKE demo on a synthetic confusion matrix — NOT real accuracy assessment")
    print(table.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Olofsson-style error-adjusted area estimation + 95% CIs "
                    "(FR-11/§6.7) from a confusion matrix (rows = map, cols = reference) "
                    "and map class proportions. --smoke runs the CI math on a synthetic "
                    "confusion matrix (labelled, writes nothing).")
    ap.add_argument("--confusion", default=None, help="Confusion matrix .npy or .csv.")
    ap.add_argument("--proportions", default=None,
                    help="'0.1,0.2,...' or a file path (.npy/.csv/.txt).")
    ap.add_argument("--total-ha", type=float, default=None)
    ap.add_argument("--z", type=float, default=Z_95)
    ap.add_argument("--out", default=None, help="Adjusted-areas CSV path (real path only).")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        return run_smoke(args.total_ha or 360000.0, z=args.z)
    missing = [n for n, v in (("--confusion", args.confusion),
                              ("--proportions", args.proportions),
                              ("--total-ha", args.total_ha)) if v is None]
    if missing:
        print(f"ERROR: refusing to run without {' '.join(missing)} "
              "(no validation sample here; --smoke for the synthetic demo only).",
              file=sys.stderr)
        return 1
    try:
        confusion = load_confusion(args.confusion)
        proportions = parse_proportions(args.proportions)
        table = error_adjusted_areas(confusion, proportions, float(args.total_ha), z=args.z)
    except (FileNotFoundError, OSError, ValueError, AssertionError) as e:
        print(f"ERROR: cannot compute adjusted areas: {e} (fail loudly).", file=sys.stderr)
        return 1
    print(table.to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
