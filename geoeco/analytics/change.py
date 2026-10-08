"""Post-classification change: 2019 vs 2025, confidence-masked.

- Masks pixels where either year's max-prob confidence < threshold
  (reduces false change, §6.6).
- Transition matrix in hectares (10 m pixel = 0.01 ha).
- Change Vector Analysis (CVA) on fused features as second opinion;
  disagreements with post-classification change are flagged.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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


def cva_magnitude_direction(feats_2019: np.ndarray,
                            feats_2025: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    n_classes = 6
    said_change = (change != NODATA_CHANGE) & ((change // n_classes) != (change % n_classes))
    thr = float(np.quantile(magnitude, quantile))
    cva_change = magnitude >= thr
    return said_change != cva_change


def run_smoke(year_from: int, year_to: int, config: ChangeConfig | None = None) -> int:
    """Tiny synthetic change demo (seed 42; labelled synthetic, writes nothing)."""
    cfg = config or ChangeConfig()
    rng = np.random.Generator(np.random.PCG64(42))
    labels_2019 = rng.integers(0, 6, (24, 24)).astype(np.uint8)
    labels_2025 = rng.integers(0, 6, (24, 24)).astype(np.uint8)
    conf_2019 = rng.integers(0, 101, (24, 24)).astype(np.uint8)
    conf_2025 = rng.integers(0, 101, (24, 24)).astype(np.uint8)
    change = confidence_masked_change(labels_2019, labels_2025, conf_2019, conf_2025, cfg)
    matrix = transition_matrix_ha(change)
    feats_2019 = rng.normal(0, 1, (4, 24, 24)).astype(np.float32)
    feats_2025 = rng.normal(0, 1, (4, 24, 24)).astype(np.float32)
    magnitude, _ = cva_magnitude_direction(feats_2019, feats_2025)
    disagree = cva_second_opinion(change, magnitude, cfg.cva_threshold_quantile)
    print(json.dumps({"smoke": True, "synthetic": True,
                      "note": "SMOKE demo on synthetic data — NOT a real change map",
                      "years": [year_from, year_to],
                      "changed_ha": round(float(matrix[matrix["from_class"]
                                                     != matrix["to_class"]]["area_ha"].sum()), 2),
                      "masked_pixels": int((change == NODATA_CHANGE).sum()),
                      "cva_disagreement_pixels": int(disagree.sum())}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Post-classification change between two years (confidence-masked, "
                    "CVA second opinion). Real path needs label + confidence .npy maps "
                    "for both years; --smoke runs a tiny synthetic demo (labelled, "
                    "writes nothing).")
    ap.add_argument("--from", dest="year_from", type=int, required=True)
    ap.add_argument("--to", dest="year_to", type=int, required=True)
    ap.add_argument("--labels-from", default=None)
    ap.add_argument("--labels-to", default=None)
    ap.add_argument("--conf-from", default=None)
    ap.add_argument("--conf-to", default=None)
    ap.add_argument("--confidence-threshold", type=int, default=60)
    ap.add_argument("--out", default=None, help="CSV path for the transition matrix (real path).")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.year_from >= args.year_to:
        print(f"ERROR: --from ({args.year_from}) must be < --to ({args.year_to}).",
              file=sys.stderr)
        return 1
    cfg = ChangeConfig(confidence_threshold=args.confidence_threshold)
    if args.smoke:
        return run_smoke(args.year_from, args.year_to, cfg)
    inputs = {"--labels-from": args.labels_from, "--labels-to": args.labels_to,
              "--conf-from": args.conf_from, "--conf-to": args.conf_to}
    missing = [n for n, v in inputs.items() if not v]
    if missing:
        print(f"ERROR: refusing to run without {' '.join(missing)} "
              "(no rasters here; --smoke for the synthetic demo only).", file=sys.stderr)
        return 1
    try:
        arrays = {n: np.load(p) for n, p in inputs.items()}
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"ERROR: cannot load inputs: {e} (fail loudly, no fake outputs).", file=sys.stderr)
        return 1
    change = confidence_masked_change(arrays["--labels-from"], arrays["--labels-to"],
                                      arrays["--conf-from"], arrays["--conf-to"], cfg)
    matrix = transition_matrix_ha(change)
    print(json.dumps({"years": [args.year_from, args.year_to],
                      "changed_ha": round(float(matrix[matrix["from_class"]
                                                     != matrix["to_class"]]["area_ha"].sum()), 2),
                      "masked_pixels": int((change == NODATA_CHANGE).sum())}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
