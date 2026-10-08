"""Ecosystem indicators: vegetation, water, built-up (§6.6, FR-09).

- Vegetation: seasonal NDVI/EVI mean + delta, computed WITHIN veg classes only.
- Water: fused rule MNDWI > thr OR SAR VV < thr (SAR covers monsoon cloud).
- Built-up: transitions into built-up + NDBI rise + SAR-texture trend proxy
  (VH temporal std increase).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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


def run_smoke(thr: IndicatorThresholds | None = None) -> int:
    """Tiny synthetic indicator demo (seed 42; labelled synthetic, writes nothing)."""
    t = thr or IndicatorThresholds()
    rng = np.random.Generator(np.random.PCG64(42))
    shape = (24, 24)
    labels = rng.integers(0, 6, shape).astype(np.uint8)
    labels_2019 = rng.integers(0, 6, shape).astype(np.uint8)
    ndvi_t1 = rng.normal(0.4, 0.2, shape)
    ndvi_t2 = rng.normal(0.4, 0.2, shape)
    mndwi = rng.normal(0.0, 0.3, shape)
    vv_db = rng.normal(-12.0, 4.0, shape)
    ndbi_2019 = rng.normal(0.0, 0.2, shape)
    ndbi_2025 = rng.normal(0.0, 0.2, shape)
    vh_std_2019 = rng.random(shape)
    vh_std_2025 = rng.random(shape)
    means = veg_means(ndvi_t2, labels)
    delta = veg_delta(ndvi_t1, ndvi_t2, labels)
    water = water_mask(mndwi, vv_db, t)
    growth = built_up_growth(labels_2019, labels, ndbi_2019, ndbi_2025,
                             vh_std_2019, vh_std_2025, t)
    print(json.dumps({"smoke": True, "synthetic": True,
                      "note": "SMOKE demo on synthetic data — NOT real indicators",
                      "veg_means": {str(k): v for k, v in means.items()},
                      "veg_delta": {str(k): v for k, v in delta.items()},
                      "water_pixels": int(water.sum()),
                      "built_up_consensus_pixels": int(growth["consensus"].sum())}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ecosystem indicators (§6.6, FR-09): vegetation means/deltas within "
                    "veg classes, fused MNDWI/SAR water mask, built-up growth consensus. "
                    "Real path needs label + index .npy artefacts; --smoke runs a tiny "
                    "synthetic demo (labelled, writes nothing).")
    ap.add_argument("--labels", default=None, help="Label map .npy (t2 year).")
    ap.add_argument("--labels-2019", default=None, help="Label map .npy (t1 year, built-up only).")
    ap.add_argument("--index-t1", default=None, help="Vegetation index .npy (t1, e.g. NDVI).")
    ap.add_argument("--index-t2", default=None, help="Vegetation index .npy (t2).")
    ap.add_argument("--mndwi", default=None)
    ap.add_argument("--vv-db", default=None)
    ap.add_argument("--ndbi-2019", default=None)
    ap.add_argument("--ndbi-2025", default=None)
    ap.add_argument("--vh-std-2019", default=None)
    ap.add_argument("--vh-std-2025", default=None)
    ap.add_argument("--mndwi-thr", type=float, default=0.2)
    ap.add_argument("--vv-db-thr", type=float, default=-15.0)
    ap.add_argument("--ndbi-rise", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="JSON summary path (real path only).")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    thr = IndicatorThresholds(mndwi=args.mndwi_thr, vv_db=args.vv_db_thr,
                              ndbi_rise=args.ndbi_rise)
    if args.smoke:
        return run_smoke(thr)
    if not args.labels:
        print("ERROR: refusing to run without --labels "
              "(no rasters here; --smoke for the synthetic demo only).", file=sys.stderr)
        return 1
    if not any([args.index_t1, args.index_t2, args.mndwi, args.vv_db,
                args.ndbi_2019, args.labels_2019]):
        print("ERROR: at least one index artefact is required alongside --labels "
              "(--index-t1/--index-t2/--mndwi/--vv-db/--ndbi-2019...).", file=sys.stderr)
        return 1
    try:
        def _load(p):
            return np.load(p) if p else None

        labels = _load(args.labels)
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"ERROR: cannot load inputs: {e} (fail loudly, no fake outputs).", file=sys.stderr)
        return 1
    summary: dict = {"labels_shape": list(labels.shape)}
    if args.index_t2:
        try:
            summary["veg_means"] = {str(k): v for k, v in
                                    veg_means(_load(args.index_t2), labels).items()}
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"ERROR: cannot load inputs: {e}.", file=sys.stderr)
            return 1
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
