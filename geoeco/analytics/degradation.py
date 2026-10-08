"""Degradation index — explicit project definition (§6.6, NOT universal).

Degrading transitions:
  tree_cover -> bare_rocky | grass_shrub
  water      -> bare_rocky | built_up
  cropland | grass_shrub -> bare_rocky
  any natural (water/tree/crop/grass) -> built_up

Index per admin unit = weighted degrading area / unit area x 100 (%).
Weights default to 1.0; water loss weighted 1.5 (scarce tank resource).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WATER, TREE, CROP, BUILT, BARE, GRASS = 0, 1, 2, 3, 4, 5
NATURAL = (WATER, TREE, CROP, GRASS)
PIXEL_HA = 0.01

DEGRADING: set[tuple[int, int]] = {
    (TREE, BARE), (TREE, GRASS),
    (WATER, BARE), (WATER, BUILT),
    (CROP, BARE), (GRASS, BARE),
}
# Any natural -> built-up (adds (TREE,BUILT),(CROP,BUILT),(GRASS,BUILT); WATER covered).
DEGRADING |= {(c, BUILT) for c in NATURAL if (c, BUILT) not in DEGRADING}


@dataclass(frozen=True)
class DegradationWeights:
    water_loss: float = 1.5
    default: float = 1.0

    def weight(self, from_class: int, to_class: int) -> float:
        if (from_class, to_class) not in DEGRADING:
            return 0.0
        return self.water_loss if from_class == WATER else self.default


def degradation_mask(labels_2019: np.ndarray, labels_2025: np.ndarray) -> np.ndarray:
    """Bool mask of degrading pixels."""
    a = labels_2019.astype(np.int8)
    b = labels_2025.astype(np.int8)
    mask = np.zeros_like(a, dtype=bool)
    for f, t in DEGRADING:
        mask |= (a == f) & (b == t)
    return mask


def degradation_by_unit(labels_2019: np.ndarray, labels_2025: np.ndarray,
                        unit_ids: np.ndarray,
                        units: pd.DataFrame,
                        weights: DegradationWeights | None = None) -> pd.DataFrame:
    """Per ward/mandal degradation % of unit area.

    units: DataFrame with ['unit_id', 'unit_name', 'area_ha'].
    unit_ids: HxW array of unit_id per pixel.
    """
    w = weights or DegradationWeights()
    a, b = labels_2019.astype(np.int8), labels_2025.astype(np.int8)
    rows = []
    for _, u in units.iterrows():
        uid = u["unit_id"]
        sel = unit_ids == uid
        if not sel.any():
            rows.append({"unit_id": uid, "unit_name": u.get("unit_name", uid),
                         "degraded_ha": 0.0, "degradation_pct": 0.0})
            continue
        wsum = sum(w.weight(f, t) for f, t in zip(a[sel].tolist(), b[sel].tolist()))
        ha = wsum * PIXEL_HA
        area = float(u.get("area_ha", sel.sum() * PIXEL_HA)) or 1e-9
        rows.append({"unit_id": uid, "unit_name": u.get("unit_name", uid),
                     "degraded_ha": ha, "degradation_pct": 100 * ha / area})
    return pd.DataFrame(rows)


DEGRADATION_DEFINITION_EPILOG = """Project degradation definition (plan §6.6 — explicit project
definition, NOT a universal ecological claim):
  degrading transitions =
    tree_cover -> bare_rocky | grass_shrub
    water -> bare_rocky | built_up
    cropland | grass_shrub -> bare_rocky
    any natural (water/tree_cover/cropland/grass_shrub) -> built_up
  index per admin unit = weighted degrading area / unit area x 100 (%);
  weights default 1.0, water loss 1.5 (scarce tank resource)."""


def run_smoke(weights: DegradationWeights | None = None) -> int:
    """Tiny synthetic degradation demo (seed 42; labelled synthetic, writes nothing)."""
    w = weights or DegradationWeights()
    rng = np.random.Generator(np.random.PCG64(42))
    labels_2019 = rng.integers(0, 6, (24, 24)).astype(np.uint8)
    labels_2025 = rng.integers(0, 6, (24, 24)).astype(np.uint8)
    mask = degradation_mask(labels_2019, labels_2025)
    unit_ids = np.where(np.arange(24 * 24).reshape(24, 24) % 2 == 0, "ward-a", "ward-b")
    units = pd.DataFrame([{"unit_id": "ward-a", "unit_name": "Ward A",
                           "area_ha": 24 * 12 * PIXEL_HA},
                          {"unit_id": "ward-b", "unit_name": "Ward B",
                           "area_ha": 24 * 12 * PIXEL_HA}])
    table = degradation_by_unit(labels_2019, labels_2025, unit_ids, units, w)
    print(json.dumps({"smoke": True, "synthetic": True,
                      "note": "SMOKE demo on synthetic data — NOT a real degradation map",
                      "degraded_pixels": int(mask.sum()),
                      "by_unit": table.to_dict(orient="records")}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Degradation index per admin unit (weighted degrading area / unit area). "
                    "Real path needs label .npy maps for both years; --smoke runs a tiny "
                    "synthetic demo (labelled, writes nothing).",
        epilog=DEGRADATION_DEFINITION_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-2019", default=None)
    ap.add_argument("--labels-2025", default=None)
    ap.add_argument("--unit-ids", default=None, help="HxW unit-id .npy (per-unit table).")
    ap.add_argument("--units", default=None, help="Units CSV [unit_id,unit_name,area_ha].")
    ap.add_argument("--water-loss-weight", type=float, default=1.5)
    ap.add_argument("--out", default=None, help="Per-unit CSV path (real path only).")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    weights = DegradationWeights(water_loss=args.water_loss_weight)
    if args.smoke:
        return run_smoke(weights)
    if not args.labels_2019 or not args.labels_2025:
        print("ERROR: refusing to run without --labels-2019 and --labels-2025 "
              "(no rasters here; --smoke for the synthetic demo only).", file=sys.stderr)
        return 1
    try:
        labels_2019 = np.load(args.labels_2019)
        labels_2025 = np.load(args.labels_2025)
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"ERROR: cannot load inputs: {e} (fail loudly, no fake outputs).", file=sys.stderr)
        return 1
    mask = degradation_mask(labels_2019, labels_2025)
    summary: dict = {"degraded_pixels": int(mask.sum()),
                     "degraded_ha": round(float(mask.sum()) * PIXEL_HA, 2)}
    if args.unit_ids and args.units:
        try:
            unit_ids = np.load(args.unit_ids)
            units = pd.read_csv(args.units)
            table = degradation_by_unit(labels_2019, labels_2025, unit_ids, units, weights)
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"ERROR: cannot load unit inputs: {e}.", file=sys.stderr)
            return 1
        summary["by_unit"] = table.to_dict(orient="records")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out, index=False)
            print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
