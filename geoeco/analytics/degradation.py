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

from dataclasses import dataclass

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
