"""Benchmark vs Dynamic World + ESA WorldCover on the same points (§6.7.4).

Crosswalks every DW (9-class) / WorldCover (11-class) label to exactly one
of our 6 classes, then scores with the same metrics as our models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from geoeco.evaluation.metrics import classification_scores

# Our ids: 0 water, 1 tree_cover, 2 cropland, 3 built_up, 4 bare_rocky, 5 grass_shrub.
# DW labels: 0 water,1 trees,2 grass,3 flooded veg,4 crops,5 shrub/scrub,6 built,7 bare,8 snow/ice.
DW_CROSSWALK: dict[int, int] = {
    0: 0, 1: 1, 2: 5, 3: 5, 4: 2, 5: 5, 6: 3, 7: 4, 8: 4,
}
# WorldCover labels: 10 tree,20 shrub,30 grass,40 crops,50 built,60 bare,70 snow,
# 80 water,90 wetland,95 mangrove,100 moss/lichen.
WC_CROSSWALK: dict[int, int] = {
    10: 1, 20: 5, 30: 5, 40: 2, 50: 3, 60: 4, 70: 4, 80: 0, 90: 0, 95: 1, 100: 5,
}


def crosswalk(values: np.ndarray | pd.Series, mapping: dict[int, int],
              product: str) -> np.ndarray:
    """Map external labels to our 6; raises if any value is unmapped."""
    arr = np.asarray(values).ravel()
    unknown = sorted(set(np.unique(arr).tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"{product}: unmapped labels {unknown} — crosswalk must cover all")
    return np.array([mapping[int(v)] for v in arr], dtype=int)


def benchmark_at_points(df: pd.DataFrame, dw_col: str = "dw_label",
                        wc_col: str = "wc_label", ref_col: str = "label") -> dict:
    """Score DW + WorldCover sampled at our test points with shared metrics."""
    y = df[ref_col].to_numpy(dtype=int)
    out: dict = {}
    if dw_col in df.columns:
        out["dynamic_world"] = classification_scores(y, crosswalk(df[dw_col], DW_CROSSWALK, "DW"))
    if wc_col in df.columns:
        out["worldcover"] = classification_scores(y, crosswalk(df[wc_col], WC_CROSSWALK, "WorldCover"))
    return out
