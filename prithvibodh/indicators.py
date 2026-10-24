"""Ecosystem indicators from land-cover maps."""
import numpy as np
import pandas as pd

from .change import NODATA, degradation_mask


def class_areas(lc_map, classes, pixel_area_ha=0.01):
    m = np.asarray(lc_map).ravel()
    m = m[(m != NODATA) & (m < len(classes))]
    counts = np.bincount(m, minlength=len(classes))
    return pd.Series(counts * pixel_area_ha, index=classes, name="area_ha")


def summary(map_a, map_b, classes, year_a, year_b, pixel_area_ha=0.01):
    a = class_areas(map_a, classes, pixel_area_ha)
    b = class_areas(map_b, classes, pixel_area_ha)
    valid = (np.asarray(map_a) != NODATA) & (np.asarray(map_b) != NODATA)
    deg = degradation_mask(map_a, map_b, classes) & valid
    total_ha = valid.sum() * pixel_area_ha
    return {
        f"water_ha_{year_a}": float(a["water"]), f"water_ha_{year_b}": float(b["water"]),
        f"tree_ha_{year_a}": float(a["tree_cover"]), f"tree_ha_{year_b}": float(b["tree_cover"]),
        "built_up_growth_ha": float(b["built_up"] - a["built_up"]),
        "degraded_ha": float(deg.sum() * pixel_area_ha),
        "degraded_pct_of_area": float(100 * deg.sum() * pixel_area_ha / max(total_ha, 1e-9)),
    }


def ndvi_change(ndvi_a, ndvi_b, veg_mask):
    """Mean NDVI change inside vegetated pixels (trees, crops, grass/shrub)."""
    d = np.asarray(ndvi_b, float) - np.asarray(ndvi_a, float)
    return float(np.nanmean(d[veg_mask])) if veg_mask.any() else float("nan")
