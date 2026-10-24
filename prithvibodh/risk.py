"""Simple environmental risk indices: flood, drought and deforestation.

These are transparent screening indices built from the satellite layers, not
calibrated hazard models. Each returns a 0-1 score per pixel (NaN where the
index does not apply) so it can be mapped and summarised.
"""
import numpy as np
from scipy.ndimage import uniform_filter

NODATA = 255


def normalise(arr, valid=None, lo=2, hi=98):
    """Scale to 0-1 using the 2nd-98th percentile of valid pixels."""
    a = np.asarray(arr, dtype="float32")
    v = np.isfinite(a) if valid is None else (valid & np.isfinite(a))
    if not v.any():
        return np.full(a.shape, np.nan, dtype="float32")
    p_lo, p_hi = np.percentile(a[v], [lo, hi])
    out = (a - p_lo) / max(p_hi - p_lo, 1e-9)
    return np.clip(out, 0, 1).astype("float32")


def nearby(mask, size):
    """Fraction of pixels inside a size x size window that are True."""
    return uniform_filter(np.asarray(mask, dtype="float32"), size=size, mode="nearest")


def flood_risk(water_dry, water_wet, elevation, slope, valid, cfg):
    """Seasonal flooding nearby + low-lying ground + flat terrain.

    Permanent water (wet in both seasons) is excluded (NaN).
    """
    r = cfg["risk"]
    seasonal = (water_wet > 0) & (water_dry == 0)
    permanent = (water_wet > 0) & (water_dry > 0)
    flood_near = normalise(nearby(seasonal, r["neighbourhood_px"]), valid)
    relief = elevation - uniform_filter(elevation.astype("float32"), size=r["relief_window_px"], mode="nearest")
    low_ground = 1 - normalise(relief, valid)
    flat = 1 - normalise(slope, valid)
    score = 0.5 * flood_near + 0.25 * low_ground + 0.25 * flat
    score[~valid | permanent] = np.nan
    return score


def drought_risk(ndvi_dry, lst_dry, lc_map, classes, valid):
    """1 - Vegetation Health Index, using dry-season greenness and heat (vegetated land only).

    VCI = scaled dry-season NDVI, TCI = 1 - scaled dry-season LST, VHI = (VCI + TCI) / 2.
    """
    idx = {c: i for i, c in enumerate(classes)}
    veg = np.isin(lc_map, [idx["tree_cover"], idx["cropland"], idx["grass_shrub"]]) & valid
    vci = normalise(ndvi_dry, veg)
    tci = 1 - normalise(lst_dry, veg)
    score = 1 - 0.5 * (vci + tci)
    score[~veg] = np.nan
    return score


def deforestation_risk(map_a, map_b, classes, cfg):
    """Remaining tree cover close to recent tree loss and to new built-up land."""
    idx = {c: i for i, c in enumerate(classes)}
    t, b = idx["tree_cover"], idx["built_up"]
    size = cfg["risk"]["neighbourhood_px"] * 3          # ~450 m
    tree_loss = (map_a == t) & (map_b != t) & (map_b != NODATA)
    new_built = (map_a != b) & (map_b == b) & (map_a != NODATA)
    trees_now = map_b == t
    near_loss = normalise(nearby(tree_loss, size), trees_now)
    near_built = normalise(nearby(new_built, size), trees_now)
    score = 0.5 * near_loss + 0.5 * near_built
    score[~trees_now] = np.nan
    return score


def levels(score, cuts):
    """0 = low, 1 = medium, 2 = high, 255 = not applicable."""
    out = np.full(score.shape, NODATA, dtype=np.uint8)
    ok = np.isfinite(score)
    out[ok] = np.digitize(score[ok], cuts).astype(np.uint8)
    return out


def summary(score, cuts, pixel_area_ha=0.01):
    lv = levels(score, cuts)
    return {name: float((lv == i).sum() * pixel_area_ha) for i, name in enumerate(["low_ha", "medium_ha", "high_ha"])}
