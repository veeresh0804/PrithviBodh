"""Change between two land-cover maps."""
import numpy as np
import pandas as pd

NODATA = 255

# Project definition of degradation (state it in the report): from -> to class names
DEGRADING = [("tree_cover", "bare_rocky"), ("tree_cover", "grass_shrub"),
             ("water", "bare_rocky"), ("water", "built_up"),
             ("cropland", "bare_rocky"), ("grass_shrub", "bare_rocky"),
             ("tree_cover", "built_up"), ("grass_shrub", "built_up"),
             ("cropland", "built_up")]


def transition_matrix(map_a, map_b, classes, pixel_area_ha=0.01):
    """Hectares moving from each class (rows, year A) to each class (columns, year B)."""
    a, b = np.asarray(map_a).ravel(), np.asarray(map_b).ravel()
    ok = (a != NODATA) & (b != NODATA) & (a < len(classes)) & (b < len(classes))
    k = len(classes)
    counts = np.bincount(a[ok] * k + b[ok], minlength=k * k).reshape(k, k)
    return pd.DataFrame(counts * pixel_area_ha, index=classes, columns=classes)


def change_map(map_a, map_b, n_classes):
    """0 = no change, 255 = no data, otherwise code = from*10 + to + 1."""
    a, b = np.asarray(map_a).astype(int), np.asarray(map_b).astype(int)
    out = np.where(a == b, 0, a * 10 + b + 1)
    bad = (a == NODATA) | (b == NODATA) | (a >= n_classes) | (b >= n_classes)
    return np.where(bad, NODATA, out).astype(np.uint8)


def degradation_mask(map_a, map_b, classes):
    idx = {c: i for i, c in enumerate(classes)}
    a, b = np.asarray(map_a), np.asarray(map_b)
    mask = np.zeros(a.shape, dtype=bool)
    for f, t in DEGRADING:
        mask |= (a == idx[f]) & (b == idx[t])
    return mask
