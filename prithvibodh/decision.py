"""Decision support: a ranked list of 2 km grid cells with the main issue and a suggested action.

Rules are simple and visible, so a planner can see why a cell was flagged.
"""
import numpy as np
import pandas as pd

NODATA = 255

ACTIONS = {
    "water_loss": "Lakes or tanks shrinking: survey boundaries, check encroachment and silting.",
    "tree_loss": "Tree cover lost: protect what remains and plan replanting.",
    "built_growth_flood": "New construction on flood-prone ground: review drainage before approvals.",
    "built_growth": "Fast urban growth: plan green space and water recharge areas.",
    "drought": "Vegetation under heat and dryness stress: prioritise water conservation and crop advisories.",
    "deforestation_risk": "Remaining trees close to recent loss or new buildings: monitor closely.",
    "degradation": "Land degrading to bare or built-up: field check and consider restoration.",
}

WEIGHTS = {"water_loss": 1.0, "tree_loss": 1.0, "built_growth_flood": 1.0, "built_growth": 0.6,
           "drought": 0.8, "deforestation_risk": 0.8, "degradation": 0.9}


def _cell_mean(arr, cell):
    h, w = arr.shape
    H, W = h // cell, w // cell
    a = arr[:H * cell, :W * cell].astype("float64").reshape(H, cell, W, cell)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN cells stay NaN
        return np.nanmean(np.nanmean(a, axis=3), axis=1)


def _scale(v):
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v)
    hi = np.nanpercentile(v[finite], 98) or 1.0
    return np.clip(np.nan_to_num(v / hi), 0, 1)


def priority_table(map_a, map_b, classes, flood, drought, defor, degraded, transform, cfg):
    """transform: (a, b, c, d, e, f) affine of the maps, x = a*col + b*row + c, y = d*col + e*row + f."""
    cell = cfg["decision"]["cell_px"]
    idx = {c: i for i, c in enumerate(classes)}
    valid = (map_a != NODATA) & (map_b != NODATA)
    nanmask = lambda m: np.where(valid, m.astype("float32"), np.nan)
    ind = {
        "water_loss": nanmask((map_a == idx["water"]) & (map_b != idx["water"])),
        "tree_loss": nanmask((map_a == idx["tree_cover"]) & (map_b != idx["tree_cover"])),
        "built_growth": nanmask((map_a != idx["built_up"]) & (map_b == idx["built_up"])),
        "drought": drought,
        "deforestation_risk": defor,
        "degradation": nanmask(degraded),
    }
    new_built = (map_a != idx["built_up"]) & (map_b == idx["built_up"])
    ind["built_growth_flood"] = np.where(new_built, np.nan_to_num(flood), 0.0).astype("float32")
    ind["built_growth_flood"][~valid] = np.nan

    cells = {k: _cell_mean(v, cell) for k, v in ind.items()}
    H, W = next(iter(cells.values())).shape
    rows, cols = np.mgrid[0:H, 0:W]
    a, b, c, d, e, f = transform
    cc, rr = cols * cell + cell / 2, rows * cell + cell / 2
    df = pd.DataFrame({"cell_row": rows.ravel(), "cell_col": cols.ravel(),
                       "x": (a * cc + b * rr + c).ravel(), "y": (d * cc + e * rr + f).ravel()})
    scaled = {}
    for k, v in cells.items():
        df[k] = np.round(v.ravel(), 4)
        scaled[k] = _scale(v.ravel()) * WEIGHTS[k]
    S = np.vstack([scaled[k] for k in WEIGHTS])
    df["priority"] = np.round(S.max(axis=0) * 0.6 + S.mean(axis=0) * 0.4, 3)
    keys = list(WEIGHTS)
    df["main_issue"] = [keys[i] for i in S.argmax(axis=0)]
    df["suggested_action"] = df["main_issue"].map(ACTIONS)
    return df.sort_values("priority", ascending=False).reset_index(drop=True)
