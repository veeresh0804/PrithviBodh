"""Feature names, groups and scale factors.

The order returned by all_columns() is the band order of the exported stack,
so the same list is used for point sampling, training and map inference.
"""
import numpy as np

OPTICAL_INDICES = ["NDVI", "NDWI", "MNDWI", "NDBI"]
SAR_BANDS = ["VV", "VH", "VV_VH"]
OPTICAL_TEXTURE = ["B8_contrast", "B8_ent"]   # GLCM texture of Sentinel-2 NIR
SAR_TEXTURE = ["VV_contrast", "VV_ent"]       # GLCM texture of Sentinel-1 VV
STATIC = ["elevation", "slope", "modis_ndvi_mean", "modis_ndvi_amp"]


def season_names(cfg):
    return list(cfg["seasons"].keys())


def optical_per_season(cfg):
    return list(cfg["sentinel2"]["bands"]) + OPTICAL_INDICES


def optical_token(cfg):
    return optical_per_season(cfg) + OPTICAL_TEXTURE


def sar_token():
    return SAR_BANDS + SAR_TEXTURE


def token_features(cfg):
    """Features each season contributes (one 'token' for the temporal models)."""
    return optical_token(cfg) + sar_token() + ["LST"]


def optical_columns(cfg):
    return [f"{s}_{b}" for s in season_names(cfg) for b in optical_token(cfg)]


def sar_columns(cfg):
    return [f"{s}_{b}" for s in season_names(cfg) for b in sar_token()]


def lst_columns(cfg):
    return [f"{s}_LST" for s in season_names(cfg)]


def all_columns(cfg):
    cols = []
    for s in season_names(cfg):
        cols += [f"{s}_{b}" for b in token_features(cfg)]
    return cols + STATIC


def feature_groups(cfg):
    """Column sets used to reproduce single-sensor 'existing' approaches and the hybrid."""
    return {
        "optical": optical_columns(cfg),
        "sar": sar_columns(cfg),
        "hybrid": all_columns(cfg),
    }


def scale_factor(column: str) -> float:
    """Multiplier used when a stack is exported as int16 (divide after reading)."""
    if column == "elevation":
        return 1.0
    if column == "slope":
        return 100.0
    if column.startswith("modis"):
        return 10000.0
    if column.endswith(("_contrast", "_ent")):
        return 1000.0                       # log-contrast and entropy
    if column.endswith(("_VV", "_VH", "_LST")):
        return 100.0                        # SAR dB (incl. VV_VH) and LST in deg C
    return 10000.0                          # S2 reflectance and spectral indices


def scale_factors(cfg) -> np.ndarray:
    return np.array([scale_factor(c) for c in all_columns(cfg)], dtype="float32")


def temporal_index(cfg):
    """Index arrays that turn an all_columns matrix into (n, seasons, token_features + static)."""
    cols = all_columns(cfg)
    pos = {c: i for i, c in enumerate(cols)}
    static_idx = [pos[c] for c in STATIC]
    idx = []
    for s in season_names(cfg):
        idx.append([pos[f"{s}_{b}"] for b in token_features(cfg)] + static_idx)
    return np.array(idx)


def to_temporal(X_all: np.ndarray, cfg) -> np.ndarray:
    """(n, all_columns) -> (n, n_seasons, n_token_features + n_static)."""
    return X_all[:, temporal_index(cfg)]
