import numpy as np

from prithvibodh import features as F


def test_columns_unique_and_grouped(cfg):
    cols = F.all_columns(cfg)
    assert len(cols) == len(set(cols))
    g = F.feature_groups(cfg)
    assert set(g["optical"]).isdisjoint(g["sar"])
    assert set(g["optical"]) | set(g["sar"]) <= set(g["hybrid"])


def test_temporal_shape(cfg):
    cols = F.all_columns(cfg)
    X = np.arange(2 * len(cols), dtype="float32").reshape(2, len(cols))
    T = F.to_temporal(X, cfg)
    n_tok = len(F.token_features(cfg)) + len(F.STATIC)
    assert T.shape == (2, len(cfg["seasons"]), n_tok)
    assert T[0, 0, 0] == X[0, cols.index(f"{F.season_names(cfg)[0]}_B2")]


def test_scale_factors_fit_int16(cfg):
    s = F.scale_factors(cfg)
    assert len(s) == len(F.all_columns(cfg))
    assert F.scale_factor("monsoon_VV") == 100 and F.scale_factor("pre_monsoon_NDVI") == 10000
    assert F.scale_factor("pre_monsoon_VV") == 100 and F.scale_factor("post_monsoon_VV_VH") == 100
    assert F.scale_factor("pre_monsoon_LST") == 100 and F.scale_factor("monsoon_B8_contrast") == 1000
    assert F.scale_factor("elevation") == 1 and F.scale_factor("slope") == 100


def test_texture_in_groups(cfg):
    g = F.feature_groups(cfg)
    assert any(c.endswith("B8_contrast") for c in g["optical"])
    assert any(c.endswith("VV_ent") for c in g["sar"])
    assert len(F.all_columns(cfg)) == 70
