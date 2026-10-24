import numpy as np
import pandas as pd
import pytest

from prithvibodh import baselines, change, evaluate, indicators, labels


def test_scores_perfect_and_ci():
    y = np.array([0, 1, 2, 3, 4, 5] * 10)
    s = evaluate.scores(y, y, 6)
    assert s["OA"] == 1 and s["macro_F1"] == 1
    lo, hi = evaluate.bootstrap_ci(y, y, 6, n_boot=50)
    assert lo == hi == 1


def test_crosswalks_cover_our_classes():
    for cw in baselines.CROSSWALKS.values():
        assert set(cw.values()) <= set(range(6))
    assert baselines.remap_value(10, baselines.ESRI) == baselines.NODATA   # clouds
    assert baselines.remap_value(50, baselines.WORLDCOVER) == 3             # built-up


def test_transition_and_indicators(cfg):
    a = np.array([[0, 1], [3, 255]], dtype=np.uint8)
    b = np.array([[0, 4], [3, 2]], dtype=np.uint8)
    tm = change.transition_matrix(a, b, cfg["classes"], pixel_area_ha=1)
    assert tm.loc["water", "water"] == 1 and tm.loc["tree_cover", "bare_rocky"] == 1
    assert tm.to_numpy().sum() == 3
    cm = change.change_map(a, b, 6)
    assert cm[0, 0] == 0 and cm[1, 1] == 255 and cm[0, 1] == 1 * 10 + 4 + 1
    s = indicators.summary(a, b, cfg["classes"], 2019, 2025, pixel_area_ha=1)
    assert s["degraded_ha"] == 1


def test_labels_validation(cfg, tmp_path):
    p = tmp_path / "labels.csv"
    pd.DataFrame({"point_id": ["a", "b", "c"], "lon": [78.3] * 3, "lat": [17.3] * 3,
                  "label": ["Water", "built_up", "cropland"],
                  "confidence": ["sure", "unsure", "sure"]}).to_csv(p, index=False)
    df = labels.load_labels(p, cfg)
    assert list(df["label"]) == [0, 2]
    pd.DataFrame({"point_id": ["a"], "lon": [1], "lat": [1], "label": ["forest"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        labels.load_labels(p, cfg)
