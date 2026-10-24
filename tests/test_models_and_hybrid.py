import numpy as np
import pytest

from prithvibodh import experiment, hybrid
from prithvibodh.spatial_cv import add_blocks, split_blocks


def test_combine_weighted_average():
    p1 = np.array([[1.0, 0.0]])
    p2 = np.array([[0.0, 1.0]])
    out = hybrid.combine({"a": p1, "b": p2}, {"a": 0.75, "b": 0.25})
    assert np.allclose(out, [[0.75, 0.25]])
    with pytest.raises(ValueError):
        hybrid.combine({"a": p1}, {"a": 0})


def test_weights_sum_to_one():
    w = hybrid.normalise_weights({"a": 0.8, "b": 0.4, "c": 0.0})
    assert abs(sum(w.values()) - 1) < 1e-9 and w["a"] > w["b"] > w["c"] == 0


def test_existing_and_hybrid_run(cfg, synthetic_df):
    df = split_blocks(add_blocks(synthetic_df, 5), 0.2, 0.2, 42)
    rows, _ = experiment.run_existing(df, cfg)
    names = [r["model"] for r in rows]
    assert any("dynamic_world" in n for n in names) and any("optical-only" in n for n in names)
    esri = [r for r in rows if "esri" in r["model"]][0]
    assert "no matching class" in esri["note"]
    rows_h, out = experiment.run_hybrid(df, cfg, include_deep=False)
    assert rows_h[-1]["model"].startswith("H*")
    assert abs(sum(out["weights"].values()) - 1) < 1e-9
    assert rows_h[-1]["macro_F1"] > 0.5      # synthetic data carries a strong signal
