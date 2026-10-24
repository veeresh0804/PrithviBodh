import numpy as np

from prithvibodh import decision, evaluate, prediction, risk


def _maps():
    a = np.zeros((40, 40), dtype=np.uint8) + 2          # cropland
    a[:10, :] = 1                                      # trees on top
    a[30:, 30:] = 0                                    # a lake
    b = a.copy()
    b[:10, :15] = 3                                    # trees -> built-up
    b[30:35, 30:] = 4                                  # part of lake -> bare
    return a, b


def test_risks_in_range(cfg):
    a, b = _maps()
    rng = np.random.default_rng(0)
    valid = np.ones(a.shape, bool)
    wet = np.zeros(a.shape); wet[20:25, :] = 1
    f = risk.flood_risk(np.zeros(a.shape), wet, rng.normal(500, 5, a.shape), rng.random(a.shape) * 5, valid, cfg)
    assert np.nanmin(f) >= 0 and np.nanmax(f) <= 1
    d = risk.drought_risk(rng.random(a.shape), rng.normal(35, 3, a.shape), b, cfg["classes"], valid)
    assert np.isnan(d[b == 3]).all() and np.isfinite(d[b == 2]).all()
    t = risk.deforestation_risk(a, b, cfg["classes"], cfg)
    assert np.isfinite(t[b == 1]).all() and np.isnan(t[b != 1]).all()
    lv = risk.levels(f, cfg["risk"]["levels"])
    assert set(np.unique(lv)) <= {0, 1, 2, 255}


def test_markov_and_ca(cfg):
    a, b = _maps()
    P = prediction.transition_probabilities(a, b, 6)
    assert np.allclose(P.sum(1), 1)
    proj = prediction.project_areas(b, P, 1, cfg["classes"], 1)
    assert abs(proj["predicted_ha"].sum() - proj["now_ha"].sum()) < 1e-6
    fut = prediction.ca_markov(b, P, steps=1, window=3)
    assert fut.shape == b.shape and (fut == 3).sum() > (b == 3).sum()   # built-up keeps growing


def test_priority_table(cfg):
    a, b = _maps()
    cfg["decision"]["cell_px"] = 10
    nan = np.full(a.shape, np.nan, dtype="float32")
    from prithvibodh.change import degradation_mask
    df = decision.priority_table(a, b, cfg["classes"], np.zeros(a.shape, "float32"), nan, nan,
                                 degradation_mask(a, b, cfg["classes"]), (10, 0, 0, 0, -10, 400), cfg)
    assert len(df) == 16 and df["priority"].is_monotonic_decreasing
    assert df.iloc[0]["main_issue"] in decision.ACTIONS and df["suggested_action"].notna().all()


def test_metrics_include_precision_recall_iou():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.array([0, 1, 1, 0, 1, 2])
    s = evaluate.scores(y, p, 3)
    assert {"precision", "recall", "mIoU"} <= set(s) and 0 < s["mIoU"] < 1
    t = evaluate.per_class_table(y, p, ["a", "b", "c"])
    assert list(t.columns) == ["precision", "recall", "F1", "IoU"]
