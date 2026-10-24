import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prithvibodh.deep import IGNORE, TemporalModel, UNetModel  # noqa: E402
from prithvibodh.features import all_columns  # noqa: E402


def test_temporal_models_learn(cfg):
    cfg["models"]["temporal"]["epochs"] = 5
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 300)
    X = rng.normal(size=(300, len(all_columns(cfg)))) + y[:, None]
    for kind in ("cnn", "transformer"):
        p = TemporalModel(kind, cfg, 6).fit(X, y).predict_proba(X)
        assert p.shape == (300, 6) and np.allclose(p.sum(1), 1, atol=1e-4)


def test_unet_sparse_labels(cfg):
    cfg["models"]["unet"].update(epochs=2, batch_size=2)
    X = np.random.rand(4, len(all_columns(cfg)), 32, 32).astype("float32")
    Y = np.full((4, 32, 32), IGNORE)
    Y[:, 16, 16] = [0, 1, 2, 3]
    m = UNetModel(cfg, 6).fit(X, Y)
    assert m.predict_proba_centre(X).shape == (4, 6)
