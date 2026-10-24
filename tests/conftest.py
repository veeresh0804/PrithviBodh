import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prithvibodh.config import load_config  # noqa: E402
from prithvibodh.features import all_columns  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config()
    c["models"]["rf"]["n_estimators"] = 30
    return c


@pytest.fixture
def synthetic_df(cfg):
    """Fake labelled points whose features carry a class signal."""
    rng = np.random.default_rng(0)
    n, k = 900, len(cfg["classes"])
    lon0, lat0, lon1, lat1 = cfg["aoi"]["bounds"]
    df = pd.DataFrame({"point_id": [f"P{i}" for i in range(n)],
                       "lon": rng.uniform(lon0, lon1, n), "lat": rng.uniform(lat0, lat1, n),
                       "label": rng.integers(0, k, n)})
    cols = all_columns(cfg)
    X = rng.normal(size=(n, len(cols))) + df["label"].to_numpy()[:, None] * 0.8
    df = pd.concat([df, pd.DataFrame(X, columns=cols)], axis=1)
    noisy = rng.random(n) < 0.3
    df["dynamic_world"] = np.where(noisy, rng.integers(0, k, n), df["label"])
    df["worldcover"] = np.where(rng.random(n) < 0.4, rng.integers(0, k, n), df["label"])
    df["esri"] = np.where(rng.random(n) < 0.05, 255, df["label"])
    return df
