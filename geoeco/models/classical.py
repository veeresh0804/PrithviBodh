"""Classical pixel models: M1 (optical RF), M2 (SAR RF), M3 (all RF/LightGBM), M5 (AlphaEarth RF).

Feature stack per season = 20 ch (14 optical + 4 SAR + 2 terrain); x3 seasons = 60
for classical models. AlphaEarth M5 uses 64-d annual embeddings.

Tuning note (PROJECT_CONTEXT §6.4): RF n_estimators=500; tune
``max_features`` in {"sqrt", 0.33, 0.5} and ``min_samples_leaf`` in {1, 2, 5}
via inner spatial GroupKFold (see geoeco.train.train_classical).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:  # optional dependency
    LGBMClassifier = None  # type: ignore[assignment,misc]
    _HAS_LGBM = False

CLASS_NAMES: list[str] = [
    "water",
    "tree_cover",
    "cropland",
    "built_up",
    "bare_rocky",
    "grass_shrub",
]
N_CLASSES = len(CLASS_NAMES)

# Column-name conventions for the point table (Parquet). Seasons: pre/monsoon/post.
SEASONS: tuple[str, ...] = ("pre", "monsoon", "post")
OPTICAL_BANDS: tuple[str, ...] = (
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "NDVI", "EVI", "MNDWI", "NDBI",
)
SAR_FEATS: tuple[str, ...] = ("VV_dB", "VH_dB", "VV_VH_ratio", "VH_std")
DEM_FEATS: tuple[str, ...] = ("elevation", "slope")
ALPHAEARTH_DIM = 64

ModelID = Literal["M1", "M2", "M3", "M3-lgbm", "M5"]

# Grid search space for inner spatial CV (documented, not hard-coded elsewhere).
RF_PARAM_GRID: dict[str, list] = {
    "max_features": ["sqrt", 0.33, 0.5],
    "min_samples_leaf": [1, 2, 5],
}


def feature_columns(model_id: ModelID, seasons: tuple[str, ...] = SEASONS) -> list[str]:
    """Return expected point-table columns for a model id."""
    if model_id == "M5":
        return [f"ae_{i:02d}" for i in range(ALPHAEARTH_DIM)]
    cols: list[str] = []
    for s in seasons:
        if model_id in ("M1", "M3", "M3-lgbm"):
            cols += [f"{s}_{b}" for b in OPTICAL_BANDS]
        if model_id in ("M2", "M3", "M3-lgbm"):
            cols += [f"{s}_{f}" for f in SAR_FEATS]
        if model_id in ("M3", "M3-lgbm"):
            cols += [f"{s}_{f}" for f in DEM_FEATS]
    return cols


def build_rf(
    n_estimators: int = 500,
    max_features: str | float = "sqrt",
    min_samples_leaf: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
    class_weight: str | None = "balanced_subsample",
) -> RandomForestClassifier:
    """RF with project defaults (500 trees). Tune max_features/min_samples_leaf."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
    )


def build_lightgbm(
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    n_estimators: int = 500,
    random_state: int = 42,
) -> LGBMClassifier:
    """LightGBM alternative for M3 (early fusion, all features)."""
    if not _HAS_LGBM:
        raise ImportError("lightgbm is not installed; pip install lightgbm")
    return LGBMClassifier(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )


@dataclass
class ClassicalModel:
    """Thin wrapper binding a model id to its sklearn estimator + columns."""

    model_id: ModelID
    estimator: RandomForestClassifier | LogisticRegression | object
    columns: list[str] = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)  # type: ignore[no-any-return,attr-defined]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)  # type: ignore[no-any-return,attr-defined]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_id": self.model_id, "columns": self.columns,
                     "estimator": self.estimator}, path)
        return path

    @staticmethod
    def load(path: str | Path) -> ClassicalModel:
        obj = joblib.load(path)
        return ClassicalModel(
            model_id=obj["model_id"], estimator=obj["estimator"], columns=obj["columns"])


MODEL_SPECS: dict[str, str] = {
    "M1": "RF optical-only baseline (14 feats x 3 seasons).",
    "M2": "RF SAR-only baseline (4 feats x 3 seasons).",
    "M3": "RF early fusion, all 60 feats (+ LightGBM variant M3-lgbm).",
    "M5": "RF on AlphaEarth 64-d embeddings (foundation-feature baseline).",
}


def make_model(model_id: ModelID, use_lgbm: bool = False, **kwargs) -> ClassicalModel:
    """Factory for M1/M2/M3/M5."""
    if model_id == "M3-lgbm" or (model_id == "M3" and use_lgbm):
        est = build_lightgbm(**kwargs) if kwargs else build_lightgbm()
        mid: ModelID = "M3-lgbm"
    else:
        est = build_rf(**kwargs) if kwargs else build_rf()
        mid = model_id
    return ClassicalModel(model_id=mid, estimator=est, columns=feature_columns(mid))
