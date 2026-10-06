"""Cloud-stress test (§6.7.5, NFR-02).

(a) Natural: evaluate on monsoon-season composites (few clear observations).
(b) Synthetic: progressively mask optical channels (25/50/75/100%) and track
macro-F1 for fused vs optical-only — fused must drop less.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone

from geoeco.evaluation.metrics import classification_scores

MASK_LEVELS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)


def synthetic_optical_mask(X: np.ndarray, n_optical: int, frac: float,
                           seed: int = 42) -> np.ndarray:
    """Zero a random frac of OPTICAL columns (first n_optical cols by convention)."""
    rng = np.random.default_rng(seed)
    Xm = X.copy().astype(float)
    n_drop = int(round(n_optical * frac))
    drop = rng.choice(n_optical, size=n_drop, replace=False)
    Xm[:, drop] = 0.0
    return Xm


def cloud_stress_curve(estimator, X_test: np.ndarray, y_test: np.ndarray,
                       n_optical: int, levels: tuple[float, ...] = MASK_LEVELS,
                       seed: int = 42) -> dict[float, dict]:
    """Macro-F1 at each synthetic masking level (0.0 = clean included)."""
    out: dict[float, dict] = {0.0: classification_scores(y_test, estimator.predict(X_test))}
    for frac in levels:
        Xm = synthetic_optical_mask(X_test, n_optical, frac, seed)
        out[frac] = classification_scores(y_test, estimator.predict(Xm))
    return out


def monsoon_drop(all_season_f1: float, monsoon_only_f1: float) -> float:
    """Absolute macro-F1 drop under monsoon-only evaluation (lower is better)."""
    return float(all_season_f1 - monsoon_only_f1)
