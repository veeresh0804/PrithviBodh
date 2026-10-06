"""M4: decision-level fusion of M1 (optical) + M2 (SAR) probabilities.

Two strategies: probability averaging and stacking (logistic-regression
meta-learner on concatenated class probabilities). Cheap comparison point
vs early (M3/M6) and feature-level (M7) fusion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


def average_proba(p_opt: np.ndarray, p_sar: np.ndarray,
                  w_opt: float = 0.5) -> np.ndarray:
    """Weighted average of per-class probabilities; rows re-normalised."""
    blended = w_opt * np.asarray(p_opt, dtype=float) + (1.0 - w_opt) * np.asarray(p_sar, dtype=float)
    blended = np.clip(blended, 1e-9, 1.0)
    return blended / blended.sum(axis=1, keepdims=True)


@dataclass
class StackingFusion:
    """Stacking meta-learner over concatenated [p_opt, p_sar] vectors."""

    meta: LogisticRegression | None = None
    random_state: int = 42

    def fit(self, p_opt: np.ndarray, p_sar: np.ndarray, y: np.ndarray) -> StackingFusion:
        X = np.concatenate([np.asarray(p_opt), np.asarray(p_sar)], axis=1)
        self.meta = LogisticRegression(
            max_iter=2000, multi_class="multinomial", random_state=self.random_state)
        self.meta.fit(X, np.asarray(y))
        return self

    def predict_proba(self, p_opt: np.ndarray, p_sar: np.ndarray) -> np.ndarray:
        if self.meta is None:
            raise RuntimeError("StackingFusion.fit must be called before predict.")
        X = np.concatenate([np.asarray(p_opt), np.asarray(p_sar)], axis=1)
        return self.meta.predict_proba(X)

    def predict(self, p_opt: np.ndarray, p_sar: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(p_opt, p_sar)
        classes = self.meta.classes_  # type: ignore[union-attr]
        return classes[np.argmax(proba, axis=1)]


def decision_predict(p_opt: np.ndarray, p_sar: np.ndarray,
                     method: str = "average", **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, probabilities) for method='average' (unsupervised)."""
    if method != "average":
        raise ValueError("Use StackingFusion for method='stacking'.")
    proba = average_proba(p_opt, p_sar, **kwargs)
    return np.argmax(proba, axis=1), proba
