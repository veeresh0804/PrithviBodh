"""Bootstrap 95% CIs for headline metrics (§6.7.2, NFR-01).

Resamples test points with replacement; reports percentile CIs for OA and
macro-F1, plus CI for the FUSED-minus-SINGLE delta (acceptance: lower > 0).
"""
from __future__ import annotations

import numpy as np

from geoeco.evaluation.metrics import classification_scores


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile CIs for OA and macro-F1."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    oas = np.empty(n_boot)
    f1s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s = classification_scores(y_true[idx], y_pred[idx])
        oas[b], f1s[b] = s["oa"], s["macro_f1"]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "oa": float(np.mean(oas)),
        "oa_ci": (float(np.percentile(oas, lo)), float(np.percentile(oas, hi))),
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_ci": (float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))),
        "n_boot": n_boot,
    }


def delta_ci(y_true: np.ndarray, y_fused: np.ndarray, y_single: np.ndarray,
             n_boot: int = 1000, seed: int = 42) -> dict:
    """CI for Δ macro-F1 (fused − best single). Acceptance: ci_low > 0."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).ravel()
    n = len(y_true)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        f = classification_scores(y_true[idx], np.asarray(y_fused).ravel()[idx])["macro_f1"]
        s = classification_scores(y_true[idx], np.asarray(y_single).ravel()[idx])["macro_f1"]
        deltas[b] = f - s
    return {
        "delta_mean": float(np.mean(deltas)),
        "delta_ci": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
        "passes": bool(np.percentile(deltas, 2.5) > 0),
    }
