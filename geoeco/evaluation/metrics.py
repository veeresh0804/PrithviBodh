"""Classification / segmentation metrics (§6.7).

OA, macro-F1, per-class F1, Cohen's kappa, confusion matrix, mIoU (patches).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray,
                          num_classes: int = 6) -> dict:
    """Return OA, macro-F1, per-class F1, kappa, confusion."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = list(range(num_classes))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "oa": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                 average="macro", zero_division=0)),
        "per_class_f1": {int(k): float(v) for k, v in enumerate(per_class)},
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def miou_from_confusion(confusion: np.ndarray) -> tuple[float, dict[int, float]]:
    """mIoU + per-class IoU from a confusion matrix."""
    C = np.asarray(confusion, dtype=float)
    inter = np.diag(C)
    union = C.sum(axis=0) + C.sum(axis=1) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
    valid = iou[~np.isnan(iou)]
    return float(np.mean(valid)) if len(valid) else float("nan"), {
        int(k): float(v) for k, v in enumerate(iou)}
