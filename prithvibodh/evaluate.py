"""Metrics with bootstrap confidence intervals and the comparison table."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             jaccard_score, precision_score, recall_score)


def scores(y_true, y_pred, n_classes):
    labels = list(range(n_classes))
    return {
        "OA": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_F1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "mIoU": jaccard_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
    }


def per_class_f1(y_true, y_pred, classes):
    f = f1_score(y_true, y_pred, labels=list(range(len(classes))), average=None, zero_division=0)
    return dict(zip(classes, f))


def per_class_table(y_true, y_pred, classes):
    """Precision, recall, F1 and IoU for every class."""
    labels = list(range(len(classes)))
    kw = dict(labels=labels, average=None, zero_division=0)
    return pd.DataFrame({"precision": precision_score(y_true, y_pred, **kw),
                         "recall": recall_score(y_true, y_pred, **kw),
                         "F1": f1_score(y_true, y_pred, **kw),
                         "IoU": jaccard_score(y_true, y_pred, **kw)}, index=classes).round(3)


def _one(y_true, y_pred, n_classes, metric):
    labels = list(range(n_classes))
    if metric == "OA":
        return accuracy_score(y_true, y_pred)
    if metric == "mIoU":
        return jaccard_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def bootstrap_ci(y_true, y_pred, n_classes, metric="macro_F1", n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        vals.append(_one(y_true[i], y_pred[i], n_classes, metric))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta_ci(y_true, pred_a, pred_b, n_classes, n_boot=1000, seed=42):
    """95% CI of macro-F1(a) - macro-F1(b) on the same points."""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d.append(_one(y[i], a[i], n_classes, "macro_F1") - _one(y[i], b[i], n_classes, "macro_F1"))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def result_row(name, group, y_true, y_pred, n_classes, note=""):
    s = scores(y_true, y_pred, n_classes)
    lo, hi = bootstrap_ci(y_true, y_pred, n_classes)
    return {"model": name, "group": group, "n_test": len(y_true),
            "OA": round(s["OA"], 3), "precision": round(s["precision"], 3),
            "recall": round(s["recall"], 3), "macro_F1": round(s["macro_F1"], 3),
            "F1_95CI": f"{lo:.3f}-{hi:.3f}", "mIoU": round(s["mIoU"], 3),
            "kappa": round(s["kappa"], 3), "note": note}


def results_table(rows):
    return pd.DataFrame(rows).sort_values(["group", "macro_F1"], ascending=[True, False])


def plot_confusion(y_true, y_pred, classes, title="", ax=None):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return ax
