"""The hybrid: weighted soft voting over all models.

Each model's weight is its macro-F1 on the spatial validation blocks, so better
models count more. Weights are learned only on 'val', never on 'test'.
"""
import numpy as np


def normalise_weights(val_scores, power=2.0):
    w = {k: max(v, 0.0) ** power for k, v in val_scores.items()}
    total = sum(w.values())
    if total == 0:
        return {k: 1.0 / len(w) for k in w}
    return {k: v / total for k, v in w.items()}


def combine(probs: dict, weights: dict):
    """Weighted average of class-probability arrays with the same shape."""
    names = [k for k in probs if weights.get(k, 0) > 0]
    if not names:
        raise ValueError("No model has a positive weight")
    shape = probs[names[0]].shape
    out = np.zeros(shape, dtype="float64")
    total = 0.0
    for k in names:
        if probs[k].shape != shape:
            raise ValueError(f"Probability shape mismatch for {k}")
        out += weights[k] * probs[k]
        total += weights[k]
    return out / total
