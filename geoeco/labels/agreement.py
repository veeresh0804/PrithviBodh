"""Inter-annotator agreement: percent agreement + Cohen's kappa.

Protocol gate: 10% double-labelled points must reach >= 85% agreement
(config ``labels.agreement``) before the test set is frozen.
"""

from __future__ import annotations

import numpy as np


def percent_agreement(y1: list[str] | np.ndarray, y2: list[str] | np.ndarray) -> float:
    """Compute fraction of items where both annotators agree.

    Args:
        y1: First annotator's labels.
        y2: Second annotator's labels.

    Returns:
        Agreement fraction in [0, 1].

    Raises:
        ValueError: On length mismatch or empty input.
    """
    a = np.asarray(list(y1), dtype=str)
    b = np.asarray(list(y2), dtype=str)
    if a.shape != b.shape:
        raise ValueError(f"Label vectors differ in shape: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("Label vectors must be non-empty")
    return float(np.mean(a == b))


def cohen_kappa(y1: list[str] | np.ndarray, y2: list[str] | np.ndarray) -> float:
    """Compute Cohen's kappa for two annotators (pure NumPy).

    Args:
        y1: First annotator's labels.
        y2: Second annotator's labels.

    Returns:
        Kappa in [-1, 1]; 0 is chance agreement. Returns 0.0 when the
        expected agreement is 1 (degenerate single-class sample).
    """
    a = np.asarray(list(y1), dtype=str)
    b = np.asarray(list(y2), dtype=str)
    if a.shape != b.shape:
        raise ValueError(f"Label vectors differ in shape: {a.shape} vs {b.shape}")
    classes = np.unique(np.concatenate([a, b]))
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in classes)
    if pe == 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def passes_gate(
    y1: list[str] | np.ndarray, y2: list[str] | np.ndarray, thr: float = 0.85
) -> tuple[bool, float, float]:
    """Check the protocol agreement gate.

    Args:
        y1: First annotator's labels.
        y2: Second annotator's labels.
        thr: Minimum percent agreement (default 0.85).

    Returns:
        (passed, agreement, kappa) tuple.
    """
    pa = percent_agreement(y1, y2)
    kappa = cohen_kappa(y1, y2)
    return pa >= thr, pa, kappa
