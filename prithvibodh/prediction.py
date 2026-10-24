"""Future land-cover prediction with a Markov chain + cellular automata (CA-Markov).

1. Markov: transition probabilities learnt from map_a -> map_b say how much of each
   class turns into each other class over one interval (e.g. 2019 -> 2025).
2. Cellular automata: those amounts are placed on the map where the target class
   is already common nearby (cities grow from their edges, not at random).

Assumption: the rate of change stays the same in the next interval. With only two
dates this cannot be validated; say so in the report.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

NODATA = 255


def transition_probabilities(map_a, map_b, n_classes):
    a, b = np.asarray(map_a).ravel(), np.asarray(map_b).ravel()
    ok = (a < n_classes) & (b < n_classes)
    counts = np.bincount(a[ok] * n_classes + b[ok], minlength=n_classes ** 2).reshape(n_classes, n_classes)
    rows = counts.sum(axis=1, keepdims=True)
    P = np.where(rows > 0, counts / np.maximum(rows, 1), np.eye(n_classes))
    return P


def project_areas(map_b, P, steps, classes, pixel_area_ha=0.01):
    """Expected area of each class after `steps` intervals (Markov only)."""
    m = np.asarray(map_b).ravel()
    m = m[m < len(classes)]
    now = np.bincount(m, minlength=len(classes)).astype(float)
    future = now @ np.linalg.matrix_power(P, steps)
    return pd.DataFrame({"now_ha": now * pixel_area_ha, "predicted_ha": future * pixel_area_ha,
                         "change_ha": (future - now) * pixel_area_ha}, index=classes).round(1)


def ca_markov(map_b, P, steps=1, window=5):
    """Spatially place the Markov transitions (simple CA-Markov)."""
    cur = np.asarray(map_b).copy()
    k = P.shape[0]
    for _ in range(steps):
        counts = np.bincount(cur[cur < k].ravel(), minlength=k)
        suit = [uniform_filter((cur == j).astype("float32"), size=window, mode="nearest") for j in range(k)]
        changed = np.zeros(cur.shape, dtype=bool)
        nxt = cur.copy()
        moves = [(int(round(counts[i] * P[i, j])), i, j) for i in range(k) for j in range(k) if i != j]
        for n, i, j in sorted(moves, reverse=True):
            if n <= 0:
                continue
            cand = np.flatnonzero((cur.ravel() == i) & ~changed.ravel())
            if cand.size == 0:
                continue
            n = min(n, cand.size)
            s = suit[j].ravel()[cand]
            pick = cand[np.argpartition(-s, n - 1)[:n]] if n < cand.size else cand
            nxt.ravel()[pick] = j
            changed.ravel()[pick] = True
        cur = nxt
    return cur
