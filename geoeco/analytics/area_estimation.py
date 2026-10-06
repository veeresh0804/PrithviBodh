"""Olofsson-style error-adjusted area estimation + confidence intervals.

Given a confusion matrix (rows = map class, cols = reference class, counts)
and map class pixel proportions, computes unbiased area proportions, areas
(ha), and 95% CIs per FR-11 / §6.7. Follows Olofsson et al. 2013/2014.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_95 = 1.96


def error_adjusted_areas(confusion: np.ndarray, map_proportions: np.ndarray,
                         total_ha: float, z: float = Z_95) -> pd.DataFrame:
    """Args:
      confusion: (K,K) counts, rows=map, cols=reference.
      map_proportions: (K,) fraction of mapped area per class (sums to 1).
      total_ha: total AOI area in ha.
    Returns DataFrame [class_id, map_ha, adj_ha, se_ha, ci_low_ha, ci_high_ha].
    """
    C = np.asarray(confusion, dtype=float)
    W = np.asarray(map_proportions, dtype=float)
    K = C.shape[0]
    assert C.shape == (K, K) and W.shape == (K,)
    n_row = C.sum(axis=1, keepdims=True)
    n_row[n_row == 0] = 1.0
    p = C / n_row  # p[i,j] = P(reference=j | map=i)
    adj_prop = (W[:, None] * p).sum(axis=0)  # unbiased reference proportions
    # Standard error of adjusted proportion (Olofsson eq.): sqrt(sum_i W_i^2 * p_ij(1-p_ij)/(n_i-1)).
    n_i = C.sum(axis=1)
    se = np.zeros(K)
    for j in range(K):
        var = 0.0
        for i in range(K):
            ni = n_i[i]
            if ni > 1:
                var += W[i] ** 2 * p[i, j] * (1 - p[i, j]) / (ni - 1)
        se[j] = float(np.sqrt(var))
    adj_ha = adj_prop * total_ha
    se_ha = se * total_ha
    return pd.DataFrame({
        "class_id": np.arange(K),
        "map_ha": W * total_ha,
        "adj_ha": adj_ha,
        "se_ha": se_ha,
        "ci_low_ha": adj_ha - z * se_ha,
        "ci_high_ha": adj_ha + z * se_ha,
    })
