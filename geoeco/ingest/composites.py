"""Season definitions + median composite logic.

Seasons (project default, config ``seasons``):
  pre      = Mar-May (months 3-5)
  monsoon  = Jun-Sep (months 6-9)
  post     = Oct-Dec (months 10-12)
Jan-Feb are excluded (rabi shoulder; avoids thin-year stacks).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Season = Literal["pre", "monsoon", "post"]

SEASONS: dict[str, tuple[int, int]] = {
    "pre": (3, 5),
    "monsoon": (6, 9),
    "post": (10, 12),
}


def season_months(season: Season) -> list[int]:
    """Return the inclusive month list for a season.

    Args:
        season: One of ``"pre"``, ``"monsoon"``, ``"post"``.

    Returns:
        List of month numbers.

    Raises:
        KeyError: For unknown season names.
    """
    if season not in SEASONS:
        raise KeyError(f"Unknown season {season!r}; expected one of {sorted(SEASONS)}")
    start, end = SEASONS[season]
    return list(range(start, end + 1))


def season_date_range(year: int, season: Season) -> tuple[str, str]:
    """Return ISO (start, end-exclusive) dates for a season/year.

    Args:
        year: Calendar year (e.g. 2019, 2025).
        season: Season name.

    Returns:
        (start_iso, end_exclusive_iso) date strings.
    """
    import calendar

    start_m, end_m = SEASONS[season]
    start = f"{year}-{start_m:02d}-01"
    last_day = calendar.monthrange(year, end_m)[1]
    # End-exclusive: first day of the month after the season end.
    if end_m == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{end_m + 1:02d}-01"
    _ = last_day
    return start, end


def median_composite(stack: np.ndarray) -> np.ndarray:
    """Temporal median over a (time, rows, cols[, bands]) stack.

    NaN-aware (uses ``nanmedian``) so cloud-masked (NaN) pixels do not
    poison the composite.

    Args:
        stack: Array with time on axis 0.

    Returns:
        Median image with the time axis removed.
    """
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"stack must have a time axis, got shape {arr.shape}")
    with np.errstate(all="ignore"):
        return np.nanmedian(arr, axis=0)


def valid_count(stack: np.ndarray) -> np.ndarray:
    """Count valid (non-NaN) observations along the time axis.

    Args:
        stack: Array with time on axis 0.

    Returns:
        Integer count array with the time axis removed.
    """
    arr = np.asarray(stack)
    return np.sum(~np.isnan(arr), axis=0).astype(np.int32)
