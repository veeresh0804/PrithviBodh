"""Season definitions + median composite logic.

Seasons (project default, config ``seasons``):
  pre      = Mar-May (months 3-5)
  monsoon  = Jun-Sep (months 6-9)
  post     = Oct-Dec (months 10-12)
Jan-Feb are excluded (rabi shoulder; avoids thin-year stacks).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

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


REPO = Path(__file__).resolve().parents[2]
DEFAULT_AOI_CFG = REPO / "configs" / "aoi" / "hyderabad.yaml"
DEFAULT_DATA_CFG = REPO / "configs" / "data" / "sentinel.yaml"
SPATIAL_CV_CFG = REPO / "configs" / "eval" / "spatial_cv.yaml"


def resolve_seed(data_cfg: dict[str, Any], explicit: int | None) -> int:
    """Seed from CLI flag, else data config, else spatial_cv.yaml, else 42."""
    if explicit is not None:
        return int(explicit)
    seed = data_cfg.get("seed")
    if isinstance(seed, int):
        return int(seed)
    try:
        from geoeco.utils.config import load_yaml_config

        cv = load_yaml_config(SPATIAL_CV_CFG)
        if isinstance(cv.get("seed"), int):
            return int(cv["seed"])
    except (FileNotFoundError, ValueError, TypeError):
        pass
    return 42


def check_season_definitions(data_cfg: dict[str, Any]) -> dict[str, list[int]]:
    """Assert config season months match the canonical SEASONS (fail loudly if not)."""
    seasons_cfg = data_cfg.get("seasons", {})
    checked: dict[str, list[int]] = {}
    for season in SEASONS:
        if season not in seasons_cfg:
            raise KeyError(f"data config seasons missing {season!r}")
        spec = seasons_cfg[season]
        months = list(spec.get("months", [])) if isinstance(spec, dict) else []
        expected = season_months(season)  # type: ignore[arg-type]
        if months != expected:
            raise ValueError(
                f"season {season!r} months {months} != canonical {expected}"
            )
        checked[season] = months
    extra = sorted(set(seasons_cfg) - set(SEASONS))
    if extra:
        raise ValueError(f"data config has unknown seasons: {extra}")
    return checked


def build_valid_obs_plan(
    aoi: dict[str, Any], data: dict[str, Any], seed: int, year: int | None = None
) -> dict[str, Any]:
    """Seasonal definition check + valid-observation plan (pure Python, no EE)."""
    from geoeco.ingest.gee_s1 import build_s1_collection_spec
    from geoeco.ingest.gee_s2 import build_s2_collection_spec

    months = check_season_definitions(data)
    years = [int(year)] if year is not None else [int(y) for y in data["years"]]
    cloud_thr = float(data["sentinel2"].get("cloud_threshold", 0.6))
    orbit = str(data["sentinel1"].get("orbit_pass", "ASCENDING"))
    windows = []
    for y in years:
        for season in months:
            start, end = season_date_range(y, season)  # type: ignore[arg-type]
            windows.append(
                {
                    "year": y,
                    "season": season,
                    "months": months[season],
                    "start": start,
                    "end": end,
                    "s2_spec": build_s2_collection_spec(start, end, cloud_thr),
                    "s1_spec": build_s1_collection_spec(  # type: ignore[arg-type]
                        start, end, orbit),
                    "track_valid_obs_count": bool(
                        data["sentinel2"].get("track_valid_obs_count", True)
                    ),
                }
            )
    return {
        "aoi": {
            "name": str(aoi.get("name", "aoi")),
            "bounds_wgs84": [float(v) for v in aoi["bounds_wgs84"]],
            "crs": aoi.get("crs"),
            "resolution_m": aoi.get("resolution_m"),
        },
        "years": years,
        "seasons": months,
        "seed": seed,
        "windows": windows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Seasonal definition check + valid-observation plan (offline)."
    )
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--out", default=None,
                    help="Optional dir to write valid_obs_plan.json into.")
    ap.add_argument("--year", type=int, default=None,
                    help="Single year to plan (default: all config years).")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    from geoeco.utils.config import load_yaml_config, require_keys

    try:
        aoi = load_yaml_config(args.config)
        require_keys(aoi, ["bounds_wgs84"], name="aoi config")
        data = load_yaml_config(args.data)
        require_keys(data, ["years", "seasons", "sentinel2", "sentinel1"],
                     name="data config")
    except FileNotFoundError as exc:
        print(f"ingest composites: missing input: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, TypeError) as exc:
        print(f"ingest composites: invalid config: {exc}", file=sys.stderr)
        return 2
    seed = resolve_seed(data, args.seed)
    try:
        plan = build_valid_obs_plan(aoi, data, seed, args.year)
    except (KeyError, ValueError) as exc:
        print(f"ingest composites: season definition check failed: {exc}",
              file=sys.stderr)
        return 2
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "valid_obs_plan.json"
        dest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"plan": str(dest), "windows": len(plan["windows"])},
                         indent=2))
    else:
        print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
