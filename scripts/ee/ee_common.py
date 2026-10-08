"""Shared helpers for the scripts/ee Earth Engine export suite (Stage 1 / M1).

Design rules (repo standing orders):
  * All AOI/dates/seasons/thresholds come from configs — never hard-coded.
  * No credentials -> loud failure (``earthengine authenticate``, exit 2).
  * No network at import time; ``ee`` is imported lazily inside functions
    that actually submit/query, so ``--help`` and unit tests work offline.
  * Nothing here ever fabricates scene counts or export artefacts.

Import order matters: this module puts the repo root on ``sys.path`` before
importing ``geoeco`` (the package is not pip-installed on every machine), so
scripts must ``import ee_common``/``from ee_common import ...`` before any
``geoeco`` import.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geoeco.utils.config import load_yaml_config, require_keys  # noqa: E402

DEFAULT_AOI_CFG = REPO / "configs" / "aoi" / "hyderabad.yaml"
DEFAULT_DATA_CFG = REPO / "configs" / "data" / "sentinel.yaml"
SPATIAL_CV_CFG = REPO / "configs" / "eval" / "spatial_cv.yaml"
EE_AUTH_HINT = "earthengine authenticate"

# Export defaults that only ever widen limits (never invent pixel values).
MAX_EXPORT_PIXELS = 10**13


def ee_available() -> tuple[bool, str]:
    """Check EE credentials without any network use (credential-file check only)."""
    try:
        from ee import oauth

        cred_path = oauth.get_credentials_path()
    except ImportError:
        return False, "earthengine-api not installed"
    except Exception as exc:  # oauth internals may vary by version
        return False, f"EE credential check failed: {exc}"
    if cred_path and os.path.isfile(cred_path):
        return True, "EE credentials present"
    return False, "no EE credentials found"


def require_ee() -> None:
    """Raise loudly when an EE call is attempted without credentials (no fake outputs)."""
    ok, reason = ee_available()
    if not ok:
        raise RuntimeError(
            f"EE call requires credentials ({reason}): "
            f"needs {EE_AUTH_HINT} before any network use"
        )


def initialize_ee(project: str | None = None) -> None:
    """Initialise the Earth Engine API (requires credentials; network call)."""
    require_ee()
    import ee  # lazy

    ee.Initialize(project=project) if project else ee.Initialize()


def load_configs(aoi_path: str | Path, data_path: str | Path) -> tuple[dict, dict]:
    """Load + validate AOI and data configs (fail loudly on missing inputs)."""
    aoi = load_yaml_config(aoi_path)
    require_keys(aoi, ["name", "bounds_wgs84", "crs", "resolution_m"], name="aoi config")
    data = load_yaml_config(data_path)
    require_keys(
        data,
        ["years", "seasons", "sentinel2", "sentinel1", "dem", "export"],
        name="data config",
    )
    return aoi, data


def resolve_seed(data_cfg: dict[str, Any], explicit: int | None) -> int:
    """Seed from CLI flag, else data config, else spatial_cv.yaml, else 42."""
    if explicit is not None:
        return int(explicit)
    seed = data_cfg.get("seed")
    if isinstance(seed, int):
        return int(seed)
    try:
        cv = load_yaml_config(SPATIAL_CV_CFG)
        if isinstance(cv.get("seed"), int):
            return int(cv["seed"])
    except (FileNotFoundError, ValueError, TypeError):
        pass
    return 42


# --------------------------------------------------------------------------
# Season windows: derived FROM CONFIG months only (never from a hard-coded map).
# --------------------------------------------------------------------------


def season_months_of(data_cfg: dict[str, Any], season: str) -> list[int]:
    """Return the configured inclusive month list for one season.

    Args:
        data_cfg: Parsed configs/data/sentinel.yaml mapping.
        season: Season key (e.g. ``"pre"``).

    Returns:
        Month numbers as ints.

    Raises:
        KeyError: If the season key is absent from the config.
        ValueError: If ``months`` is missing or not a list.
    """
    seasons = data_cfg.get("seasons", {})
    if season not in seasons:
        raise KeyError(f"season {season!r} not in config seasons {sorted(seasons)}")
    spec = seasons[season]
    if not isinstance(spec, dict) or "months" not in spec:
        raise ValueError(f"season {season!r} config must be a mapping with 'months'")
    return [int(m) for m in spec["months"]]


def season_window(year: int, season: str, months: list[int]) -> tuple[str, str]:
    """Config-months -> (start, end-exclusive) ISO dates for one season/year.

    Args:
        year: Calendar year (e.g. 2019, 2025).
        season: Season name (used in error messages only).
        months: Inclusive month numbers from the config.

    Returns:
        ``(start_iso, end_exclusive_iso)``; end is the first day after the
        last configured month (``year+1-01-01`` when the season ends in Dec).

    Raises:
        ValueError: If months are empty, out of 1..12, duplicated,
            unsorted, or non-contiguous (a gap could not form a single
            date window without silently pulling in excluded months).
    """
    ms = [int(m) for m in months]
    if not ms:
        raise ValueError(f"season {season!r}: empty 'months' list in config")
    if any(m < 1 or m > 12 for m in ms):
        raise ValueError(f"season {season!r}: months must be 1..12, got {ms}")
    if ms != sorted(ms) or len(set(ms)) != len(ms):
        raise ValueError(f"season {season!r}: months must be sorted+unique, got {ms}")
    if ms != list(range(ms[0], ms[-1] + 1)):
        raise ValueError(
            f"season {season!r}: months {ms} are not contiguous; a single "
            "start-end window would silently include the gap months"
        )
    start = f"{year}-{ms[0]:02d}-01"
    end = f"{year + 1}-01-01" if ms[-1] == 12 else f"{year}-{ms[-1] + 1:02d}-01"
    return start, end


def seasonal_windows(
    data_cfg: dict[str, Any], year: int | None = None, season: str | None = None
) -> list[dict[str, Any]]:
    """Build year x season windows from config (years, seasons.months).

    Args:
        data_cfg: Parsed data config.
        year: Restrict to one year (default: every config year).
        season: Restrict to one season key (default: every config season).

    Returns:
        dicts with ``year``, ``season``, ``months``, ``start``, ``end``.
    """
    years = [int(year)] if year is not None else [int(y) for y in data_cfg["years"]]
    seasons = [season] if season else list(data_cfg["seasons"].keys())
    windows: list[dict[str, Any]] = []
    for y in years:
        for s in seasons:
            months = season_months_of(data_cfg, s)
            start, end = season_window(y, s, months)
            windows.append(
                {"year": y, "season": s, "months": months, "start": start, "end": end}
            )
    return windows


# --------------------------------------------------------------------------
# Grid: AOI WGS84 bounds -> processing CRS, snapped OUTWARD to the 10 m grid.
# --------------------------------------------------------------------------


def grid_from_bounds_utm(bounds_utm: tuple[float, float, float, float],
                         resolution_m: float) -> dict[str, Any]:
    """Pure grid description from already-projected (UTM) bounds.

    Bounds must already be aligned to ``resolution_m`` (see :func:`utm_grid`);
    the function validates that alignment instead of assuming it.

    Args:
        bounds_utm: ``(minx, miny, maxx, maxy)`` in the processing CRS (metres).
        resolution_m: Pixel size in metres.

    Returns:
        dict with ``resolution_m``, ``bounds_utm``, ``width_px``, ``height_px``,
        ``crs_transform`` (EE/GDAL affine ``[res, 0, x0, 0, -res, y1]`` —
        column-major: x = res*col + x0, y = -res*row + y1) and ``affine_gdal``
        tuple ``(x0, res, 0, y1, 0, -res)`` (GDAL geotransform order).

    Raises:
        ValueError: On non-positive resolution, degenerate bounds, or bounds
            that are not exact multiples of the resolution.
    """
    minx, miny, maxx, maxy = (float(v) for v in bounds_utm)
    res = float(resolution_m)
    if res <= 0:
        raise ValueError(f"resolution_m must be > 0, got {res}")
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"degenerate bounds: {(minx, miny, maxx, maxy)}")
    for value in (minx, miny, maxx, maxy):
        if not math.isclose(value / res, round(value / res), abs_tol=1e-6):
            raise ValueError(
                f"bounds value {value} is not aligned to resolution {res}"
            )
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    if width < 1 or height < 1:
        raise ValueError(f"grid would be empty: {(width, height)} px")
    return {
        "resolution_m": res,
        "bounds_utm": [minx, miny, maxx, maxy],
        "width_px": width,
        "height_px": height,
        "crs_transform": [res, 0.0, minx, 0.0, -res, maxy],
        "affine_gdal": (minx, res, 0.0, maxy, 0.0, -res),
    }


def utm_grid(aoi_cfg: dict[str, Any]) -> dict[str, Any]:
    """Project AOI WGS84 bounds into the config CRS and snap to the 10 m grid.

    All four rectangle edges are densified (51 samples/edge) before
    projection because projected edges of a lon/lat rectangle are curved;
    then each extent is snapped OUTWARD to a multiple of ``resolution_m``
    so the AOI is fully covered and pixel edges land on exact grid lines.

    Args:
        aoi_cfg: Parsed AOI config (needs bounds_wgs84, crs, resolution_m).

    Returns:
        Grid dict from :func:`grid_from_bounds_utm` plus ``crs``.

    Raises:
        ImportError: If ``pyproj`` is not installed.
        ValueError: On invalid config bounds/resolution.
    """
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "utm_grid needs pyproj: pip install pyproj"
        ) from exc

    crs = str(aoi_cfg["crs"])
    res = float(aoi_cfg["resolution_m"])
    minlon, minlat, maxlon, maxlat = (float(v) for v in aoi_cfg["bounds_wgs84"])
    if maxlon <= minlon or maxlat <= minlat:
        raise ValueError(f"invalid bounds_wgs84: {(minlon, minlat, maxlon, maxlat)}")

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lons: list[float] = []
    lats: list[float] = []
    n = 51
    for i in range(n):
        t = i / (n - 1)
        lons.extend([minlon + t * (maxlon - minlon), minlon + t * (maxlon - minlon),
                     minlon, maxlon])
        lats.extend([minlat, maxlat, minlat + t * (maxlat - minlat),
                     minlat + t * (maxlat - minlat)])
    xs, ys = transformer.transform(lons, lats)
    minx = math.floor(min(xs) / res) * res
    maxx = math.ceil(max(xs) / res) * res
    miny = math.floor(min(ys) / res) * res
    maxy = math.ceil(max(ys) / res) * res
    grid = grid_from_bounds_utm((minx, miny, maxx, maxy), res)
    grid["crs"] = crs
    return grid


def utm_grid_or_status(aoi_cfg: dict[str, Any]) -> dict[str, Any]:
    """:func:`utm_grid`, or a loud status dict when pyproj is unavailable.

    Used by ``--dry-run`` plans so they still render offline; ``--submit``
    paths must call :func:`utm_grid` (strict) instead.
    """
    try:
        return utm_grid(aoi_cfg)
    except ImportError as exc:
        return {
            "status": "unavailable",
            "requires": "pyproj",
            "note": str(exc),
            "crs": str(aoi_cfg.get("crs")),
            "resolution_m": float(aoi_cfg.get("resolution_m", 10.0)),
        }


def tile_region(grid: dict[str, Any], pixels: int) -> list[float]:
    """Centre square tile of ``pixels`` x ``pixels`` snapped to the grid.

    Args:
        grid: Output of :func:`utm_grid` / :func:`grid_from_bounds_utm`.
        pixels: Tile side in pixels.

    Returns:
        ``[minx, miny, maxx, maxy]`` in the grid CRS: exactly ``pixels``
        pixels per side, aligned to grid lines, inside the grid bounds.

    Raises:
        ValueError: If the tile does not fit the grid.
    """
    px = int(pixels)
    width = int(grid["width_px"])
    height = int(grid["height_px"])
    if px < 1 or px > min(width, height):
        raise ValueError(f"tile {px}px does not fit grid {width}x{height}px")
    res = float(grid["resolution_m"])
    minx, miny, maxx, maxy = grid["bounds_utm"]
    col0 = (width - px) // 2
    row0 = (height - px) // 2
    x0 = minx + col0 * res
    y1 = maxy - row0 * res
    x1 = x0 + px * res
    y0 = y1 - px * res
    return [x0, y0, x1, y1]


# --------------------------------------------------------------------------
# Export submission (network; caller must have called initialize_ee first).
# --------------------------------------------------------------------------


def scaling_note(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Provenance for the export dtype/scale (pure; no EE needed).

    ``configs/data/sentinel.yaml export.format`` says ``COG_int16_scaled``,
    but per-band scale factors only exist if the config provides
    ``export.scales`` (mapping ``band -> factor``, optional ``default``).
    With no factors configured the export stays native float32 and says so
    loudly — scale factors are never invented here.

    Args:
        data_cfg: Parsed data config.

    Returns:
        JSON-serialisable note with ``dtype`` and ``int16_scaled``.
    """
    scales = data_cfg.get("export", {}).get("scales")
    if not isinstance(scales, dict) or not scales:
        return {
            "dtype": "float32",
            "int16_scaled": False,
            "note": (
                "configs export.format is "
                f"{data_cfg.get('export', {}).get('format')!r} but "
                "export.scales has no per-band factors; exporting native "
                "float32 (scale factors are not invented). Add export.scales "
                "to configs/data/sentinel.yaml to enable int16 scaling."
            ),
        }
    default = float(scales.get("default", 1.0))
    used: dict[str, float] = {"default": default}
    for band, factor in scales.items():
        if band != "default":
            used[str(band)] = float(factor)
    return {"dtype": "int16", "int16_scaled": True, "scales": used, "nodata": -9999}


def apply_configured_scales(
    image: Any, data_cfg: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Apply optional config-driven int16 scaling, or explain why not.

    Args:
        image: ``ee.Image`` to scale.
        data_cfg: Parsed data config.

    Returns:
        ``(image, note)`` where ``note`` is :func:`scaling_note` output.
    """
    note = scaling_note(data_cfg)
    if not note.get("int16_scaled"):
        return image, note
    import ee  # lazy

    factors = dict(note["scales"])
    default = float(factors.pop("default", 1.0))
    scaled = ee.Image(image.multiply(default))
    for band, factor in factors.items():
        scaled = scaled.addBands(
            image.select(band).multiply(float(factor)).toInt16().rename(band),
            overwrite=True,
        )
    return scaled.toInt16(), note


def submit_image_export(
    image: Any,
    description: str,
    file_prefix: str,
    grid: dict[str, Any],
    *,
    drive_folder: str | None = None,
    bucket: str | None = None,
    scaling_note: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create + start ONE cloud-optimised GeoTIFF export task (network).

    Args:
        image: ``ee.Image`` to export.
        description: EE task description (alnum + ``-_.``).
        file_prefix: Output filename prefix (``.tif`` appended by EE).
        grid: Output of :func:`utm_grid` (provides CRS + exact transform).
        drive_folder: Google Drive folder (None = Drive root).
        bucket: If set, export to Cloud Storage instead of Drive.
        scaling_note: Provenance from :func:`apply_configured_scales`.

    Returns:
        JSON-serialisable task record (id, destination, grid); the file
        exists ONLY after the EE task finishes — this dict never claims
        otherwise.
    """
    import ee  # lazy

    region = ee.Geometry.Rectangle(grid["bounds_utm"], str(grid["crs"]), False)
    params: dict[str, Any] = dict(
        image=image,
        description=description,
        region=region,
        crs=str(grid["crs"]),
        crsTransform=[float(v) for v in grid["crs_transform"]],
        fileFormat="GeoTIFF",
        maxPixels=MAX_EXPORT_PIXELS,
        fileNamePrefix=file_prefix,
        formatOptions={"cloudOptimized": True},
    )
    if scaling_note and scaling_note.get("int16_scaled"):
        params["formatOptions"] = {
            "cloudOptimized": True,
            "noData": int(scaling_note.get("nodata", -9999)),
        }
    if bucket:
        task = ee.batch.Export.image.toCloudStorage(bucket=str(bucket), **params)
        destination = f"gs://{bucket}/{file_prefix}.tif"
    else:
        task = ee.batch.Export.image.toDrive(folder=drive_folder, **params)
        folder_part = f"/{drive_folder}" if drive_folder else ""
        destination = f"Google Drive{folder_part}/{file_prefix}.tif"
    task.start()
    try:
        task_id = task.status().get("id")
    except Exception:  # status() is informational only
        task_id = None
    return {
        "description": description,
        "task_id": task_id,
        "destination": destination,
        "submitted": True,
        "exists_on_disk": False,  # becomes True only when the EE task completes
        "grid": {
            "crs": grid["crs"],
            "resolution_m": grid["resolution_m"],
            "crs_transform": grid["crs_transform"],
            "width_px": grid["width_px"],
            "height_px": grid["height_px"],
        },
        "scaling": scaling_note or {"dtype": "float32", "int16_scaled": False},
    }
