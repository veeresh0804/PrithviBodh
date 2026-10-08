"""Seasonal Sentinel-2 L2A composite export (Stage 1 / M1, config-driven).

Pipeline per config year/season window:
  COPERNICUS/S2_SR_HARMONIZED joined with Cloud Score+
  (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED) -> keep pixels with
  ``cs >= sentinel2.cloud_threshold`` -> 20 m bands resampled per config ->
  median composite + ``valid_count`` -> reflectance (``/ 0.0001``) ->
  indices from config ``sentinel2.indices`` (geoeco.ingest.gee_indices) ->
  one COG (cloudOptimized GeoTIFF) export per window on the AOI's
  EPSG:32644 / 10 m grid (grid + CRS come from the AOI config).

Modes:
  * default / ``--dry-run``: print the export PLAN as JSON (no EE needed,
    exit 0, ``exported: false`` — claims nothing).
  * ``--submit``: create the Earth Engine export tasks; without credentials
    this fails loudly with ``earthengine authenticate`` and exit 2.

No AOI/dates/thresholds are hard-coded here: everything comes from
configs/aoi/*.yaml + configs/data/sentinel.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ee_common import (
    DEFAULT_AOI_CFG,
    DEFAULT_DATA_CFG,
    EE_AUTH_HINT,
    apply_configured_scales,
    ee_available,
    initialize_ee,
    load_configs,
    resolve_seed,
    scaling_note,
    seasonal_windows,
    submit_image_export,
    utm_grid,
    utm_grid_or_status,
)

S2_BANDS_DEFAULT: tuple[str, ...] = (
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
)
RESAMPLE_MODES: tuple[str, ...] = ("nearest", "bilinear", "cubic", "bicubic")


def build_s2_image(geom: Any, start: str, end: str, cloud_thr: float,
                   s2cfg: dict[str, Any]) -> Any:
    """Build one season's cloud-masked S2 composite + indices (needs EE init).

    Args:
        geom: ``ee.Geometry`` AOI.
        start: ISO start date (inclusive).
        end: ISO end date (exclusive).
        cloud_thr: Cloud Score+ clear threshold (keep ``cs >= thr``).
        s2cfg: ``data_cfg["sentinel2"]`` mapping.

    Returns:
        ``ee.Image`` with config bands (reflectance), config indices, and
        ``valid_count`` when ``track_valid_obs_count`` is set.
    """
    from geoeco.ingest import gee_indices
    from geoeco.ingest.gee_s2 import build_s2_collection, s2_seasonal_composite

    collection = build_s2_collection(geom, start, end, cloud_thr)
    resample = str(s2cfg.get("resample_20m_to_10m", "bilinear"))
    if resample not in RESAMPLE_MODES:
        raise ValueError(
            f"sentinel2.resample_20m_to_10m {resample!r} not one of {RESAMPLE_MODES}"
        )
    collection = collection.map(lambda img: img.resample(resample))
    composite = s2_seasonal_composite(collection)

    bands = [str(b) for b in s2cfg.get("bands", list(S2_BANDS_DEFAULT))]
    reflectance_scale = float(s2cfg.get("reflectance_scale", 0.0001))
    reflectance = composite.select(bands).divide(reflectance_scale)
    indices = gee_indices.supported_indices(list(s2cfg.get("indices", [])))
    out = gee_indices.add_index_bands(reflectance, indices)
    if s2cfg.get("track_valid_obs_count", True):
        out = out.addBands(composite.select("valid_count"))
    return out


def build_plan(
    aoi: dict[str, Any],
    data: dict[str, Any],
    seed: int,
    *,
    year: int | None = None,
    season: str | None = None,
    cloud_thr: float | None = None,
    drive_folder: str | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Pure-Python export plan (no EE): windows, grid, bands, filenames.

    Raises:
        ValueError: On out-of-range cloud threshold or invalid seasons.
        KeyError: On unknown season key.
    """
    s2cfg = data["sentinel2"]
    thr = float(
        cloud_thr if cloud_thr is not None else s2cfg.get("cloud_threshold", 0.6)
    )
    if not 0.0 < thr < 1.0:
        raise ValueError(f"cloud threshold must be in (0,1), got {thr}")
    windows = seasonal_windows(data, year=year, season=season)
    aoi_name = str(aoi["name"])
    for w in windows:
        w["filename"] = f"{aoi_name}_s2_{w['year']}_{w['season']}.tif"
    destination = (
        f"gs://{bucket}" if bucket
        else f"Google Drive/{drive_folder}" if drive_folder
        else "Google Drive (root)"
    )
    return {
        "sensor": "sentinel-2",
        "status": "planned",
        "exported": False,
        "generated_by": "scripts/ee/export_s2_composites.py",
        "aoi": {
            "name": aoi_name,
            "bounds_wgs84": [float(v) for v in aoi["bounds_wgs84"]],
            "crs": str(aoi["crs"]),
            "resolution_m": float(aoi["resolution_m"]),
        },
        "seed": seed,
        "query": {
            "source": str(s2cfg["source"]),
            "cloud_mask_source": str(s2cfg["cloud_mask_source"]),
            "cloud_score_band": str(s2cfg.get("cloud_score_band", "cs")),
            "cloud_threshold": thr,
            "bands": [str(b) for b in s2cfg.get("bands", list(S2_BANDS_DEFAULT))],
            "indices": [str(i) for i in s2cfg.get("indices", [])],
            "resample_20m_to_10m": str(s2cfg.get("resample_20m_to_10m", "bilinear")),
            "reflectance_scale": float(s2cfg.get("reflectance_scale", 0.0001)),
            "composite": str(s2cfg.get("composite", "median")),
            "track_valid_obs_count": bool(s2cfg.get("track_valid_obs_count", True)),
        },
        "grid": utm_grid_or_status(aoi),
        "windows": windows,
        "export_options": {
            "fileFormat": "GeoTIFF",
            "cloudOptimized": True,
            "destination": destination,
            "drive_folder": drive_folder,
            "format": str(data["export"].get("format")),
            "internal_tiling": data["export"].get("internal_tiling"),
            "scaling": scaling_note(data),
        },
        "task_count": len(windows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plan/submit Sentinel-2 L2A seasonal COG exports. "
        "Default prints a plan (no EE needed); --submit creates Earth Engine "
        f"tasks and requires credentials ({EE_AUTH_HINT})."
    )
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--year", type=int, default=None,
                    help="Single config year (default: all config years).")
    ap.add_argument("--season", default=None,
                    help="Single config season key (default: all seasons).")
    ap.add_argument("--cloud-thr", type=float, default=None,
                    help="Override sentinel2.cloud_threshold (0..1).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--drive-folder", default=None,
                    help="Google Drive folder for exports (default: Drive root).")
    ap.add_argument("--bucket", default=None,
                    help="Cloud Storage bucket (overrides --drive-folder).")
    ap.add_argument("--project", default=None, help="GCP project for ee.Initialize.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print the plan only (default behaviour).")
    mode.add_argument("--submit", action="store_true",
                      help="Create the Earth Engine export tasks (needs auth).")
    args = ap.parse_args(argv)

    try:
        aoi, data = load_configs(args.config, args.data)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"export s2: missing/invalid config: {exc}", file=sys.stderr)
        return 2
    if args.season is not None and args.season not in data["seasons"]:
        print(f"export s2: unknown season {args.season!r}; config has "
              f"{sorted(data['seasons'])}", file=sys.stderr)
        return 2
    seed = resolve_seed(data, args.seed)
    try:
        plan = build_plan(
            aoi, data, seed,
            year=args.year, season=args.season, cloud_thr=args.cloud_thr,
            drive_folder=args.drive_folder, bucket=args.bucket,
        )
    except (KeyError, ValueError, TypeError) as exc:
        print(f"export s2: invalid plan: {exc}", file=sys.stderr)
        return 2

    if not args.submit:
        print(json.dumps(plan, indent=2))
        print(
            "export s2: plan only — nothing exported (exported=false); "
            "re-run with --submit after `earthengine authenticate`",
            file=sys.stderr,
        )
        return 0

    ok, reason = ee_available()
    if not ok:
        print(
            f"export s2: {reason}: needs {EE_AUTH_HINT} before --submit; "
            "nothing exported",
            file=sys.stderr,
        )
        return 2
    try:
        initialize_ee(args.project)
    except Exception as exc:
        print(
            f"export s2: EE init failed ({exc}): needs {EE_AUTH_HINT}; "
            "nothing exported",
            file=sys.stderr,
        )
        return 2
    try:
        import ee  # lazy

        grid = utm_grid(aoi)  # strict: pyproj required to submit
        geom = ee.Geometry.Rectangle(grid["bounds_utm"], str(grid["crs"]), False)
        s2cfg = data["sentinel2"]
        thr = float(plan["query"]["cloud_threshold"])
        submitted = []
        for w in plan["windows"]:
            image = build_s2_image(geom, w["start"], w["end"], thr, s2cfg)
            image, note = apply_configured_scales(image, data)
            prefix = str(w["filename"])[:-len(".tif")]
            info = submit_image_export(
                image,
                description=f"{aoi['name']}-s2-{w['year']}-{w['season']}",
                file_prefix=prefix,
                grid=grid,
                drive_folder=args.drive_folder,
                bucket=args.bucket,
                scaling_note=note,
            )
            submitted.append(info)
    except (RuntimeError, ImportError) as exc:  # creds/pyproj missing
        print(f"export s2: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"export s2: EE export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "submitted": submitted,
        "task_count": len(submitted),
        "note": "tasks queued on Earth Engine; files exist only after the "
                "tasks finish (check: python -m ee.cli.eecli task list)",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
