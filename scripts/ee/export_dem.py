"""Copernicus DEM GLO-30 export (Stage 1 / M1, config-driven).

Reads ``configs/data/sentinel.yaml dem`` (source + derived bands) and the
AOI grid config, mosaics the DEM collection with the catalog-recommended
``setDefaultProjection`` recipe, derives elevation + slope, and exports ONE
cloud-optimised GeoTIFF on the same EPSG:32644 / 10 m grid as the S1/S2
composites.

Modes: default/--dry-run prints the plan (no EE); --submit creates the
Earth Engine task (needs credentials, else exit 2 with
``earthengine authenticate``).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# ee_common first: it puts the repo root on sys.path for the geoeco imports.
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
    submit_image_export,
    utm_grid,
    utm_grid_or_status,
)

# config dem.source -> verified Earth Engine dataset id (catalog:
# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30 ).
# NOTE: the literal id "COPERNICUS/DEM_GLO30" does not exist in the catalog;
# the catalog id is "COPERNICUS/DEM/GLO30" (superseded by GLO30_2024_1 —
# override with --dem-asset if the project prefers the new version).
DEM_EE_IDS: dict[str, str] = {
    "Copernicus_GLO30": "COPERNICUS/DEM/GLO30",
}
SUPPORTED_DERIVED: tuple[str, ...] = ("elevation", "slope")


def resolve_dem_id(data_cfg: dict[str, Any], explicit: str | None = None) -> str:
    """Map config ``dem.source`` to an EE dataset id (or use ``--dem-asset``).

    Raises:
        ValueError: If the config source has no known EE mapping and no
            explicit override was given (fail loudly, never guess an id).
    """
    if explicit:
        return str(explicit)
    source = str(data_cfg.get("dem", {}).get("source", ""))
    if source in DEM_EE_IDS:
        return DEM_EE_IDS[source]
    raise ValueError(
        f"dem.source {source!r} has no known EE dataset id; known sources: "
        f"{sorted(DEM_EE_IDS)} — pass --dem-asset <ee.ImageCollection id> "
        "to override"
    )


def derived_bands(data_cfg: dict[str, Any]) -> list[str]:
    """Validate config ``dem.derived`` against :data:`SUPPORTED_DERIVED`.

    Raises:
        ValueError: On an unknown derived band name.
    """
    derived = [str(d) for d in data_cfg.get("dem", {}).get("derived", [])]
    unknown = [d for d in derived if d not in SUPPORTED_DERIVED]
    if unknown:
        raise ValueError(
            f"dem.derived {unknown} unsupported; supported: {list(SUPPORTED_DERIVED)}"
        )
    if not derived:
        raise ValueError("dem.derived is empty — nothing to export")
    return derived


def build_dem_image(dem_id: str, derived: list[str]) -> Any:
    """Mosaic the DEM collection and derive configured bands (needs EE init).

    Follows the catalog recipe: mosaic + ``setDefaultProjection`` of the
    native projection before ``ee.Terrain.slope`` so terrain is computed at
    the native 30 m scale, then reprojected to the 10 m export grid.

    Args:
        dem_id: EE ImageCollection id (e.g. ``COPERNICUS/DEM/GLO30``).
        derived: Subset of :data:`SUPPORTED_DERIVED`.

    Returns:
        ``ee.Image`` with bands ``elevation`` and/or ``slope`` (float32).
    """
    import ee  # lazy

    collection = ee.ImageCollection(dem_id)
    native_proj = collection.first().projection()
    dem = collection.mosaic().setDefaultProjection(native_proj)
    elevation = dem.select("DEM").rename("elevation")
    if derived == ["elevation"]:
        return elevation.toFloat()
    parts = []
    if "elevation" in derived:
        parts.append(elevation)
    if "slope" in derived:
        parts.append(ee.Terrain.slope(elevation).rename("slope"))
    out = parts[0] if len(parts) == 1 else parts[0].addBands(parts[1])
    return out.toFloat()


def build_plan(
    aoi: dict[str, Any],
    data: dict[str, Any],
    seed: int,
    dem_id: str,
    *,
    drive_folder: str | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Pure-Python export plan (no EE): single static DEM export on the grid.

    Raises:
        ValueError/KeyError: On invalid config.
    """
    derived = derived_bands(data)
    aoi_name = str(aoi["name"])
    slug = dem_id.replace("/", "_").lower()
    filename = f"{aoi_name}_{slug}.tif"
    destination = (
        f"gs://{bucket}" if bucket
        else f"Google Drive/{drive_folder}" if drive_folder
        else "Google Drive (root)"
    )
    dem_cfg = data["dem"]
    return {
        "sensor": "copernicus-dem",
        "status": "planned",
        "exported": False,
        "generated_by": "scripts/ee/export_dem.py",
        "aoi": {
            "name": aoi_name,
            "bounds_wgs84": [float(v) for v in aoi["bounds_wgs84"]],
            "crs": str(aoi["crs"]),
            "resolution_m": float(aoi["resolution_m"]),
        },
        "seed": seed,
        "query": {
            "dem_id": dem_id,
            "config_source": str(dem_cfg.get("source")),
            "derived": derived,
            "target_resolution_m": dem_cfg.get("target_resolution_m"),
            "native_resolution_m": 30,
            "upsample_note": (
                "DEM is native 30 m exported on the 10 m AOI grid; "
                "EE resamples with its default method (no resampling method "
                "is configured — see docs/stage1_ee_setup.md)"
            ),
        },
        "grid": utm_grid_or_status(aoi),
        "windows": [
            {"year": None, "season": "static", "filename": filename,
             "start": None, "end": None}
        ],
        "export_options": {
            "fileFormat": "GeoTIFF",
            "cloudOptimized": True,
            "destination": destination,
            "drive_folder": drive_folder,
            "format": str(data["export"].get("format")),
            "internal_tiling": data["export"].get("internal_tiling"),
            "scaling": scaling_note(data),
        },
        "task_count": 1,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plan/submit the Copernicus DEM GLO-30 export (elevation/"
        f"slope, AOI grid). --submit needs {EE_AUTH_HINT}."
    )
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--dem-asset", default=None,
                    help="Override the EE ImageCollection id for the DEM.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--drive-folder", default=None)
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--project", default=None, help="GCP project for ee.Initialize.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print the plan only (default behaviour).")
    mode.add_argument("--submit", action="store_true",
                      help="Create the Earth Engine export task (needs auth).")
    args = ap.parse_args(argv)

    try:
        aoi, data = load_configs(args.config, args.data)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"export dem: missing/invalid config: {exc}", file=sys.stderr)
        return 2
    seed = resolve_seed(data, args.seed)
    try:
        dem_id = resolve_dem_id(data, args.dem_asset)
        plan = build_plan(
            aoi, data, seed, dem_id,
            drive_folder=args.drive_folder, bucket=args.bucket,
        )
    except (KeyError, ValueError, TypeError) as exc:
        print(f"export dem: invalid plan: {exc}", file=sys.stderr)
        return 2

    if not args.submit:
        print(json.dumps(plan, indent=2))
        print(
            "export dem: plan only — nothing exported (exported=false); "
            "re-run with --submit after `earthengine authenticate`",
            file=sys.stderr,
        )
        return 0

    ok, reason = ee_available()
    if not ok:
        print(
            f"export dem: {reason}: needs {EE_AUTH_HINT} before --submit; "
            "nothing exported",
            file=sys.stderr,
        )
        return 2
    try:
        initialize_ee(args.project)
    except Exception as exc:
        print(
            f"export dem: EE init failed ({exc}): needs {EE_AUTH_HINT}; "
            "nothing exported",
            file=sys.stderr,
        )
        return 2
    try:
        grid = utm_grid(aoi)  # strict: pyproj required to submit
        derived = derived_bands(data)
        image = build_dem_image(dem_id, derived)
        image, note = apply_configured_scales(image, data)
        filename = str(plan["windows"][0]["filename"])
        info = submit_image_export(
            image,
            description=f"{aoi['name']}-dem",
            file_prefix=filename[:-len(".tif")],
            grid=grid,
            drive_folder=args.drive_folder,
            bucket=args.bucket,
            scaling_note=note,
        )
    except (RuntimeError, ImportError) as exc:
        print(f"export dem: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"export dem: EE export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "submitted": [info],
        "task_count": 1,
        "note": "task queued on Earth Engine; the file exists only after the "
                "task finishes (check: python -m ee.cli.eecli task list)",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
