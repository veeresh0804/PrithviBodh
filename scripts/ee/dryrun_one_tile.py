#!/usr/bin/env python
"""Fetch ONE small 10 m tile per sensor straight from Earth Engine (no export
task, no Drive copy) and print byte-level checksums to stdout as JSON.

Purpose: prove, before any bulk export, that the assembled image is what the
export task will later write — the tile is sampled on the exact export grid
same CRS, same ``crsTransform``, same dtype), so re-running the script must
reproduce the same SHA-256.

IMPORTANT: the printed hashes are a *pipeline fingerprint* (raw band bytes
sampled via ``Image.sampleRectangle``), **not** the checksum of the eventual
COG file — COG headers/compression depend on the exporter and cannot run
offline.  This run never writes data files (only the optional ``--json-out``
report), and it never claims any export exists.

Auth: without Earth Engine credentials this script prints an error to
stderr and exits 2 without writing anything (the human sequence in
``scripts/ee/README.md`` starts with ``python -m ee.cli.eecli authenticate``).

Examples::

    python scripts/ee/dryrun_one_tile.py --year 2019 --season pre
    python scripts/ee/dryrun_one_tile.py --sensor s1 --orbit ASCENDING
    python scripts/ee/dryrun_one_tile.py --sensor dem --pixels 512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
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
    seasonal_windows,
    submit_image_export,
    tile_region,
    utm_grid,
)
from export_dem import derived_bands, resolve_dem_id  # noqa: E402
from geoeco.ingest.gee_s1 import resolve_orbit  # noqa: E402  (after ee_common)

NODATA = -9999.0
HARD_CAP_PIXELS = 1024  # sampleRectangle practical cap per side (loud clamp)
SENSORS: tuple[str, ...] = ("s2", "s1", "dem")


def cast_array(values: Any, dtype: str, band: str) -> Any:
    """2D nested lists from EE -> contiguous little-endian ndarray of ``dtype``.

    Raises:
        ValueError: If the payload is not a non-empty rectangular 2D list
            (a live-EE contract violation must not be silently hashed).
    """
    import numpy as np

    if not isinstance(values, list) or not values:
        raise ValueError(f"band {band!r}: expected a 2D array, got {type(values)!r}")
    width = len(values[0])
    if width == 0 or any(not isinstance(row, list) or len(row) != width
                          for row in values):
        raise ValueError(f"band {band!r}: payload is not a rectangular 2D array")
    arr = np.asarray(values, dtype=np.float64)
    target = "<i2" if dtype == "int16" else "<f4"
    return np.ascontiguousarray(arr.astype(target))


def fingerprint(band: str, values: Any, dtype: str) -> dict[str, Any]:
    """sha256 + size of one band's raw little-endian payload (fingerprint)."""
    arr = cast_array(values, dtype, band)
    payload = arr.tobytes(order="C")
    return {
        "band": band,
        "height_px": int(arr.shape[0]),
        "width_px": int(arr.shape[1]),
        "dtype": dtype,
        "byte_order": "little",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "meaning": (
            "sha256 of raw band bytes sampled on the export grid "
            "(pipeline fingerprint, NOT a COG file checksum)"
        ),
    }


def build_sensor_image(
    sensor: str,
    geom: Any,
    data: dict[str, Any],
    window: dict[str, Any] | None,
    orbit: str | None,
    dem_id: str | None,
) -> Any:
    """Assemble the export-ready image for one sensor via the export modules.

    Uses the exact same build functions as export_s*.py, so the fingerprint
    covers the real pipeline, not a copy of it.
    """
    if sensor == "s2":
        from export_s2_composites import build_s2_image

        assert window is not None
        return build_s2_image(
            geom, window["start"], window["end"],
            float(data["sentinel2"]["cloud_threshold"]), data["sentinel2"],
        )
    if sensor == "s1":
        from export_s1_composites import build_s1_image

        assert window is not None and orbit is not None
        return build_s1_image(
            geom, window["start"], window["end"], orbit, data["sentinel1"]
        )
    if sensor == "dem":
        from export_dem import build_dem_image

        assert dem_id is not None
        return build_dem_image(dem_id, derived_bands(data))
    raise ValueError(f"unknown sensor {sensor!r}")


def sample_tile(
    image: Any,
    grid: dict[str, Any],
    region: list[float],
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproject onto the export grid, sample one tile, fingerprint each band.

    Returns:
        ``(band_records, scaling_note)``.
    """
    import ee  # lazy

    scaled, note = apply_configured_scales(image, data)
    grid_image = scaled.reprojection(
        crs=str(grid["crs"]),
        crsTransform=[float(v) for v in grid["crs_transform"]],
    )
    bands = [str(b) for b in grid_image.bandNames().getInfo()]
    rect = ee.Geometry.Rectangle(region, str(grid["crs"]), False)
    sampled = grid_image.sampleRectangle(region=rect, defaultValue=NODATA)
    records = []
    for band in bands:
        values = ee.Array(sampled.get(band)).getInfo()
        records.append(fingerprint(band, values, str(note.get("dtype", "float32"))))
    return records, note


def export_name(
    sensor: str, aoi_name: str, window: dict[str, Any] | None,
    orbit: str | None, dem_id: str | None,
) -> tuple[str, str]:
    """(description, file_prefix) matching the export_s*.py task naming."""
    if sensor == "s2":
        assert window is not None
        return (
            f"{aoi_name}-s2-{window['year']}-{window['season']}",
            f"{aoi_name}_s2_{window['year']}_{window['season']}",
        )
    if sensor == "s1":
        assert window is not None and orbit is not None
        return (
            f"{aoi_name}-s1-{orbit[:3].lower()}-{window['year']}-{window['season']}",
            f"{aoi_name}_s1_{orbit}_{window['year']}_{window['season']}",
        )
    assert dem_id is not None
    slug = dem_id.replace("/", "_").lower()
    return f"{aoi_name}-dem", f"{aoi_name}_{slug}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sample one 10 m tile per sensor from Earth Engine and "
        "print per-band SHA-256 fingerprints as JSON (no export task, no file "
        f"writes). Without credentials: exit 2, nothing written ({EE_AUTH_HINT})."
    )
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--sensor", choices=[*SENSORS, "all"], default="all",
                    help="which sensor to sample (default: all)")
    ap.add_argument("--year", type=int, default=None,
                    help="config year (default: first year in config)")
    ap.add_argument("--season", default=None,
                    help="config season key (default: first season in config)")
    ap.add_argument("--orbit", default=None,
                    help="ASCENDING|DESCENDING (overrides config orbit_pass; "
                         "required for s1 while config is TBD)")
    ap.add_argument("--pixels", type=int, default=None,
                    help="tile side in pixels (default: export.internal_tiling "
                         f"from config; capped at {HARD_CAP_PIXELS})")
    ap.add_argument("--dem-asset", default=None,
                    help="override the EE ImageCollection id for the DEM.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--project", default=None, help="GCP project for ee.Initialize.")
    ap.add_argument("--export", action="store_true",
                    help="ALSO submit the full-size export task per sensor "
                         "(adds tasks; fingerprint sampling itself never exports).")
    ap.add_argument("--drive-folder", default=None,
                    help="Google Drive folder (only with --export).")
    ap.add_argument("--bucket", default=None,
                    help="Cloud Storage bucket (only with --export).")
    ap.add_argument("--json-out", default=None,
                    help="Also write the JSON report to this path (written only "
                         "on success).")
    args = ap.parse_args(argv)

    # --- auth FIRST: no config reads, no file writes, exit 2 without creds ---
    ok, reason = ee_available()
    if not ok:
        print(
            f"dryrun: {reason}: needs {EE_AUTH_HINT} before any tile can be "
            "sampled; nothing written",
            file=sys.stderr,
        )
        return 2

    try:
        aoi, data = load_configs(args.config, args.data)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"dryrun: missing/invalid config: {exc}", file=sys.stderr)
        return 2

    seasons = list(data["seasons"])
    season = args.season or seasons[0]
    if season not in data["seasons"]:
        print(f"dryrun: unknown season {season!r}; config has {sorted(seasons)}",
              file=sys.stderr)
        return 2
    years = [int(y) for y in data["years"]]
    year = args.year if args.year is not None else years[0]
    if year not in years:
        print(f"dryrun: year {year} not in config years {years}", file=sys.stderr)
        return 2

    sensors = list(SENSORS) if args.sensor == "all" else [args.sensor]

    try:
        grid = utm_grid(aoi)  # strict: pyproj required to sample the real grid
    except (ImportError, ValueError) as exc:
        print(f"dryrun: grid unavailable: {exc}", file=sys.stderr)
        return 2

    window = None
    orbit = None
    dem_id = None
    try:
        if any(s in sensors for s in ("s2", "s1")):
            window = seasonal_windows(data, year=year, season=season)[0]
        if "s1" in sensors:
            orbit = resolve_orbit(data, args.orbit)
        if "dem" in sensors:
            dem_id = resolve_dem_id(data, args.dem_asset)
    except ValueError as exc:
        print(f"dryrun: {exc}", file=sys.stderr)
        if orbit is None and "s1" in sensors:
            print(
                "dryrun: run `python scripts/ee/availability_report.py` (EE auth "
                "required), pick the orbit, set configs sentinel1.orbit_pass, or "
                "pass --orbit ASCENDING|DESCENDING — the orbit is never guessed",
                file=sys.stderr,
            )
        return 2

    pixels = args.pixels
    if pixels is None:
        pixels = int(data["export"].get("internal_tiling", 256))
    if pixels < 1:
        print(f"dryrun: --pixels must be >= 1, got {pixels}", file=sys.stderr)
        return 2
    if pixels > HARD_CAP_PIXELS:
        print(f"dryrun: NOTE: --pixels {pixels} capped to {HARD_CAP_PIXELS} "
              "(sampleRectangle limit)", file=sys.stderr)
        pixels = HARD_CAP_PIXELS
    try:
        region = tile_region(grid, pixels)
    except ValueError as exc:
        print(f"dryrun: tile does not fit the AOI grid: {exc}", file=sys.stderr)
        return 2

    try:
        initialize_ee(args.project)
    except Exception as exc:  # noqa: BLE001 - auth failures vary by install
        print(
            f"dryrun: EE init failed ({exc}): needs {EE_AUTH_HINT}; nothing written",
            file=sys.stderr,
        )
        return 2

    seed = resolve_seed(data, args.seed)
    aoi_name = str(aoi["name"])
    sensors_out: list[dict[str, Any]] = []
    exit_code = 0
    for sensor in sensors:
        try:
            import ee  # lazy

            geom = ee.Geometry.Rectangle(grid["bounds_utm"], str(grid["crs"]), False)
            image = build_sensor_image(sensor, geom, data, window, orbit, dem_id)
            band_records, note = sample_tile(image, grid, region, data)
            for rec in band_records:
                if rec["width_px"] != pixels or rec["height_px"] != pixels:
                    print(
                        f"dryrun: WARNING {sensor}/{rec['band']}: sampled "
                        f"{rec['width_px']}x{rec['height_px']} px, expected "
                        f"{pixels}x{pixels} — fingerprint still reported, but "
                        "the sample grid differs from the export grid",
                        file=sys.stderr,
                    )
            record: dict[str, Any] = {
                "sensor": sensor,
                "window": ({"year": window["year"], "season": window["season"],
                            "start": window["start"], "end": window["end"]}
                           if window and sensor != "dem" else None),
                "orbit": orbit if sensor == "s1" else None,
                "dem_id": dem_id if sensor == "dem" else None,
                "sample": {
                    "region_utm": [float(v) for v in region],
                    "pixels": [pixels, pixels],
                    "crs": str(grid["crs"]),
                    "crs_transform": [float(v) for v in grid["crs_transform"]],
                    "resolution_m": float(grid["resolution_m"]),
                    "nodata": NODATA,
                },
                "scaling": note,
                "bands": band_records,
            }
            if args.export:
                description, prefix = export_name(
                    sensor, aoi_name, window, orbit, dem_id)
                record["export"] = submit_image_export(
                    image, description=description, file_prefix=prefix,
                    grid=grid, drive_folder=args.drive_folder,
                    bucket=args.bucket, scaling_note=note,
                )
            sensors_out.append(record)
        except (RuntimeError, ImportError) as exc:  # creds/pyproj missing
            print(f"dryrun: {sensor}: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - live EE failures
            print(f"dryrun: {sensor}: sampling failed: {exc}", file=sys.stderr)
            exit_code = 1

    report = {
        "kind": "dryrun-fingerprint",
        "status": "sampled",
        "generated_by": "scripts/ee/dryrun_one_tile.py",
        "aoi": aoi_name,
        "seed": seed,
        "season": season,
        "year": year,
        "orbit": orbit,
        "submitted_export_tasks": bool(args.export),
        "note": (
            "Per-band SHA-256 of raw bytes sampled on the export grid — a "
            "pipeline fingerprint for reproducibility checks, NOT the "
            "checksum of a COG file. No data files were written by this run; "
            "exported files exist only after their EE task finishes "
            "(python -m ee.cli.eecli task list)."
        ),
        "sensors": sensors_out,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"dryrun: wrote {path}", file=sys.stderr)
    if exit_code == 0 and not args.export:
        print(
            "dryrun: fingerprints only — nothing exported; "
            f"re-run the export_s*.py --submit after {EE_AUTH_HINT}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
