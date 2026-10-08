"""Seasonal Sentinel-1 GRD composite export (Stage 1 / M1, config-driven).

Pipeline per config year/season window:
  COPERNICUS/S1_GRD IW VV+VH (orbit from config / --orbit) -> border/edge
  noise mask (IW incidence-angle window + scene-border buffer) -> speckle
  filter (in-EE focal-median approximation; see SPECKLE_REFINED_LEE_NOTE) ->
  seasonal features VV, VH, VVVH (=VV-VH), VH_std via
  geoeco.ingest.gee_s1.s1_seasonal_features -> one COG export per window on
  the AOI's EPSG:32644 / 10 m grid.

The orbit is REQUIRED: config sentinel1.orbit_pass is a TBD placeholder
until the availability report picks one, so this script fails loudly
(exit 2) instead of guessing an orbit.

Modes: default/--dry-run prints the plan (no EE); --submit creates tasks
(needs credentials, else exit 2 with `earthengine authenticate`).
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
    seasonal_windows,
    submit_image_export,
    utm_grid,
    utm_grid_or_status,
)
from geoeco.ingest.gee_s1 import resolve_orbit  # noqa: E402  (after ee_common)

# Border/edge-noise constants from the documented production one-liner in
# geoeco/ingest/gee_s1.py::mask_s1_edge_noise (read-only reference file).
ANGLE_MIN_DEG = 31.0
ANGLE_MAX_DEG = 46.0
DEFAULT_BORDER_BUFFER_M = 500.0  # same source (stub signature default)
DEFAULT_SPECKLE_KERNEL_M = 70.0  # geoeco/ingest/gee_s1.py speckle stub default

SPECKLE_REFINED_LEE_NOTE = (
    "config sentinel1.speckle_filter='RefinedLee' but Google Earth Engine "
    "has no native Refined Lee operator (documented in "
    "geoeco/ingest/gee_s1.py): applying the in-EE focal-median "
    "approximation on linear power. A true Refined Lee needs SNAP/external "
    "pre-processing — see docs/stage1_ee_setup.md."
)
SUPPORTED_SPECKLE: tuple[str, ...] = ("RefinedLee", "focal_median", "none")
REQUIRED_POLARISATIONS: tuple[str, ...] = ("VV", "VH")


def speckle_spec(s1cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate + describe the config speckle setting (pure, no EE).

    Raises:
        ValueError: On an unknown ``speckle_filter`` value, or when the
            configured polarisations cannot produce the configured
            features (VV/VH ratio + VH stats need both VV and VH).
    """
    mode = str(s1cfg.get("speckle_filter", "focal_median"))
    if mode not in SUPPORTED_SPECKLE:
        raise ValueError(
            f"sentinel1.speckle_filter {mode!r} not one of {SUPPORTED_SPECKLE}"
        )
    pols = [str(p) for p in s1cfg.get("polarizations", list(REQUIRED_POLARISATIONS))]
    missing = [p for p in REQUIRED_POLARISATIONS if p not in pols]
    if missing and mode != "none":
        raise ValueError(
            f"sentinel1.polarizations {pols} missing {missing}: the config "
            "features (VV/VH ratio, VH_std) require both VV and VH"
        )
    kernel_m = float(s1cfg.get("speckle_kernel_m", DEFAULT_SPECKLE_KERNEL_M))
    return {
        "mode": mode,
        "kernel_m": kernel_m,
        "note": SPECKLE_REFINED_LEE_NOTE if mode == "RefinedLee" else None,
    }


def mask_edge_noise(image: Any, border_buffer_m: float = DEFAULT_BORDER_BUFFER_M) -> Any:
    """Mask S1 scene border/edge noise.

    Pixels outside the valid IW incidence-angle window
    (``ANGLE_MIN_DEG``..``ANGLE_MAX_DEG``) or within ``border_buffer_m``
    of the scene border are masked. Bounds mirror the documented
    production approach in ``geoeco/ingest/gee_s1.py``.

    Args:
        image: S1 GRD scene ``ee.Image`` (must carry the ``angle`` band).
        border_buffer_m: Scene-border buffer in metres (0 disables).

    Returns:
        Edge-masked ``ee.Image``.
    """
    import ee  # lazy

    angle = image.select("angle")
    out = image.updateMask(angle.gt(ANGLE_MIN_DEG).And(angle.lt(ANGLE_MAX_DEG)))
    if border_buffer_m and border_buffer_m > 0:
        shrunk = image.geometry().buffer(-float(border_buffer_m))
        out = out.updateMask(ee.Image.constant(1).clip(shrunk))
    return out


def speckle_filter_image(image: Any, kernel_m: float = DEFAULT_SPECKLE_KERNEL_M) -> Any:
    """In-EE speckle filter: focal median on linear power, back to dB.

    This is the documented GEE approximation for the config's
    ``RefinedLee`` (GEE has no native Refined Lee operator). Returns the
    filtered ``VV``/``VH`` dB bands with masks preserved.

    Args:
        image: S1 GRD scene ``ee.Image`` with ``VV``/``VH`` in dB.
        kernel_m: Square kernel size in metres (~7 px at 10 m).

    Returns:
        ``ee.Image`` with smoothed ``VV``/``VH`` bands.
    """
    import ee  # lazy

    linear = ee.Image(10).pow(image.select(["VV", "VH"]).divide(10))
    smoothed = linear.focalMedian(float(kernel_m), "square", "meters")
    return ee.Image(10).multiply(smoothed.log10()).rename(["VV", "VH"])


def build_s1_image(geom: Any, start: str, end: str, orbit: str,
                   s1cfg: dict[str, Any]) -> Any:
    """Build one season's edge-masked, speckle-filtered S1 features (EE init).

    Args:
        geom: ``ee.Geometry`` AOI.
        start: ISO start date (inclusive).
        end: ISO end date (exclusive).
        orbit: ``ASCENDING`` or ``DESCENDING`` (resolved from config/--orbit).
        s1cfg: ``data_cfg["sentinel1"]`` mapping.

    Returns:
        ``ee.Image`` with bands ``VV``, ``VH``, ``VVVH``, ``VH_std``.
    """
    import ee  # lazy

    from geoeco.ingest.gee_s1 import build_s1_collection, s1_seasonal_features

    collection = build_s1_collection(geom, start, end, orbit)  # type: ignore[arg-type]
    collection = collection.filter(
        ee.Filter.eq("instrumentMode", str(s1cfg.get("mode", "IW")))
    )
    for pol in s1cfg.get("polarizations", list(REQUIRED_POLARISATIONS)):
        collection = collection.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", str(pol))
        )
    if s1cfg.get("edge_noise_mask", True):
        buffer_m = float(s1cfg.get("border_buffer_m", DEFAULT_BORDER_BUFFER_M))
        collection = collection.map(lambda img: mask_edge_noise(img, buffer_m))
    spec = speckle_spec(s1cfg)
    if spec["mode"] != "none":
        kernel_m = float(spec["kernel_m"])
        collection = collection.map(lambda img: speckle_filter_image(img, kernel_m))
    return s1_seasonal_features(collection)


def build_plan(
    aoi: dict[str, Any],
    data: dict[str, Any],
    seed: int,
    orbit: str,
    *,
    year: int | None = None,
    season: str | None = None,
    drive_folder: str | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Pure-Python export plan (no EE): windows, orbit, grid, filenames.

    Raises:
        ValueError/KeyError: On invalid seasons or invalid S1 config.
    """
    s1cfg = data["sentinel1"]
    spec = speckle_spec(s1cfg)
    windows = seasonal_windows(data, year=year, season=season)
    aoi_name = str(aoi["name"])
    for w in windows:
        w["filename"] = f"{aoi_name}_s1_{orbit}_{w['year']}_{w['season']}.tif"
    destination = (
        f"gs://{bucket}" if bucket
        else f"Google Drive/{drive_folder}" if drive_folder
        else "Google Drive (root)"
    )
    return {
        "sensor": "sentinel-1",
        "status": "planned",
        "exported": False,
        "generated_by": "scripts/ee/export_s1_composites.py",
        "aoi": {
            "name": aoi_name,
            "bounds_wgs84": [float(v) for v in aoi["bounds_wgs84"]],
            "crs": str(aoi["crs"]),
            "resolution_m": float(aoi["resolution_m"]),
        },
        "seed": seed,
        "query": {
            "source": str(s1cfg["source"]),
            "mode": str(s1cfg.get("mode", "IW")),
            "polarizations": [str(p) for p in s1cfg.get("polarizations", [])],
            "orbit_pass": orbit,
            "edge_noise_mask": bool(s1cfg.get("edge_noise_mask", True)),
            "border_buffer_m": float(
                s1cfg.get("border_buffer_m", DEFAULT_BORDER_BUFFER_M)
            ),
            "angle_window_deg": [ANGLE_MIN_DEG, ANGLE_MAX_DEG],
            "angle_window_source": (
                "geoeco/ingest/gee_s1.py mask_s1_edge_noise production sketch"
            ),
            "speckle_filter": spec["mode"],
            "speckle_note": spec["note"],
            "speckle_kernel_m": spec["kernel_m"],
            "features": [str(f) for f in s1cfg.get("features", [])],
            "composite": str(s1cfg.get("composite", "median")),
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
        description="Plan/submit Sentinel-1 GRD seasonal COG exports. "
        "The orbit must come from configs sentinel1.orbit_pass or --orbit "
        f"(run availability_report.py first). --submit needs {EE_AUTH_HINT}."
    )
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--season", default=None)
    ap.add_argument("--orbit", default=None,
                    help="ASCENDING|DESCENDING (overrides config orbit_pass).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--drive-folder", default=None)
    ap.add_argument("--bucket", default=None)
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
        print(f"export s1: missing/invalid config: {exc}", file=sys.stderr)
        return 2
    if args.season is not None and args.season not in data["seasons"]:
        print(f"export s1: unknown season {args.season!r}; config has "
              f"{sorted(data['seasons'])}", file=sys.stderr)
        return 2
    try:
        orbit = resolve_orbit(data, args.orbit)
    except ValueError as exc:
        print(f"export s1: {exc}", file=sys.stderr)
        print(
            "export s1: run `python scripts/ee/availability_report.py` (EE "
            "auth required), pick the orbit, set configs sentinel1.orbit_pass, "
            "or pass --orbit ASCENDING|DESCENDING — the orbit is never guessed",
            file=sys.stderr,
        )
        return 2
    seed = resolve_seed(data, args.seed)
    try:
        plan = build_plan(
            aoi, data, seed, orbit,
            year=args.year, season=args.season,
            drive_folder=args.drive_folder, bucket=args.bucket,
        )
    except (KeyError, ValueError, TypeError) as exc:
        print(f"export s1: invalid plan: {exc}", file=sys.stderr)
        return 2

    if plan["query"]["speckle_note"]:
        print(f"export s1: NOTE: {plan['query']['speckle_note']}", file=sys.stderr)
    if not args.submit:
        print(json.dumps(plan, indent=2))
        print(
            "export s1: plan only — nothing exported (exported=false); "
            "re-run with --submit after `earthengine authenticate`",
            file=sys.stderr,
        )
        return 0

    ok, reason = ee_available()
    if not ok:
        print(
            f"export s1: {reason}: needs {EE_AUTH_HINT} before --submit; "
            "nothing exported",
            file=sys.stderr,
        )
        return 2
    try:
        initialize_ee(args.project)
    except Exception as exc:
        print(
            f"export s1: EE init failed ({exc}): needs {EE_AUTH_HINT}; "
            "nothing exported",
            file=sys.stderr,
        )
        return 2
    try:
        import ee  # lazy

        grid = utm_grid(aoi)  # strict: pyproj required to submit
        geom = ee.Geometry.Rectangle(grid["bounds_utm"], str(grid["crs"]), False)
        s1cfg = data["sentinel1"]
        submitted = []
        for w in plan["windows"]:
            image = build_s1_image(geom, w["start"], w["end"], orbit, s1cfg)
            image, note = apply_configured_scales(image, data)
            prefix = str(w["filename"])[:-len(".tif")]
            info = submit_image_export(
                image,
                description=f"{aoi['name']}-s1-{orbit[:3].lower()}-"
                            f"{w['year']}-{w['season']}",
                file_prefix=prefix,
                grid=grid,
                drive_folder=args.drive_folder,
                bucket=args.bucket,
                scaling_note=note,
            )
            submitted.append(info)
    except (RuntimeError, ImportError) as exc:
        print(f"export s1: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"export s1: EE export failed: {exc}", file=sys.stderr)
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
