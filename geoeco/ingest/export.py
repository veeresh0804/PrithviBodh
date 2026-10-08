"""COG export helpers: int16 scaled output, 256 tiling, STAC hook.

Default export: Cloud-Optimised GeoTIFF, int16 with scale/offset,
256x256 internal tiling, DEFLATE. STAC registration itself lives in
:mod:`geoeco.ingest.stac`; :func:`stac_register_stub` records the
intent/metadata so pipelines can call export without a STAC backend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from geoeco.ingest.composites import season_date_range
from geoeco.ingest.stac import item_dict
from geoeco.utils.config import load_yaml_config, require_keys

REPO = Path(__file__).resolve().parents[2]
DEFAULT_AOI_CFG = REPO / "configs" / "aoi" / "hyderabad.yaml"
DEFAULT_DATA_CFG = REPO / "configs" / "data" / "sentinel.yaml"
DEFAULT_OUT = REPO / "data" / "raw" / "composites"
SPATIAL_CV_CFG = REPO / "configs" / "eval" / "spatial_cv.yaml"
EE_AUTH_HINT = "needs earthengine authenticate"

COG_BLOCKSIZE: int = 256
COG_COMPRESS: str = "DEFLATE"
COG_DTYPE: str = "int16"


def scale_to_int16(
    arr: np.ndarray, scale: float = 0.0001, nodata: int = -9999
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale a float reflectance-like array to int16 for COG export.

    Uses ``stored = round(value / scale)``; NaN maps to ``nodata``.

    Args:
        arr: Float array (e.g. reflectance 0-1, indices -1..1).
        scale: Scale factor recorded in metadata.
        nodata: Nodata sentinel (must fit int16).

    Returns:
        (int16_array, meta) with scale/nodata description.
    """
    data = np.asarray(arr, dtype=np.float64)
    out = np.full(data.shape, nodata, dtype=np.int16)
    valid = ~np.isnan(data)
    out[valid] = np.clip(np.rint(data[valid] / scale), -32768, 32767).astype(np.int16)
    return out, {"dtype": "int16", "scale": scale, "nodata": nodata}


def cog_profile(
    width: int, height: int, count: int, crs: str = "EPSG:32644", transform: Any = None
) -> dict[str, Any]:
    """Build a rasterio-style COG profile dict (no I/O).

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        count: Band count.
        crs: Target CRS string.
        transform: Affine transform (passed through).

    Returns:
        Profile dict ready for ``rasterio.open(..., **profile)``.
    """
    return {
        "driver": "COG",
        "dtype": COG_DTYPE,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "blocksize": COG_BLOCKSIZE,
        "compress": COG_COMPRESS,
        "BIGTIFF": "IF_SAFER",
    }


def write_cog(
    path: str, arr: np.ndarray, crs: str = "EPSG:32644", transform: Any = None
) -> dict[str, Any]:
    """Write an int16 array as COG (lazy rasterio import).

    Args:
        path: Destination ``.tif`` path.
        arr: Array shaped (bands, rows, cols) or (rows, cols).
        crs: CRS string.
        transform: Affine transform.

    Returns:
        The profile used for writing.

    Raises:
        ImportError: If ``rasterio`` is not installed.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("write_cog needs rasterio: pip install rasterio") from exc
    data = np.asarray(arr)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    _, rows, cols = data.shape
    profile = cog_profile(cols, rows, data.shape[0], crs, transform)
    # rasterio has no "COG" driver string for open(); use GTiff with
    # COMPRESS + TILED + COPY_SRC_OVERVIEWS workflow instead.
    profile["driver"] = "GTiff"
    profile.update({"tiled": True, "compress": COG_COMPRESS.lower()})
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return profile


def stac_register_stub(cog_path: str, item_id: str) -> dict[str, Any]:
    """Record STAC-registration intent (STUB).

    Real registration is :func:`geoeco.ingest.stac.create_item`; this
    stub returns the minimal record the export stage logs to DVC/MLflow
    so offline runs stay reproducible without a catalog backend.

    Args:
        cog_path: Path/URL of the exported COG.
        item_id: Intended STAC item id.

    Returns:
        Stub record dict.
    """
    return {"item_id": item_id, "asset_href": cog_path, "status": "pending-stac-write"}


def ee_available() -> tuple[bool, str]:
    """Check EE credentials without any network use (credential-file check only)."""
    try:
        from ee import oauth

        # Pure path expansion (no I/O, no provider call): a missing
        # earthengine-api is the only failure mode, so no broad except.
        cred_path = oauth.get_credentials_path()
    except ImportError:
        return False, "earthengine-api not installed"
    if cred_path and os.path.isfile(cred_path):
        return True, "EE credentials present"
    return False, "no EE credentials found"


def require_ee() -> None:
    """Raise loudly when an EE call is attempted without credentials (no fake outputs)."""
    ok, reason = ee_available()
    if not ok:
        raise RuntimeError(
            f"EE call requires credentials ({reason}): "
            f"{EE_AUTH_HINT} before any network use"
        )


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


def load_ingest_configs(aoi_path: str | Path, data_path: str | Path) -> tuple[dict, dict]:
    """Load + validate AOI and data configs (fail loudly on missing/invalid inputs)."""
    aoi = load_yaml_config(aoi_path)
    require_keys(aoi, ["bounds_wgs84", "crs", "resolution_m"], name="aoi config")
    data = load_yaml_config(data_path)
    require_keys(
        data,
        ["years", "seasons", "sentinel2", "sentinel1", "export"],
        name="data config",
    )
    return aoi, data


def build_export_manifest(
    aoi: dict[str, Any], data: dict[str, Any], seed: int, ee_ok: bool
) -> dict[str, Any]:
    """Describe what WOULD be exported (bands, seasons, grid); no network, no I/O."""
    bounds = [float(v) for v in aoi["bounds_wgs84"]]
    if len(bounds) != 4:
        raise ValueError(f"bounds_wgs84 must have 4 values, got {bounds}")
    years = [int(y) for y in data["years"]]
    seasons_cfg = data["seasons"]
    windows: list[dict[str, Any]] = []
    for year in years:
        for season, spec in seasons_cfg.items():
            start, end = season_date_range(year, season)
            months = list(spec.get("months", [])) if isinstance(spec, dict) else []
            windows.append(
                {"year": year, "season": season, "months": months,
                 "start": start, "end": end}
            )
    aoi_name = str(aoi.get("name", "aoi"))
    items = [
        item_dict(
            item_id=f"{aoi_name}-{w['year']}-{w['season']}",
            cog_href=f"{w['year']}/{w['season']}/composite.tif",
            bounds_wgs84=(bounds[0], bounds[1], bounds[2], bounds[3]),
            start=w["start"],
            end=w["end"],
        )
        for w in windows
    ]
    return {
        "aoi": {
            "name": aoi_name,
            "bounds_wgs84": bounds,
            "crs": aoi["crs"],
            "resolution_m": aoi["resolution_m"],
        },
        "grid": {
            "crs": aoi["crs"],
            "resolution_m": aoi["resolution_m"],
            "bounds_wgs84": bounds,
        },
        "years": years,
        "seasons": {
            name: {
                "months": list(spec.get("months", [])) if isinstance(spec, dict) else [],
                "label": spec.get("label", name) if isinstance(spec, dict) else name,
            }
            for name, spec in seasons_cfg.items()
        },
        "windows": windows,
        "sentinel2": {
            "bands": list(data["sentinel2"].get("bands", [])),
            "cloud_threshold": data["sentinel2"].get("cloud_threshold"),
        },
        "sentinel1": {
            "polarizations": list(data["sentinel1"].get("polarizations", [])),
            "orbit_pass": data["sentinel1"].get("orbit_pass"),
            "features": list(data["sentinel1"].get("features", [])),
        },
        "export": dict(data["export"]),
        "seed": seed,
        "ee_available": ee_ok,
        "status": "planned",
        "items": items,
    }


def run_live_availability(
    aoi: dict[str, Any], data: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Query real GEE collection sizes (requires EE auth; network only here)."""
    require_ee()
    import ee

    from geoeco.ingest.gee_s1 import build_s1_collection
    from geoeco.ingest.gee_s2 import build_s2_collection

    ee.Initialize()
    bounds = [float(v) for v in aoi["bounds_wgs84"]]
    geom = ee.Geometry.Rectangle(bounds)
    cloud_thr = float(data["sentinel2"].get("cloud_threshold", 0.6))
    orbit = str(data["sentinel1"].get("orbit_pass", "ASCENDING"))
    results = []
    for w in manifest["windows"]:
        s2 = build_s2_collection(geom, w["start"], w["end"], cloud_thr)
        s1 = build_s1_collection(geom, w["start"], w["end"], orbit)  # type: ignore[arg-type]
        results.append(
            {
                "year": w["year"],
                "season": w["season"],
                "s2_size": s2.size().getInfo(),
                "s1_size": s1.size().getInfo(),
            }
        )
    manifest["availability"] = results
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="STAC-backed COG export planner: validates configs, writes a "
        "manifest of what WOULD be exported; full GEE export only with EE creds."
    )
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write the manifest only, skip live EE queries.")
    args = ap.parse_args(argv)
    try:
        aoi, data = load_ingest_configs(args.config, args.data)
    except FileNotFoundError as exc:
        print(f"ingest export: missing input: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, TypeError) as exc:
        print(f"ingest export: invalid config: {exc}", file=sys.stderr)
        return 2
    seed = resolve_seed(data, args.seed)
    try:
        manifest = build_export_manifest(aoi, data, seed, False)
    except (KeyError, ValueError) as exc:
        print(f"ingest export: invalid season/year spec: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok, reason = ee_available()
    manifest["ee_available"] = ok
    manifest_path = out / "manifest.json"
    if args.dry_run or not ok:
        manifest["status"] = "needs-ee-auth"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(
            f"ingest export: EE unavailable ({reason}): {EE_AUTH_HINT}; "
            f"wrote manifest of what WOULD be exported to {manifest_path}",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = run_live_availability(aoi, data, manifest)
    except RuntimeError as exc:
        print(f"ingest export: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - EE/network boundary: provider + transport errors
        print(f"ingest export: EE export failed: {exc}", file=sys.stderr)
        return 1
    manifest["status"] = "export-ready"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path),
                      "windows": len(manifest["windows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
