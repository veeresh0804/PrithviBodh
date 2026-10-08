"""Sentinel-2 SR ingest helpers (GEE).

Source: ``COPERNICUS/S2_SR_HARMONIZED`` masked with
``GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`` (default threshold 0.6,
config ``s2.cloud_thr``). Bands B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
(20 m bands resampled to 10 m). Outputs per season: median composite
+ valid-observation count (for cloud analysis / cloud-stress test).

``earthengine-api`` is imported lazily so the package imports cleanly
without EE credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from geoeco.ingest.composites import season_date_range

S2_SOURCE: str = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SOURCE: str = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
DEFAULT_CLOUD_THR: float = 0.6
S2_BANDS: tuple[str, ...] = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
CLOUD_BAND: str = "cs"


def build_s2_collection_spec(
    start_date: str, end_date: str, cloud_thr: float = DEFAULT_CLOUD_THR
) -> dict[str, Any]:
    """Return a JSON-serialisable spec of the S2 query (no EE needed).

    Args:
        start_date: ISO start date (inclusive).
        end_date: ISO end date (exclusive).
        cloud_thr: Keep pixels with Cloud Score+ ``cs >= thr`` clear.

    Returns:
        Spec dict for logging/DVC/MLflow.
    """
    if not 0.0 < cloud_thr < 1.0:
        raise ValueError(f"cloud_thr must be in (0,1), got {cloud_thr}")
    return {
        "collection": S2_SOURCE,
        "cloud_collection": CLOUD_SOURCE,
        "bands": list(S2_BANDS),
        "cloud_band": CLOUD_BAND,
        "cloud_thr": cloud_thr,
        "date_range": [start_date, end_date],
        "outputs": ["median", "valid_count"],
    }


def mask_s2_clouds(image: Any, cloud_thr: float = DEFAULT_CLOUD_THR) -> Any:
    """Mask clouds on an S2 SR image joined with Cloud Score+.

    Expects the image to already carry the ``cs`` band (via a
    ``linkCollection``/join on ``system:index`` in the caller).
    Keeps pixels with ``cs >= cloud_thr``.

    Args:
        image: ``ee.Image`` with S2 SR bands + ``cs`` band.
        cloud_thr: Clear-sky threshold (default 0.6).

    Returns:
        Cloud-masked ``ee.Image``.
    """
    return image.updateMask(image.select(CLOUD_BAND).gte(cloud_thr))


def s2_seasonal_composite(collection: Any) -> Any:
    """Median composite + valid-observation count for a season.

    Args:
        collection: Season-filtered, cloud-masked S2 ``ee.ImageCollection``.

    Returns:
        ``ee.Image`` with median bands plus ``valid_count`` band.
    """
    median = collection.select(list(S2_BANDS)).median()
    count = collection.select(S2_BANDS[0]).count().rename("valid_count")
    return median.addBands(count)


def build_s2_collection(
    aoi: Any, start_date: str, end_date: str, cloud_thr: float = DEFAULT_CLOUD_THR
) -> Any:
    """Build a cloud-masked S2 SR ``ee.ImageCollection``.

    Joins ``S2_SR_HARMONIZED`` with Cloud Score+ on ``system:index``,
    applies :func:`mask_s2_clouds`, and filters to ``S2_BANDS``.

    Args:
        aoi: ``ee.Geometry`` of the area of interest.
        start_date: ISO start date.
        end_date: ISO end date.
        cloud_thr: Clear-sky threshold.

    Returns:
        Cloud-masked ``ee.ImageCollection``.

    Raises:
        ImportError: If ``earthengine-api`` is not installed.
    """
    require_ee()
    import ee  # lazy

    s2 = (
        ee.ImageCollection(S2_SOURCE)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    )
    cs = (
        ee.ImageCollection(CLOUD_SOURCE).filterBounds(aoi).filterDate(start_date, end_date)
    )
    linked = ee.Join.saveFirst("cloud").apply(s2, cs, ee.Filter.equals("system:index", "system:index"))
    def _attach(img: Any) -> Any:
        cloud_img = ee.Image(img.get("cloud")).select(CLOUD_BAND)
        return ee.Image(img).addBands(cloud_img)

    return ee.ImageCollection(linked).map(_attach).map(lambda img: mask_s2_clouds(img, cloud_thr))


REPO = Path(__file__).resolve().parents[2]
DEFAULT_AOI_CFG = REPO / "configs" / "aoi" / "hyderabad.yaml"
DEFAULT_DATA_CFG = REPO / "configs" / "data" / "sentinel.yaml"
SPATIAL_CV_CFG = REPO / "configs" / "eval" / "spatial_cv.yaml"
EE_AUTH_HINT = "needs earthengine authenticate"


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
        from geoeco.utils.config import load_yaml_config

        cv = load_yaml_config(SPATIAL_CV_CFG)
        if isinstance(cv.get("seed"), int):
            return int(cv["seed"])
    except (FileNotFoundError, ValueError, TypeError):
        pass
    return 42


def resolve_cloud_thr(data_cfg: dict[str, Any], explicit: float | None) -> float:
    """Cloud threshold from CLI flag else config (validated by the spec builder)."""
    thr = float(explicit) if explicit is not None else float(
        data_cfg.get("sentinel2", {}).get("cloud_threshold", DEFAULT_CLOUD_THR))
    return thr


def build_s2_query_plan(
    aoi: dict[str, Any], data: dict[str, Any], seed: int,
    year: int | None, cloud_thr: float,
) -> dict[str, Any]:
    """Collection query specs per year/season window (pure Python, no EE)."""
    s2cfg = data["sentinel2"]
    years = [int(year)] if year is not None else [int(y) for y in data["years"]]
    seasons_cfg = data["seasons"]
    windows = []
    for y in years:
        for season in seasons_cfg:
            start, end = season_date_range(y, season)
            spec = build_s2_collection_spec(start, end, cloud_thr)
            months = seasons_cfg[season].get("months", []) if isinstance(
                seasons_cfg[season], dict) else []
            windows.append({"year": y, "season": season, "months": list(months),
                            "start": start, "end": end, "spec": spec})
    return {
        "source": S2_SOURCE,
        "cloud_source": CLOUD_SOURCE,
        "bands": list(s2cfg.get("bands", list(S2_BANDS))),
        "cloud_threshold": cloud_thr,
        "seed": seed,
        "windows": windows,
    }


def run_live_counts(aoi: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Query real GEE S2 collection sizes (requires EE auth; network only here)."""
    require_ee()
    import ee

    ee.Initialize()
    bounds = [float(v) for v in aoi["bounds_wgs84"]]
    geom = ee.Geometry.Rectangle(bounds)
    out = []
    for w in plan["windows"]:
        col = build_s2_collection(geom, w["start"], w["end"], plan["cloud_threshold"])
        out.append({"year": w["year"], "season": w["season"],
                    "size": col.size().getInfo()})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sentinel-2 SR collection query: offline specs, live counts "
        "only with EE creds."
    )
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--year", type=int, default=None,
                    help="Single year to query (default: all config years).")
    ap.add_argument("--cloud-thr", type=float, default=None,
                    help="Override configs sentinel2.cloud_threshold.")
    ap.add_argument("--out", default=None,
                    help="Optional dir to write s2_collection_specs.json into.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write specs only, skip live EE queries.")
    args = ap.parse_args(argv)
    from geoeco.utils.config import load_yaml_config, require_keys

    try:
        aoi = load_yaml_config(args.config)
        require_keys(aoi, ["bounds_wgs84"], name="aoi config")
        data = load_yaml_config(args.data)
        require_keys(data, ["years", "seasons", "sentinel2"], name="data config")
        cloud_thr = resolve_cloud_thr(data, args.cloud_thr)
    except FileNotFoundError as exc:
        print(f"ingest s2: missing input: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, TypeError) as exc:
        print(f"ingest s2: invalid config: {exc}", file=sys.stderr)
        return 2
    seed = resolve_seed(data, args.seed)
    try:
        plan = build_s2_query_plan(aoi, data, seed, args.year, cloud_thr)
    except (KeyError, ValueError) as exc:
        print(f"ingest s2: invalid season/year spec: {exc}", file=sys.stderr)
        return 2
    dest = None
    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        dest = outdir / "s2_collection_specs.json"
        dest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ok, reason = ee_available()
    if args.dry_run or not ok:
        if dest is None:
            print(json.dumps(plan, indent=2))
        print(
            f"ingest s2: EE unavailable ({reason}): {EE_AUTH_HINT}; "
            "wrote offline collection specs only",
            file=sys.stderr,
        )
        return 2
    try:
        counts = run_live_counts(aoi, plan)
    except RuntimeError as exc:
        print(f"ingest s2: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - EE/network boundary: provider + transport errors
        print(f"ingest s2: EE query failed: {exc}", file=sys.stderr)
        return 1
    plan["live_counts"] = counts
    if dest is not None:
        dest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts,
                      "specs": str(dest) if dest else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
