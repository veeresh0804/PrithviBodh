"""Sentinel-1 GRD ingest helpers (GEE).

Source: ``COPERNICUS/S1_GRD``, IW mode, VV+VH. Orbit (ASCENDING /
DESCENDING) is a parameter chosen after the week-2 availability check
(config ``s1.orbit``); primary years 2019 (S1A+B) and 2025 (S1C).

``earthengine-api`` is imported lazily inside builder functions so the
package imports without EE credentials. Pure-Python helpers
(``*_spec``) return JSON-serialisable dicts for logging/DVC/MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from geoeco.ingest.composites import season_date_range

S1_SOURCE: str = "COPERNICUS/S1_GRD"
S1_MODE: str = "IW"
S1_POLARISATIONS: tuple[str, str] = ("VV", "VH")

Orbit = Literal["ASCENDING", "DESCENDING"]


def build_s1_collection_spec(
    start_date: str,
    end_date: str,
    orbit: Orbit = "ASCENDING",
    instrument_mode: str = S1_MODE,
) -> dict[str, Any]:
    """Return a JSON-serialisable spec of the S1 query (no EE needed).

    Args:
        start_date: ISO start date (inclusive).
        end_date: ISO end date (exclusive).
        orbit: Orbit pass; set from availability report config.
        instrument_mode: Normally ``"IW"``.

    Returns:
        Spec dict mirroring the EE filter chain.
    """
    return {
        "collection": S1_SOURCE,
        "filters": {
            "instrumentMode": instrument_mode,
            "polarisation": list(S1_POLARISATIONS),
            "orbitPass": orbit,
            "date_range": [start_date, end_date],
        },
        "features": ["VV_dB", "VH_dB", "VV_minus_VH_ratio", "VH_std_seasonal"],
    }


def build_s1_collection(aoi: Any, start_date: str, end_date: str, orbit: Orbit = "ASCENDING") -> Any:
    """Build a filtered S1 GRD ``ee.ImageCollection``.

    Args:
        aoi: ``ee.Geometry`` of the area of interest.
        start_date: ISO start date.
        end_date: ISO end date.
        orbit: Orbit pass to keep (single orbit for consistency).

    Returns:
        Filtered ``ee.ImageCollection`` with VV+VH bands in dB.

    Raises:
        ImportError: If ``earthengine-api`` is not installed.
    """
    require_ee()
    import ee  # lazy: needs `earthengine authenticate` beforehand

    spec = build_s1_collection_spec(start_date, end_date, orbit)
    flt = spec["filters"]
    col = (
        ee.ImageCollection(spec["collection"])
        .filterBounds(aoi)
        .filterDate(flt["date_range"][0], flt["date_range"][1])
        .filter(ee.Filter.eq("instrumentMode", flt["instrumentMode"]))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", flt["orbitPass"]))
    )
    return col


def mask_s1_edge_noise(image: Any, buffer_m: float = 500.0) -> Any:
    """Mask S1 scene edge noise (STUB with documented approach).

    Production approach: mask pixels where ``angle`` band is outside the
    valid IW swath range or within ``buffer_m`` of the scene border
    (``image.geometry().buffer(-buffer).difference`` pattern). Kept as a
    pass-through stub here so unit tests run without EE; the EE masking
    expression is applied by the caller notebook/pipeline step.

    Args:
        image: ``ee.Image`` (S1 GRD scene).
        buffer_m: Border buffer in metres.

    Returns:
        The input image (stub); production code chains ``updateMask``.
    """
    # STUB: production one-liner (EE-side):
    #   edge = image.select('angle').gt(31).and(image.select('angle').lt(46))
    #   return image.updateMask(edge)
    _ = buffer_m
    return image


def apply_refined_lee_fallback(image: Any, kernel_m: float = 70.0) -> Any:
    """Speckle filter: Refined Lee with focal-median fallback.

    Native GEE has no Refined Lee operator. Preferred path is an
    external SNAP/PolSAR-corrected export; the in-EE fallback used here
    is a focal-median on linear power (``10^(dB/10)`` -> ``focalMedian``
    -> back to dB), which approximates Lee smoothing while staying
    server-side and quota-safe.

    Args:
        image: ``ee.Image`` with VV/VH in dB.
        kernel_m: Square kernel size in metres (~7 px at 10 m).

    Returns:
        The input image (stub); wire ``focalMedian`` in production.
    """
    # Production EE sketch (per polarisation band, linear domain):
    #   lin = ee.Image(10).pow(img.divide(10))
    #   sm = lin.focalMedian(kernel_m, 'square', 'meters')
    #   return ee.Image(10).multiply(sm.log10()).copyProperties(img)
    _ = kernel_m
    return image


def s1_seasonal_features(collection: Any) -> Any:
    """Derive seasonal S1 features: median VV/VH (dB), ratio, std VH.

    Outputs per season: ``VV`` (median dB), ``VH`` (median dB),
    ``VVVH`` = VV − VH (dB ratio), ``VH_std`` (temporal std of VH,
    captures crop/water dynamics).

    Args:
        collection: Season-filtered, edge-masked S1 ``ee.ImageCollection``.

    Returns:
        Single ``ee.Image`` with bands [VV, VH, VVVH, VH_std].

    Raises:
        ImportError: If ``earthengine-api`` is not installed.
    """
    require_ee()
    import ee  # lazy

    vv = collection.select("VV").median().rename("VV")
    vh = collection.select("VH").median().rename("VH")
    ratio = vv.subtract(vh).rename("VVVH")
    vh_std = collection.select("VH").reduce(ee.Reducer.stdDev()).rename("VH_std")
    return vv.addBands([vh, ratio, vh_std])


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


def resolve_orbit(data_cfg: dict[str, Any], explicit: str | None) -> str:
    """Orbit from CLI flag else config; TBD placeholders fail loudly (no guessing)."""
    orbit = explicit or data_cfg.get("sentinel1", {}).get("orbit_pass", "ASCENDING")
    orbit = str(orbit)
    if orbit not in ("ASCENDING", "DESCENDING"):
        raise ValueError(
            f"orbit_pass is {orbit!r}: set configs sentinel1.orbit_pass or pass "
            "--orbit ASCENDING|DESCENDING (week-2 availability check picks one)"
        )
    return orbit


def build_s1_query_plan(
    aoi: dict[str, Any], data: dict[str, Any], seed: int,
    year: int | None, orbit: str,
) -> dict[str, Any]:
    """Collection query specs per year/season window (pure Python, no EE)."""
    s1cfg = data["sentinel1"]
    years = [int(year)] if year is not None else [int(y) for y in data["years"]]
    seasons_cfg = data["seasons"]
    windows = []
    for y in years:
        for season in seasons_cfg:
            start, end = season_date_range(y, season)
            spec = build_s1_collection_spec(start, end, orbit)  # type: ignore[arg-type]
            months = seasons_cfg[season].get("months", []) if isinstance(
                seasons_cfg[season], dict) else []
            windows.append({"year": y, "season": season, "months": list(months),
                            "start": start, "end": end, "spec": spec})
    return {
        "source": S1_SOURCE,
        "mode": s1cfg.get("mode", S1_MODE),
        "polarizations": list(s1cfg.get("polarizations", list(S1_POLARISATIONS))),
        "orbit": orbit,
        "features": list(s1cfg.get("features", [])),
        "seed": seed,
        "windows": windows,
    }


def run_live_counts(aoi: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Query real GEE S1 collection sizes (requires EE auth; network only here)."""
    require_ee()
    import ee

    ee.Initialize()
    bounds = [float(v) for v in aoi["bounds_wgs84"]]
    geom = ee.Geometry.Rectangle(bounds)
    out = []
    for w in plan["windows"]:
        col = build_s1_collection(  # type: ignore[arg-type]
            geom, w["start"], w["end"], plan["orbit"])
        out.append({"year": w["year"], "season": w["season"],
                    "size": col.size().getInfo()})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sentinel-1 GRD collection query: offline specs, live counts "
        "only with EE creds."
    )
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--year", type=int, default=None,
                    help="Single year to query (default: all config years).")
    ap.add_argument("--orbit", default=None,
                    help="Override configs sentinel1.orbit_pass.")
    ap.add_argument("--out", default=None,
                    help="Optional dir to write s1_collection_specs.json into.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write specs only, skip live EE queries.")
    args = ap.parse_args(argv)
    from geoeco.utils.config import load_yaml_config, require_keys

    try:
        aoi = load_yaml_config(args.config)
        require_keys(aoi, ["bounds_wgs84"], name="aoi config")
        data = load_yaml_config(args.data)
        require_keys(data, ["years", "seasons", "sentinel1"], name="data config")
        orbit = resolve_orbit(data, args.orbit)
    except FileNotFoundError as exc:
        print(f"ingest s1: missing input: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, TypeError) as exc:
        print(f"ingest s1: invalid config: {exc}", file=sys.stderr)
        return 2
    seed = resolve_seed(data, args.seed)
    try:
        plan = build_s1_query_plan(aoi, data, seed, args.year, orbit)
    except (KeyError, ValueError) as exc:
        print(f"ingest s1: invalid season/year spec: {exc}", file=sys.stderr)
        return 2
    dest = None
    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        dest = outdir / "s1_collection_specs.json"
        dest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ok, reason = ee_available()
    if args.dry_run or not ok:
        if dest is None:
            print(json.dumps(plan, indent=2))
        print(
            f"ingest s1: EE unavailable ({reason}): {EE_AUTH_HINT}; "
            "wrote offline collection specs only",
            file=sys.stderr,
        )
        return 2
    try:
        counts = run_live_counts(aoi, plan)
    except RuntimeError as exc:
        print(f"ingest s1: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - EE/network boundary: provider + transport errors
        print(f"ingest s1: EE query failed: {exc}", file=sys.stderr)
        return 1
    plan["live_counts"] = counts
    if dest is not None:
        dest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts,
                      "specs": str(dest) if dest else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
