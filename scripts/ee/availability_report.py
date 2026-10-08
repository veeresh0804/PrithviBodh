#!/usr/bin/env python
"""Live Earth Engine availability report for the Hyderabad study (Stage 1 / M1).

Queries REAL scene statistics per config year/season window and writes both a
machine-readable JSON and a human-readable markdown report:

  docs/ee_availability/availability_report.json   (default --out)
  docs/ee_availability/availability_report.md

Per window:
  * Sentinel-2: scene count after the documented scene-cloud filter, cloudy
    pixel percentage statistics, Cloud Score+ join parity (how many scenes
    actually have a matching CS+ scene on ``system:index``).
  * Sentinel-1: scene count split by orbit pass, per-platform counts
    (S1A/S1B/S1C...), distinct relative orbits — all IW + VV/VH + config
    polarisation filters, both passes (no orbit guess).

Then a single orbit recommendation for ``configs/data/sentinel.yaml
sentinel1.orbit_pass`` with the criterion stated in the output.

Rules (repo standing orders):
  * Auth FIRST: without Earth Engine credentials this exits 2 and writes NO
    files (``python -m ee.cli.eecli authenticate`` first).
  * No fabricated counts: every number is queried from Earth Engine; a query
    failure aborts the whole report (exit 1) instead of writing partial or
    invented numbers.
  * All windows/months/AOI come from configs — nothing hard-coded.

Exit codes: 0 = report written, 2 = missing creds/config/auth,
1 = live EE query failure.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ee_common first: it puts the repo root on sys.path for the geoeco imports.
from ee_common import (
    DEFAULT_AOI_CFG,
    DEFAULT_DATA_CFG,
    EE_AUTH_HINT,
    ee_available,
    initialize_ee,
    load_configs,
    seasonal_windows,
)

DEFAULT_OUT_DIR = "docs/ee_availability"
JSON_NAME = "availability_report.json"
MD_NAME = "availability_report.md"

# Scene-cloud pre-filter used by the production S2 collection builder; it is
# NOT in configs (read-only geoeco/ingest/gee_s2.py uses the literal 80) —
# mirrored here and reported so the report and the export match scene-for-scene.
SCENE_CLOUD_FILTER_PCT = 80

# Cited from PROJECT_CONTEXT.md §2.6 as context only (never presented as query
# results): primary years 2019 (S1A+B) and 2025 (S1C; S1D from 2026-04-17).
CONTEXT_NOTES = [
    "PROJECT_CONTEXT.md §2.6: primary years are 2019 (Sentinel-1 A+B) and "
    "2025 (Sentinel-1 C; S1D enters service 2026-04-17) — platform counts "
    "above are queried live and should be read against that context.",
    "The S2 scene counts apply the same scene-level CLOUDY_PIXEL_PERCENTAGE "
    f"< {SCENE_CLOUD_FILTER_PCT} pre-filter as the export pipeline "
    "(geoeco/ingest/gee_s2.py), so counts are comparable with exports.",
    "Pixel-level masking in the exports uses Cloud Score+ "
    "(cs >= sentinel2.cloud_threshold); the parity column shows how many "
    "scenes have a CS+ counterpart at all.",
]


def _stats(values: list[Any]) -> dict[str, Any]:
    """Summary statistics for a queried numeric property list (pure).

    Args:
        values: Raw list from ``aggregate_array`` (may contain None/NaN).

    Returns:
        ``{"count": n, "min": .., "max": .., "mean": .., "median": ..}``
        with ``count: 0`` and None stats when no finite value exists.
    """
    nums = [
        float(v) for v in values
        if v is not None and not isinstance(v, bool) and math.isfinite(float(v))
    ]
    if not nums:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(nums),
        "min": min(nums),
        "max": max(nums),
        "mean": sum(nums) / len(nums),
        "median": float(statistics.median(nums)),
    }


def query_s2_window(geom: Any, window: dict[str, Any],
                    s2cfg: dict[str, Any]) -> dict[str, Any]:
    """Live EE query: S2 scene counts + cloudy stats + Cloud Score+ parity.

    Args:
        geom: ``ee.Geometry`` AOI.
        window: One window dict from :func:`seasonal_windows`.
        s2cfg: ``data_cfg["sentinel2"]`` mapping (source ids from config).

    Returns:
        JSON-serialisable stats for this window (numbers are all queried).
    """
    import ee  # lazy

    source = str(s2cfg["source"])
    cloud_source = str(s2cfg["cloud_mask_source"])
    scenes = (
        ee.ImageCollection(source)
        .filterBounds(geom)
        .filterDate(window["start"], window["end"])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", SCENE_CLOUD_FILTER_PCT))
    )
    cloudy = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    scene_count = int(scenes.size().getInfo())

    cs = (
        ee.ImageCollection(cloud_source)
        .filterBounds(geom)
        .filterDate(window["start"], window["end"])
    )
    # Same system:index join as geoeco/ingest/gee_s2.py::build_s2_collection,
    # but counted instead of mapped, so parity is measurable.
    linked = ee.Join.saveFirst("cloud").apply(
        scenes, cs, ee.Filter.equals("system:index", "system:index")
    )
    matched = int(
        ee.ImageCollection(linked)
        .filter(ee.Filter.notNull(["cloud"]))
        .size()
        .getInfo()
    )
    return {
        "source": source,
        "cloud_mask_source": cloud_source,
        "scene_count": scene_count,
        "scene_cloud_filter_pct": SCENE_CLOUD_FILTER_PCT,
        "cloudy_pixel_percentage": _stats(cloudy),
        "scenes_with_cloud_score_plus": matched,
        "scenes_without_cloud_score_plus": scene_count - matched,
        "cloud_threshold": float(s2cfg.get("cloud_threshold", 0.6)),
    }


def query_s1_window(geom: Any, window: dict[str, Any],
                    s1cfg: dict[str, Any]) -> dict[str, Any]:
    """Live EE query: S1 counts by pass/platform + distinct relative orbits.

    Orbit-agnostic on purpose: the availability report is what picks the
    orbit, so no ``orbitProperties_pass`` filter is applied here. Mode and
    polarisations come from config, matching the export filters.

    Args:
        geom: ``ee.Geometry`` AOI.
        window: One window dict from :func:`seasonal_windows`.
        s1cfg: ``data_cfg["sentinel1"]`` mapping.

    Returns:
        JSON-serialisable counts for this window (all queried, none guessed).
    """
    import ee  # lazy

    col = (
        ee.ImageCollection(str(s1cfg["source"]))
        .filterBounds(geom)
        .filterDate(window["start"], window["end"])
        .filter(ee.Filter.eq("instrumentMode", str(s1cfg.get("mode", "IW"))))
    )
    for pol in s1cfg.get("polarizations", []):
        col = col.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", str(pol))
        )
    passes = [p for p in col.aggregate_array("orbitProperties_pass").getInfo()
              if p is not None]
    platforms_raw = [p for p in col.aggregate_array("platform_number").getInfo()
                     if p is not None]
    orbits_raw = [o for o in col.aggregate_array("relativeOrbitNumber_start").getInfo()
                  if o is not None]
    return {
        "source": str(s1cfg["source"]),
        "mode": str(s1cfg.get("mode", "IW")),
        "polarizations": [str(p) for p in s1cfg.get("polarizations", [])],
        "scene_count": len(passes),
        "by_orbit_pass": dict(sorted(Counter(passes).items())),
        "by_platform": dict(sorted(
            Counter(f"S1{str(p)}" for p in platforms_raw).items()
        )),
        "relative_orbits": sorted({int(o) for o in orbits_raw}),
        "note": (
            "counts are orbit-agnostic (both passes) — the orbit filter is "
            "deliberately absent here because this report picks the orbit"
        ),
    }


def recommend_orbit(s1_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the orbit pass with the most scenes across the queried windows.

    Args:
        s1_stats: One ``query_s1_window`` result per window.

    Returns:
        ``{"orbit": "ASCENDING"|"DESCENDING"|None, "totals": {...},
        "criterion": str, "reason": str}``. A tie yields ``orbit: None`` —
        the report never silently breaks a tie for the user.
    """
    totals: Counter = Counter()
    for stat in s1_stats:
        totals.update(stat.get("by_orbit_pass", {}))
    criterion = (
        "orbit pass with the largest total scene count across all report "
        "windows (config years x seasons); a tie yields no recommendation "
        "and needs a human decision"
    )
    asc, desc = int(totals.get("ASCENDING", 0)), int(totals.get("DESCENDING", 0))
    if asc == 0 and desc == 0:
        return {"orbit": None, "totals": dict(totals), "criterion": criterion,
                "reason": "no Sentinel-1 scenes were found in any window"}
    if asc == desc:
        return {"orbit": None, "totals": dict(totals), "criterion": criterion,
                "reason": f"tie ({asc} vs {desc} scenes) — no recommendation "
                          "is invented; pick manually"}
    orbit = "ASCENDING" if asc > desc else "DESCENDING"
    return {"orbit": orbit, "totals": dict(totals), "criterion": criterion,
            "reason": f"{orbit.lower()} has more scenes ({max(asc, desc)} vs "
                      f"{min(asc, desc)})"}


def build_report(aoi: dict[str, Any], data: dict[str, Any],
                 aoi_path: str, data_path: str) -> dict[str, Any]:
    """Run every live query and assemble the full report (network).

    Nothing is written here — the caller writes files only after this
    returns, so a mid-query failure can never leave a partial report.

    Raises:
        Exception: Any EE/network failure propagates (caller maps to exit 1).
    """
    import ee  # lazy

    geom = ee.Geometry.Rectangle([float(v) for v in aoi["bounds_wgs84"]])
    windows_out = []
    for w in seasonal_windows(data):
        print(
            f"availability: querying {w['year']}/{w['season']} "
            f"({w['start']}..{w['end']}) ...",
            file=sys.stderr,
        )
        windows_out.append({
            "year": w["year"],
            "season": w["season"],
            "months": w["months"],
            "start": w["start"],
            "end": w["end"],
            "sentinel2": query_s2_window(geom, w, data["sentinel2"]),
            "sentinel1": query_s1_window(geom, w, data["sentinel1"]),
        })

    s2_by_year: dict[str, int] = {}
    s1_by_year: dict[str, int] = {}
    platforms_by_year: dict[str, dict[str, int]] = {}
    for w in windows_out:
        year = str(w["year"])
        s2_by_year[year] = s2_by_year.get(year, 0) + int(
            w["sentinel2"]["scene_count"])
        s1_by_year[year] = s1_by_year.get(year, 0) + int(
            w["sentinel1"]["scene_count"])
        bucket = platforms_by_year.setdefault(year, {})
        for plat, count in w["sentinel1"]["by_platform"].items():
            bucket[plat] = bucket.get(plat, 0) + int(count)

    s1_stats = [w["sentinel1"] for w in windows_out]
    return {
        "kind": "ee-availability-report",
        "generated_by": "scripts/ee/availability_report.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": (
            "Every count below was queried live from Earth Engine for this "
            "AOI/time window; nothing is estimated, interpolated or "
            "fabricated. A query failure aborts the report instead of "
            "writing partial numbers."
        ),
        "aoi": {
            "name": str(aoi["name"]),
            "bounds_wgs84": [float(v) for v in aoi["bounds_wgs84"]],
            "crs": str(aoi["crs"]),
            "resolution_m": float(aoi["resolution_m"]),
        },
        "years": [int(y) for y in data["years"]],
        "seasons": {
            str(name): {
                "months": [int(m) for m in spec.get("months", [])],
                "label": spec.get("label"),
            }
            for name, spec in data["seasons"].items()
        },
        "config_sources": {"aoi": str(aoi_path), "data": str(data_path)},
        "windows": windows_out,
        "aggregates": {
            "sentinel2_by_year": s2_by_year,
            "sentinel1_by_year": s1_by_year,
            "platform_counts_by_year": platforms_by_year,
        },
        "orbit_recommendation": recommend_orbit(s1_stats),
        "context_notes": list(CONTEXT_NOTES),
        "next_step": (
            "Set configs/data/sentinel.yaml sentinel1.orbit_pass to the "
            "recommended orbit (or pass --orbit to the export scripts), then "
            "run the export_s*.py --submit commands in scripts/ee/README.md."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report dict as markdown (pure; no EE, no I/O)."""
    aoi = report["aoi"]
    rec = report["orbit_recommendation"]
    lines: list[str] = [
        f"# EE Availability Report — {aoi['name']}",
        "",
        f"- **AOI:** {aoi['name']} — bounds {aoi['bounds_wgs84']} (WGS84)",
        f"- **CRS / resolution:** {aoi['crs']} @ {aoi['resolution_m']} m",
        f"- **Years:** {report['years']}",
        "- **Seasons:** " + ", ".join(
            f"{name} (months {spec['months']})"
            for name, spec in report["seasons"].items()
        ),
        f"- **Generated:** {report['generated_at']} by {report['generated_by']}",
        f"- **Provenance:** {report['provenance']}",
        "",
        "## Orbit recommendation",
        "",
        f"- **Recommended orbit:** **{rec['orbit'] or 'NONE'}**",
        f"- **Totals by pass:** {rec['totals']}",
        f"- **Criterion:** {rec['criterion']}",
        f"- **Reason:** {rec['reason']}",
        "",
        "## Scene counts per window",
        "",
        "| year | season | S2 scenes | S2 cloudy median % | S2 with CS+ | "
        "S1 scenes | S1 ASC | S1 DESC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for w in report["windows"]:
        s2, s1 = w["sentinel2"], w["sentinel1"]
        cloudy_median = s2["cloudy_pixel_percentage"]["median"]
        lines.append(
            f"| {w['year']} | {w['season']} | {s2['scene_count']} | "
            f"{'n/a' if cloudy_median is None else round(cloudy_median, 2)} | "
            f"{s2['scenes_with_cloud_score_plus']} | {s1['scene_count']} | "
            f"{s1['by_orbit_pass'].get('ASCENDING', 0)} | "
            f"{s1['by_orbit_pass'].get('DESCENDING', 0)} |"
        )
    lines += ["", "## Aggregates by year", ""]
    agg = report["aggregates"]
    for year in report["years"]:
        key = str(year)
        platforms = ", ".join(
            f"{p}: {c}"
            for p, c in sorted(agg["platform_counts_by_year"].get(key, {}).items())
        ) or "none"
        lines.append(
            f"- **{year}** — S2 scenes: {agg['sentinel2_by_year'].get(key, 0)}, "
            f"S1 scenes: {agg['sentinel1_by_year'].get(key, 0)}; "
            f"platforms: {platforms}"
        )
    rel_orbits: set[int] = set()
    for w in report["windows"]:
        rel_orbits.update(w["sentinel1"]["relative_orbits"])
    lines += [
        "",
        f"- **Distinct S1 relative orbits (all windows):** "
        f"{sorted(rel_orbits) or 'none'}",
        "",
        "## Per-window detail",
        "",
    ]
    for w in report["windows"]:
        s1 = w["sentinel1"]
        lines.append(
            f"- **{w['year']}/{w['season']}** ({w['start']} → {w['end']}): "
            f"S1 platforms {s1['by_platform'] or '{}'}, relative orbits "
            f"{s1['relative_orbits'] or '[]'}; S2 scene-cloud filter "
            f"CLOUDY_PIXEL_PERCENTAGE < {s2_filter_pct(report)}"
        )
    lines += ["", "## Context notes", ""]
    lines += [f"- {note}" for note in report["context_notes"]]
    lines += [
        "",
        "## Config sources",
        "",
        f"- AOI: `{report['config_sources']['aoi']}`",
        f"- Data: `{report['config_sources']['data']}`",
        "",
        f"**Next step:** {report['next_step']}",
        "",
    ]
    return "\n".join(lines)


def s2_filter_pct(report: dict[str, Any]) -> int:
    """Scene-cloud filter actually used (first window's queried value)."""
    for w in report["windows"]:
        return int(w["sentinel2"]["scene_cloud_filter_pct"])
    return SCENE_CLOUD_FILTER_PCT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Query live Earth Engine scene availability per config "
        "window and write JSON + markdown reports. Without credentials: "
        f"exit 2, nothing written ({EE_AUTH_HINT})."
    )
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--out", default=DEFAULT_OUT_DIR,
                    help=f"output directory (default: {DEFAULT_OUT_DIR}; "
                         "written only after every query succeeds)")
    ap.add_argument("--project", default=None, help="GCP project for ee.Initialize.")
    args = ap.parse_args(argv)

    # --- auth FIRST: no config reads, no file writes, exit 2 without creds ---
    ok, reason = ee_available()
    if not ok:
        print(
            f"availability report: {reason}: needs {EE_AUTH_HINT}; "
            "no files written",
            file=sys.stderr,
        )
        return 2

    try:
        aoi, data = load_configs(args.config, args.data)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"availability report: missing/invalid config: {exc}",
              file=sys.stderr)
        return 2
    try:
        initialize_ee(args.project)
    except Exception as exc:  # noqa: BLE001 - auth failures vary by install
        print(
            f"availability report: EE init failed ({exc}): needs "
            f"{EE_AUTH_HINT}; no files written",
            file=sys.stderr,
        )
        return 2

    try:
        report = build_report(aoi, data, args.config, args.data)
        markdown = render_markdown(report)
    except (RuntimeError, ImportError) as exc:  # creds/ee missing mid-run
        print(f"availability report: {exc}; no files written", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - live EE failure
        print(f"availability report: EE query failed: {exc}; no files written",
              file=sys.stderr)
        return 1

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / JSON_NAME
    md_path = outdir / MD_NAME
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    rec = report["orbit_recommendation"]
    print(f"availability report: wrote {json_path}", file=sys.stderr)
    print(f"availability report: wrote {md_path}", file=sys.stderr)
    print(
        f"availability report: recommended orbit = {rec['orbit'] or 'NONE'} "
        f"({rec['reason']})",
        file=sys.stderr,
    )
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "orbit_recommendation": rec,
        "windows_queried": len(report["windows"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
