"""Pre-registered seed-acceptance checker (covariate balance, NEVER accuracy).

Rule source: `configs/labels/labelling.yaml` -> `seed_acceptance` block.
Haversine/proxy-table logic is REUSED from `data/labels/audit/collect.py`
(`summarize` + `hav`) so the printed train-vs-test proxy table is exactly the
audit-A3 table (mean distance-to-centre, mean lon/lat, n). Cell geometry is
reused from `geoeco.labels.grid.build_cells`; band indexing reuses
`geoeco.labels.pipeline.fold_of_lon`. Nothing is hard-coded here: thresholds,
core point, bands, and AOI bounds all come from configs. No benchmark
(DW/WorldCover) data is read. Labels are never touched (coordinates only).

Usage:
    python -m geoeco.labels.seed_check --skeleton <skeleton.geojson>

Exit status: 0 when all gates pass; 2 when rejected (failing gate names are
printed); 1 on missing inputs / misconfiguration (fail loudly, never invent).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from geoeco.labels.pipeline import REPO, fold_of_lon, load_labelling_config
from geoeco.utils.config import require_keys

AUDIT_COLLECT = REPO / "data" / "labels" / "audit" / "collect.py"

GATE_CORE = "core_both_sides"
GATE_GAP = "mean_dist_gap"
GATE_BANDS = "lon_band_span"


def _load_audit_collect():
    """Import the audit helper for its haversine + proxy-table logic.

    Raises:
        FileNotFoundError: If the helper is absent (report, don't invent).
    """
    if not AUDIT_COLLECT.is_file():
        raise FileNotFoundError(
            f"Audit helper not found: {AUDIT_COLLECT} (refusing to re-derive "
            "haversine numbers — restore the file and re-run)")
    spec = importlib.util.spec_from_file_location("audit_collect", AUDIT_COLLECT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot import audit helper: {AUDIT_COLLECT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _acceptance_cfg(labelling: dict) -> dict:
    if "seed_acceptance" not in labelling:
        raise KeyError("labelling.yaml missing required block: seed_acceptance")
    acc = labelling["seed_acceptance"]
    require_keys(acc, ["core_lon", "core_lat", "core_radius_km",
                       "min_core_points_per_side", "max_mean_dist_gap_km",
                       "n_lon_bands", "min_bands_per_side",
                       "min_points_per_band"], name="seed_acceptance")
    return acc


def _core_cell_id(minlon: float, minlat: float, maxlon: float, maxlat: float,
                  cell_m: float, core_lon: float, core_lat: float) -> str | None:
    """Diagnostic only: id of the fine cell containing the core point.

    Reuses grid.build_cells geometry (same origin + cell size as the design).
    Imported lazily so this module never duplicates the math.
    """
    from geoeco.labels.grid import build_cells
    for c in build_cells(minlon, minlat, maxlon, maxlat, cell_m):
        if c["x0"] <= core_lon < c["x1"] or (core_lon == maxlon and c["x1"] == maxlon):
            if c["y0"] <= core_lat < c["y1"] or (core_lat == maxlat and c["y1"] == maxlat):
                # Edge-clamp: maxlon/maxlat belong to the last column/row.
                return c["id"]
    # Fallback for exact-max-edge containment (build_cells clamps x1/y1).
    for c in build_cells(minlon, minlat, maxlon, maxlat, cell_m):
        if c["x0"] <= core_lon <= c["x1"] and c["y0"] <= core_lat <= c["y1"]:
            return c["id"]
    return None


def check_seed_acceptance(skeleton_geojson: str | Path) -> tuple[bool, dict]:
    """Check one skeleton against the pre-registered acceptance rule.

    Args:
        skeleton_geojson: Path to a sampling-skeleton GeoJSON (any design:
            grid cells or B0-B4 bands). Only coordinates + split are read.

    Returns:
        (passed, table): passed is True iff ALL gates pass; table holds the
        audit-A3 proxy numbers plus per-gate details (JSON-serialisable).

    Raises:
        FileNotFoundError: Missing skeleton, helper, or config.
        KeyError: Missing seed_acceptance block/keys.
        ValueError: Empty skeleton or centre mismatch (never silently
            substituted).
    """
    ctx = load_labelling_config()
    labelling, aoi = ctx["labelling"], ctx["aoi"]
    acc = _acceptance_cfg(labelling)

    core_lon = float(acc["core_lon"])
    core_lat = float(acc["core_lat"])
    core_radius_km = float(acc["core_radius_km"])
    min_core = int(acc["min_core_points_per_side"])
    max_gap = float(acc["max_mean_dist_gap_km"])
    n_bands = int(acc["n_lon_bands"])
    min_span = int(acc["min_bands_per_side"])
    min_per_band = int(acc["min_points_per_band"])

    minlon, minlat, maxlon, maxlat = (float(v) for v in aoi["bounds_wgs84"])

    path = Path(skeleton_geojson)
    if not path.is_file():
        raise FileNotFoundError(f"Skeleton not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    if not feats:
        raise ValueError(f"Skeleton has no features: {path}")

    collect = _load_audit_collect()
    helper_centre = tuple(collect.CENTER)
    if abs(helper_centre[0] - core_lon) > 1e-9 or abs(helper_centre[1] - core_lat) > 1e-9:
        raise ValueError(
            f"Centre mismatch: seed_acceptance core ({core_lon}, {core_lat}) != "
            f"audit helper CENTER {helper_centre} — fix configs, not code")

    # Proxy table, EXACTLY as audit A3 (same function, same rounding).
    _, _, dist = collect.summarize(feats)

    by_split: dict[str, list[tuple[float, float]]] = {"test": [], "train": []}
    for f in feats:
        props, geom = f["properties"], f["geometry"]
        split = props.get("split")
        if split not in by_split:
            raise ValueError(f"Feature {props.get('id')} has bad split {split!r}")
        lon, lat = geom["coordinates"]
        by_split[split].append((float(lon), float(lat)))
    if not by_split["test"] or not by_split["train"]:
        raise ValueError("Skeleton must contain both test and train points")

    gap = round(abs(dist["test"]["mean_dist_km"] - dist["train"]["mean_dist_km"]), 2)
    gap_ok = gap <= max_gap

    # GATE 1 — core presence on both sides (haversine reused from collect).
    core_counts = {}
    for split, pts in by_split.items():
        core_counts[split] = sum(1 for lon, lat in pts
                                 if collect.hav(lon, lat) <= core_radius_km)
    core_ok = (core_counts["test"] >= min_core
               and core_counts["train"] >= min_core)

    # GATE 3 — longitude-band span (indexing reused from pipeline.fold_of_lon).
    band_w = (maxlon - minlon) / n_bands
    import numpy as np
    span: dict[str, list[int]] = {}
    for split, pts in by_split.items():
        lon_arr = np.array([lon for lon, _ in pts])
        idx = fold_of_lon(lon_arr, minlon, band_w, n_bands)
        counts: dict[int, int] = {}
        for i in idx:
            counts[int(i)] = counts.get(int(i), 0) + 1
        span[split] = sorted(b for b, n in counts.items() if n >= min_per_band)
    bands_ok = len(span["test"]) >= min_span and len(span["train"]) >= min_span

    gates = {
        GATE_CORE: {"passed": core_ok, "detail":
                    f"within {core_radius_km} km of core: "
                    f"test={core_counts['test']} train={core_counts['train']} "
                    f"(need >= {min_core} each)"},
        GATE_GAP: {"passed": gap_ok, "detail":
                   f"|{dist['test']['mean_dist_km']} - "
                   f"{dist['train']['mean_dist_km']}| = {gap} km "
                   f"(need <= {max_gap} km)"},
        GATE_BANDS: {"passed": bands_ok, "detail":
                     f"bands spanned (of {n_bands}): test={span['test']} "
                     f"({len(span['test'])}) train={span['train']} "
                     f"({len(span['train'])}) (need >= {min_span} each)"},
    }
    passed = bool(core_ok and gap_ok and bands_ok)

    table = {
        "skeleton": str(path),
        "seed": labelling["seed"],
        "note": ("seed shown for traceability only; acceptance is by covariate "
                 "balance, never by model accuracy"),
        "test": dist["test"],
        "train": dist["train"],
        "mean_dist_gap_km": gap,
        "core": {"radius_km": core_radius_km,
                 "test_within": core_counts["test"],
                 "train_within": core_counts["train"],
                 "required_per_side": min_core},
        "bands": {"n_bands": n_bands, "test_bands": span["test"],
                  "train_bands": span["train"], "required_per_side": min_span,
                  "min_points_per_band": min_per_band},
        "core_cell_id": _core_cell_id(minlon, minlat, maxlon, maxlat,
                                      float(labelling["fine_block_size_m"]),
                                      core_lon, core_lat),
        "gates": gates,
        "passed": passed,
    }
    return passed, table


def _print_report(table: dict) -> str:
    lines = []
    lines.append(f"skeleton: {table['skeleton']}")
    lines.append("")
    lines.append("| split | n | mean_dist_km | mean_lon | mean_lat |")
    lines.append("|---|---|---|---|---|")
    for split in ("test", "train"):
        r = table[split]
        lines.append(f"| {split} | {r['n']} | {r['mean_dist_km']} | "
                     f"{r['mean_lon']} | {r['mean_lat']} |")
    lines.append("")
    lines.append(f"mean_dist_gap_km: {table['mean_dist_gap_km']}")
    lines.append(f"core (within {table['core']['radius_km']} km): "
                 f"test={table['core']['test_within']} "
                 f"train={table['core']['train_within']}")
    lines.append(f"bands: test={table['bands']['test_bands']} "
                 f"train={table['bands']['train_bands']}")
    lines.append(f"core_cell_id (diagnostic): {table['core_cell_id']}")
    lines.append("")
    for name, g in table["gates"].items():
        lines.append(f"gate {name}: {'PASS' if g['passed'] else 'FAIL'} — {g['detail']}")
    lines.append("")
    failed = [n for n, g in table["gates"].items() if not g["passed"]]
    if failed:
        lines.append(f"VERDICT: REJECT (failing gates: {', '.join(failed)})")
    else:
        lines.append("VERDICT: ACCEPT (all gates pass)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check a sampling skeleton against "
                                 "the pre-registered seed-acceptance rule.")
    ap.add_argument("--skeleton", required=True,
                    help="Path to a sampling-skeleton GeoJSON")
    args = ap.parse_args(argv)
    try:
        passed, table = check_seed_acceptance(args.skeleton)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(_print_report(table))
    if not passed:
        failed = [n for n, g in table["gates"].items() if not g["passed"]]
        print(f"REJECTED — failing gates: {', '.join(failed)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
