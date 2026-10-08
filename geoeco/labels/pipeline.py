"""Hyderabad labelling pipeline: sampling -> blocks -> overlap -> member split.

HARD RULES enforced here:
1. `label` is ALWAYS written empty (None). This module never fills, suggests,
   or copies any label — test points are human-only.
2. Benchmark products (Dynamic World / WorldCover) are never read here.
   Optional train-only suggestions live in prelabel.py (off by default).
3. Seed and every parameter come from configs/labels/labelling.yaml
   (seed asserted equal to configs/eval/spatial_cv.yaml). Nothing hard-coded.
4. Missing inputs fail loudly (FileNotFoundError / KeyError), never invented.

Design notes (also in docs/label_run_report.md):
- Class-stratified *placement* is impossible pre-labelling without a class map,
  and using DW/WorldCover as that map is forbidden for test design (rule 2).
  Points are uniform-random (seeded) per split region; class balance is checked
  post-labelling by validate.py with a top-up protocol.
- Folds B0-B4 are contiguous longitude bands (~11.9 km). This exceeds the
  spatial_cv 3-10 km range (provisional deviation, logged); the `fine_block`
  column (5 km grid, in-range) preserves fine granularity for analysis.
- Test folds {B0,B1,B2} and train folds {B3,B4} are disjoint by construction;
  leakage check asserts this on every run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geoeco.utils.config import load_yaml_config, require_keys

REPO = Path(__file__).resolve().parents[2]
LABELLING_CFG = REPO / "configs" / "labels" / "labelling.yaml"
SPATIAL_CV_CFG = REPO / "configs" / "eval" / "spatial_cv.yaml"
AOI_CFG = REPO / "configs" / "aoi" / "hyderabad.yaml"

# Required output columns for member files (label empty by rule 1).
COLUMNS = ["id", "block", "geometry", "label", "label_name", "labeller", "notes"]
EXTRA_COLUMNS = ["split", "fine_block", "overlap_id"]


def load_labelling_config(path: str | Path = LABELLING_CFG) -> dict:
    """Load and cross-check labelling + AOI + spatial-CV configs."""
    cfg = load_yaml_config(path)
    require_keys(cfg, ["seed", "splits", "folds", "test_folds", "train_folds",
                       "fine_block_size_m", "overlap_fraction", "members",
                       "agreement_threshold"], name="labelling.yaml")
    cv = load_yaml_config(SPATIAL_CV_CFG)
    if cfg["seed"] != cv.get("seed"):
        raise ValueError(
            f"Seed mismatch: labelling.yaml seed={cfg['seed']} vs "
            f"spatial_cv.yaml seed={cv.get('seed')} (must be equal)")
    aoi = load_yaml_config(AOI_CFG)
    require_keys(aoi, ["bounds_wgs84"], name="hyderabad.yaml")
    overlap = set(cfg["test_folds"]) & set(cfg["train_folds"])
    if overlap:
        raise ValueError(f"test_folds and train_folds share blocks: {sorted(overlap)}")
    if not set(cfg["test_folds"]) | set(cfg["train_folds"]) <= set(cfg["folds"]):
        raise ValueError("test/train folds must be subsets of folds")
    return {"labelling": cfg, "aoi": aoi, "cv": cv}


def fold_of_lon(lon: np.ndarray, minlon: float, band: float, n: int) -> np.ndarray:
    """Map longitudes to contiguous fold indices 0..n-1 (deterministic)."""
    idx = np.floor((np.asarray(lon, dtype=float) - minlon) / band).astype(int)
    return np.clip(idx, 0, n - 1)


def fine_block_of(lon: np.ndarray, lat: np.ndarray, minlon: float, minlat: float,
                  dlon: float, dlat: float) -> list[str]:
    """Fine grid cell ids at ~5 km (provisional size, in config range)."""
    fx = np.floor((np.asarray(lon, dtype=float) - minlon) / dlon).astype(int)
    fy = np.floor((np.asarray(lat, dtype=float) - minlat) / dlat).astype(int)
    return [f"fx{x}_fy{y}" for x, y in zip(fx, fy)]


def sample_points(rng: np.random.Generator, n: int, lon_lo: float, lon_hi: float,
                  lat_lo: float, lat_hi: float, start_id: int, split: str,
                  folds: list[str], minlon: float, band: float,
                  dlon: float, dlat: float, minlat: float) -> list[dict]:
    """Uniform seeded sample; fold + fine_block derived from coordinates."""
    lon = rng.uniform(lon_lo, lon_hi, n)
    lat = rng.uniform(lat_lo, lat_hi, n)
    fidx = fold_of_lon(lon, minlon, band, len(folds))
    fb = fine_block_of(lon, lat, minlon, minlat, dlon, dlat)
    pts = []
    for k in range(n):
        pts.append({
            "id": f"HYD-{start_id + k:04d}",
            "block": folds[int(fidx[k])],
            "geometry": {"type": "Point", "coordinates": [float(lon[k]), float(lat[k])]},
            "label": None,
            "label_name": "",
            "labeller": "",
            "notes": "",
            "split": split,
            "fine_block": fb[k],
            "overlap_id": None,
        })
    return pts


def build_skeleton() -> tuple[list[dict], dict]:
    """Generate the ~3,500-point skeleton. Returns (points, report_counts)."""
    ctx = load_labelling_config()
    cfg, aoi = ctx["labelling"], ctx["aoi"]
    seed = cfg["seed"]
    minlon, minlat, maxlon, maxlat = (float(v) for v in aoi["bounds_wgs84"])
    folds: list[str] = list(cfg["folds"])
    band = (maxlon - minlon) / len(folds)
    # 5 km grid in degrees at Hyderabad latitude (~17.38 N).
    dlat = cfg["fine_block_size_m"] / 111_190.0
    dlon = cfg["fine_block_size_m"] / (111_320.0 * np.cos(np.radians(17.38)))
    test_folds: list[str] = list(cfg["test_folds"])
    train_folds: list[str] = list(cfg["train_folds"])
    # Contiguous fold regions: test = west bands, train = east bands.
    test_lo = minlon + band * min(folds.index(f) for f in test_folds)
    test_hi = minlon + band * (max(folds.index(f) for f in test_folds) + 1)
    train_lo = minlon + band * min(folds.index(f) for f in train_folds)
    train_hi = minlon + band * (max(folds.index(f) for f in train_folds) + 1)

    rng = np.random.Generator(np.random.PCG64(seed))
    pts = sample_points(rng, int(cfg["splits"]["test"]), test_lo, test_hi,
                        minlat, maxlat, 1, "test", folds,
                        minlon, band, dlon, dlat, minlat)
    pts += sample_points(rng, int(cfg["splits"]["train"]), train_lo, train_hi,
                         minlat, maxlat, 1 + int(cfg["splits"]["test"]), "train",
                         folds, minlon, band, dlon, dlat, minlat)
    # Every point's band must land in its split's fold set (construction check).
    for p in pts:
        allowed = test_folds if p["split"] == "test" else train_folds
        if p["block"] not in allowed:
            raise ValueError(f"Point {p['id']} in {p['block']} outside {p['split']} folds")
    # Leakage gate on construction: no shared block between splits.
    test_blocks = {p["block"] for p in pts if p["split"] == "test"}
    train_blocks = {p["block"] for p in pts if p["split"] == "train"}
    if test_blocks & train_blocks:
        raise ValueError(f"Construction leakage: {sorted(test_blocks & train_blocks)}")
    counts = {"total": len(pts),
              "test": sum(1 for p in pts if p["split"] == "test"),
              "train": sum(1 for p in pts if p["split"] == "train"),
              "test_blocks": sorted(test_blocks), "train_blocks": sorted(train_blocks)}
    return pts, counts


def select_overlap(pts: list[dict], fraction: float, seed: int) -> list[dict]:
    """Select ~fraction of points per (split, block) for double-labelling.

    Spread across members happens at split time (round-robin); spread across
    classes is impossible pre-labelling (rule 1) — covered post-hoc by the
    balance check in validate.py.
    """
    rng = np.random.Generator(np.random.PCG64(seed + 1))
    chosen: list[dict] = []
    groups: dict[tuple[str, str], list[int]] = {}
    for i, p in enumerate(pts):
        groups.setdefault((p["split"], p["block"]), []).append(i)
    for key in sorted(groups):
        idx = groups[key]
        k = round(len(idx) * fraction)
        for j in rng.permutation(idx)[:k]:
            pts[j]["overlap_id"] = f"OV-{key[0]}-{key[1]}-{len(chosen):03d}"
            chosen.append(pts[j])
    return chosen


def write_skeleton(pts: list[dict], path: Path) -> None:
    """Write skeleton GeoJSON (labels empty). Overwrites with a note."""
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "id": p["id"],
              "geometry": p["geometry"],
              "properties": {k: p[k] for k in COLUMNS[1:] + EXTRA_COLUMNS[1:] if k != "geometry"}} 
             for p in pts]
    # Keep property set exactly: required + extras (minus geometry dup).
    for f, p in zip(feats, pts):
        f["properties"] = {c: p[c] for c in
                           ["block", "label", "label_name", "labeller", "notes",
                            "split", "fine_block", "overlap_id"]}
        f["properties"]["id"] = p["id"]
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, indent=1)
        fh.write("\n")


def split_members(pts: list[dict], members: list[dict], seed: int,
                  outdir: Path) -> tuple[list[Path], dict]:
    """One file per member (~1,200 primary each); overlap rows duplicated.

    Primary assignment is a seeded shuffle round-robin. Each overlap point is
    additionally appended to exactly one second member's file (pairs cycle
    A-B, B-C, C-A), with `labeller` set to the owning member of that row.
    """
    mids = [m["id"] for m in members]
    if len(mids) < 2:
        raise ValueError("Need >= 2 members for double-labelling")
    rng = np.random.Generator(np.random.PCG64(seed + 2))
    order = rng.permutation(len(pts))
    primary: dict[str, list[dict]] = {m: [] for m in mids}
    for k, i in enumerate(order):
        primary[mids[k % len(mids)]].append(pts[i])
    pairs = [(mids[i % len(mids)], mids[(i + 1) % len(mids)]) for i in range(len(mids))]
    overlap = [p for p in pts if p["overlap_id"]]
    for k, p in enumerate(sorted(overlap, key=lambda q: q["id"])):
        a, b = pairs[k % len(pairs)]
        # Primary row already sits with one member; add the copy to the other.
        holder = next(m for m in (a, b) if not any(r["id"] == p["id"] for r in primary[m]))
        primary[holder].append(p)
    outdir.mkdir(parents=True, exist_ok=True)
    paths, counts = [], {}
    for m in mids:
        rows = sorted(primary[m], key=lambda r: r["id"])
        for r in rows:
            r = dict(r)
        out = []
        for r in rows:
            out.append({"type": "Feature", "id": r["id"], "geometry": r["geometry"],
                        "properties": {"id": r["id"], "block": r["block"],
                                       "label": None, "label_name": "",
                                       "labeller": m, "notes": "",
                                       "split": r["split"], "fine_block": r["fine_block"],
                                       "overlap_id": r["overlap_id"]}})
        dest = outdir / f"member_{m}.geojson"
        with dest.open("w", encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection", "features": out}, fh, indent=1)
            fh.write("\n")
        paths.append(dest)
        counts[m] = len(out)
    return paths, counts


def write_overlap_index(pts: list[dict], members: list[dict], path: Path) -> int:
    """Overlap manifest: point_id, members holding it, status. For validate.py."""
    mids = [m["id"] for m in members]
    pairs = [(mids[i % len(mids)], mids[(i + 1) % len(mids)]) for i in range(len(mids))]
    overlap = sorted([p for p in pts if p["overlap_id"]], key=lambda q: q["id"])
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["overlap_id", "point_id", "block", "split", "member_a", "member_b", "status"])
        for k, p in enumerate(overlap):
            a, b = pairs[k % len(pairs)]
            w.writerow([p["overlap_id"], p["id"], p["block"], p["split"], a, b, "pending"])
    return len(overlap)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hyderabad labelling pipeline (sample/split).")
    ap.add_argument("--stage", choices=["sample", "split", "all"], default="all")
    ap.add_argument("--config", default=str(LABELLING_CFG))
    ap.add_argument("--prelabel-train", action="store_true", default=False,
                    help="OPTIONAL train-only DW/WC suggestions (off by default).")
    args = ap.parse_args(argv)
    ctx = load_labelling_config(args.config)
    cfg = ctx["labelling"]
    labels_dir = REPO / "data" / "labels"
    pts, counts = build_skeleton()
    n_over = 0
    if args.stage in ("sample", "all"):
        n_over = len(select_overlap(pts, float(cfg["overlap_fraction"]), int(cfg["seed"])))
        write_skeleton(pts, labels_dir / "hyd_sampling_skeleton.geojson")
    if args.stage in ("split", "all"):
        if not any(p["overlap_id"] for p in pts):
            n_over = len(select_overlap(pts, float(cfg["overlap_fraction"]), int(cfg["seed"])))
        members = cfg["members"]
        paths, mcounts = split_members(pts, members, int(cfg["seed"]), labels_dir / "members")
        n_over = write_overlap_index(pts, members, labels_dir / "overlap_index.csv")
        print(json.dumps({"counts": counts, "overlap_points": n_over,
                          "member_files": {str(p): mcounts[m["id"]]
                                           for p, m in zip(paths, members)}},
                         indent=2))
    if args.prelabel_train:
        from geoeco.labels.prelabel import run_pretrain_suggestions
        run_pretrain_suggestions(labels_dir / "members")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
