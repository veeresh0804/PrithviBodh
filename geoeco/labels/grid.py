"""Grid-based alternative block design (AUDIT ONLY — never overwrites pipeline output).

Design: square cells (default 5 km from configs/labels/labelling.yaml
fine_block_size_m, provisional) over the Hyderabad AOI. Cells are assigned to
test/train at random (seeded) with a one-cell 8-neighbourhood buffer between
opposite sides; buffer cells get no points. Because assignment is spatial (not
class-based — classes are unknown pre-labelling and benchmarks are forbidden
for test design), both sides get geographic spread; per-class balance is still
verified post-labelling by validate.py.

Guarantee: any test cell and train cell have Chebyshev distance >= 2, so the
minimum test-train point distance is >= one cell width (see
tests/test_labelling_buffer.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geoeco.labels.pipeline import REPO, load_labelling_config

KM_PER_DEG_LAT = 111.19
KM_PER_DEG_LON_AT_HYD = 111.32 * float(np.cos(np.radians(17.38)))


def build_cells(minlon: float, minlat: float, maxlon: float, maxlat: float,
                cell_m: float) -> list[dict]:
    """Regular square grid covering the AOI bounds. Deterministic."""
    dlat = (cell_m / 1000.0) / KM_PER_DEG_LAT
    dlon = (cell_m / 1000.0) / KM_PER_DEG_LON_AT_HYD
    ncols = int(np.ceil((maxlon - minlon) / dlon))
    nrows = int(np.ceil((maxlat - minlat) / dlat))
    cells = []
    for r in range(nrows):
        for c in range(ncols):
            cells.append({"id": f"g{c:02d}_{r:02d}", "col": c, "row": r,
                          "x0": minlon + c * dlon, "y0": minlat + r * dlat,
                          "x1": min(minlon + (c + 1) * dlon, maxlon),
                          "y1": min(minlat + (r + 1) * dlat, maxlat)})
    return cells


def neighbours(cell: dict, by_pos: dict[tuple[int, int], str]) -> list[str]:
    """Ids of the 8-neighbourhood cells."""
    out = []
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc == 0 and dr == 0:
                continue
            nid = by_pos.get((cell["col"] + dc, cell["row"] + dr))
            if nid:
                out.append(nid)
    return out


def assign_cells(cells: list[dict], seed: int, test_quota: int,
                 train_quota: int) -> dict[str, str]:
    """Random greedy assignment with opposite-side buffer. Fails loudly if
    quotas cannot be met (rule 4) instead of silently relaxing the buffer."""
    rng = np.random.Generator(np.random.PCG64(seed))
    by_pos = {(c["col"], c["row"]): c["id"] for c in cells}
    by_id = {c["id"]: c for c in cells}
    side: dict[str, str] = {}
    counts = {"test": 0, "train": 0}
    # Multi-pass greedy (buffer rule never relaxed): each pass reshuffles the
    # still-unassigned cells and fills gaps left by earlier passes.
    for attempt in range(10):
        order = rng.permutation([c["id"] for c in cells if c["id"] not in side])
        progressed = False
        for cid in order:
            if counts["test"] >= test_quota and counts["train"] >= train_quota:
                break
            for cand in sorted(("test", "train"),
                               key=lambda s: counts[s] / (test_quota if s == "test" else train_quota)):
                quota = test_quota if cand == "test" else train_quota
                if counts[cand] >= quota:
                    continue
                opp = "train" if cand == "test" else "test"
                if any(side.get(n) == opp for n in neighbours(by_id[cid], by_pos)):
                    continue
                side[cid] = cand
                counts[cand] += 1
                progressed = True
                break
        if counts["test"] >= test_quota and counts["train"] >= train_quota:
            break
        if not progressed:
            break
    if counts["test"] < test_quota or counts["train"] < train_quota:
        raise ValueError(f"Cell quotas unmet {counts} vs test={test_quota} train={train_quota}; "
                         "buffer too strict for this grid — report, do not relax silently.")
    return side


def sample_grid_points(side: dict[str, str], cells: list[dict], n_test: int,
                       n_train: int, seed: int) -> list[dict]:
    """Uniform seeded sampling inside assigned cells. Labels always empty."""
    by_id = {c["id"]: c for c in cells}
    rng = np.random.Generator(np.random.PCG64(seed + 100))
    pts: list[dict] = []
    for split, n in (("test", n_test), ("train", n_train)):
        pool = [by_id[c] for c, s in side.items() if s == split]
        pick = rng.integers(0, len(pool), n)
        for k, ci in enumerate(pick):
            cell = pool[int(ci)]
            lon = rng.uniform(cell["x0"], cell["x1"])
            lat = rng.uniform(cell["y0"], cell["y1"])
            pts.append({"id": f"HYD-G-{len(pts) + 1:04d}", "block": cell["id"],
                        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                        "label": None, "label_name": "", "labeller": "", "notes": "",
                        "split": split, "fine_block": cell["id"], "overlap_id": None})
    return pts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build grid-design audit skeleton.")
    ap.add_argument("--outdir", default=str(REPO / "data" / "labels" / "audit"))
    ap.add_argument("--test-quota", type=int, default=48, help="test cells")
    ap.add_argument("--train-quota", type=int, default=36, help="train cells")
    args = ap.parse_args(argv)
    ctx = load_labelling_config()
    cfg, aoi = ctx["labelling"], ctx["aoi"]
    minlon, minlat, maxlon, maxlat = (float(v) for v in aoi["bounds_wgs84"])
    cells = build_cells(minlon, minlat, maxlon, maxlat, float(cfg["fine_block_size_m"]))
    side = assign_cells(cells, int(cfg["seed"]), args.test_quota, args.train_quota)
    pts = sample_grid_points(side, cells, int(cfg["splits"]["test"]),
                             int(cfg["splits"]["train"]), int(cfg["seed"]))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(outdir / "grid_cells.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "col", "row", "side", "x0", "y0", "x1", "y1"])
        for c in cells:
            w.writerow([c["id"], c["col"], c["row"], side.get(c["id"], "buffer"),
                        round(c["x0"], 5), round(c["y0"], 5),
                        round(c["x1"], 5), round(c["y1"], 5)])
    feats = [{"type": "Feature", "id": p["id"], "geometry": p["geometry"],
              "properties": {k: p[k] for k in
                             ["id", "block", "label", "label_name", "labeller",
                              "notes", "split", "fine_block", "overlap_id"]}} for p in pts]
    with open(outdir / "hyd_sampling_skeleton_grid.geojson", "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, indent=1)
        fh.write("\n")
    counts = {"test_cells": sum(1 for s in side.values() if s == "test"),
              "train_cells": sum(1 for s in side.values() if s == "train"),
              "buffer_cells": sum(1 for s in side.values() if s not in ("test", "train")) + len(cells) - len(side),
              "n_points": len(pts)}
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
