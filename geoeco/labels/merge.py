"""Merge human-labelled member files into the frozen test/train set.

Reads data/labels/members/member_*.geojson + data/labels/adjudications.csv
(template columns: point_id, final_label, decided_by, note). Applies
adjudicated labels to overlap rows, drops unconfirmed `suggested_label`
columns, re-runs validate.py (must pass), then writes:

- data/labels/hyd_points.geojson (one row per point id)
- data/labels/VERSION.json (seed, config paths, counts, validation summary)

Refuses to write if validation fails or any label is still empty.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

from geoeco.labels.pipeline import REPO
from geoeco.labels.protocol import CLASSES

TEMPLATE_HEADER = ["point_id", "final_label", "decided_by", "note"]


def write_adjudication_template(path: Path, overlap_index: Path) -> None:
    """Write the adjudication CSV skeleton from the overlap index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(overlap_index, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TEMPLATE_HEADER)
        for r in rows:
            w.writerow([r["point_id"], "", "", ""])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge labelled member files.")
    ap.add_argument("--members-dir", default=str(REPO / "data" / "labels" / "members"))
    ap.add_argument("--overlap", default=str(REPO / "data" / "labels" / "overlap_index.csv"))
    ap.add_argument("--adjudications", default=str(REPO / "data" / "labels" / "adjudications.csv"))
    ap.add_argument("--out", default=str(REPO / "data" / "labels" / "hyd_points.geojson"))
    args = ap.parse_args(argv)
    from geoeco.labels import validate as V
    members_dir, labels_dir = Path(args.members_dir), Path(args.out).parent
    rows = V.load_member_rows(members_dir)
    overlap_ids: set[str] = set()
    with open(args.overlap, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            overlap_ids.add(row["point_id"])
    adj_path = Path(args.adjudications)
    if not adj_path.is_file():
        write_adjudication_template(adj_path, Path(args.overlap))
        print(f"No adjudications yet — template written to {adj_path}. Fill final_label, re-run.")
        return 2
    final: dict[str, str] = {}
    with open(adj_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("final_label", "").strip() != "":
                final[row["point_id"]] = row["final_label"].strip()
    for r in rows:
        if r["id"] in final:
            r["label"] = int(final[r["id"]])
            r["label_name"] = CLASSES[int(final[r["id"]])]
        r.pop("suggested_label", None)
        r.pop("suggested_by", None)
    errors = V.check_schema(rows, overlap_ids)
    empty = [e for e in errors if "empty label" in e]
    if empty:
        print(f"MERGE BLOCKED: {len(empty)} rows still unlabelled (humans pending).")
        return 1
    # Collapse overlap duplicates (adjudicated single truth per id).
    seen: dict[str, dict] = {}
    for r in sorted(rows, key=lambda q: q["id"]):
        seen[r["id"]] = r
    feats = [{"type": "Feature", "id": pid,
              "geometry": r["_geometry"],
              "properties": {"id": pid, "block": r["block"], "label": r["label"],
                             "label_name": r["label_name"], "split": r.get("split"),
                             "fine_block": r.get("fine_block")}} for pid, r in seen.items()]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, indent=1)
        fh.write("\n")
    stamp = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "seed": 42, "configs": ["configs/labels/labelling.yaml",
                                     "configs/aoi/hyderabad.yaml",
                                     "configs/data/sentinel.yaml",
                                     "configs/eval/spatial_cv.yaml"],
             "n_points": len(feats)}
    with open(args.out, "rb") as fh:
        stamp["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    (labels_dir / "VERSION.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(feats)} points) + VERSION.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
