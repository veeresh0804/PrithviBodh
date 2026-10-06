"""Post-labelling validation for merged member files.

Checks (run via `python -m geoeco.labels.validate`):
1. Overlap agreement: Cohen's kappa + raw agreement on the overlap set
   (threshold from configs/labels/labelling.yaml, default 0.85). Every
   disagreement is listed with both labels; below threshold prints the
   adjudication queue for the guide and exits non-zero.
2. Leakage: no block ID in both train and test (same gate as CI).
3. Schema: labels in 0-5, label_name matches label, no empty labels in
   merged input, no duplicate ids outside the overlap set.
4. Class balance report per member and overall (flags classes with < 5%
   share for a top-up round).

Reads data/labels/members/member_*.geojson + data/labels/overlap_index.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from geoeco.features.sampling import assert_no_leakage
from geoeco.labels.agreement import cohen_kappa, passes_gate, percent_agreement
from geoeco.labels.pipeline import REPO, load_labelling_config
from geoeco.labels.protocol import CLASS_IDS, CLASSES


def load_member_rows(members_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(members_dir.glob("member_*.geojson")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for feat in data["features"]:
            p = dict(feat["properties"])
            p["_file"] = f.name
            p["_geometry"] = feat["geometry"]
            rows.append(p)
    return rows


def check_schema(rows: list[dict], overlap_ids: set[str]) -> list[str]:
    """Validate merged rows. Returns error list (empty = pass)."""
    import re
    try:
        folds = set(load_labelling_config()["labelling"]["folds"])
    except Exception:
        folds = {"B0", "B1", "B2", "B3", "B4"}
    cell_re = re.compile(r"^g\d{2}_\d{2}$")
    errors: list[str] = []
    seen: dict[str, int] = {}
    for r in rows:
        pid = r.get("id")
        if not pid:
            errors.append(f"Row missing id in {r.get('_file')}")
            continue
        seen[pid] = seen.get(pid, 0) + 1
        lab, name = r.get("label"), r.get("label_name")
        if lab is None or lab == "":
            errors.append(f"{pid}: empty label")
        elif int(lab) not in range(6):
            errors.append(f"{pid}: label {lab} not in 0-5")
        elif name != CLASSES[int(lab)]:
            errors.append(f"{pid}: label_name {name!r} != {CLASSES[int(lab)]!r}")
        blk = r.get("block")
        if blk not in folds and not (isinstance(blk, str) and cell_re.match(blk)):
            errors.append(f"{pid}: bad block {blk!r}")
    dups = {k: v for k, v in seen.items() if v > 2}
    for k, v in dups.items():
        errors.append(f"{k}: appears {v} times (max 2, overlap only)")
    twice = {k for k, v in seen.items() if v == 2}
    illegal = twice - overlap_ids
    for k in sorted(illegal):
        errors.append(f"{k}: duplicated outside overlap set")
    return errors


def overlap_agreement(rows: list[dict], overlap_ids: set[str]) -> dict:
    """Pairwise agreement over double-labelled points (labelled pairs only)."""
    by_id: dict[str, list] = {}
    for r in rows:
        if r["id"] in overlap_ids:
            by_id.setdefault(r["id"], []).append(r)
    pairs = [(v[0], v[1]) for v in by_id.values()
             if len(v) == 2 and v[0].get("label") not in (None, "")
             and v[1].get("label") not in (None, "")]
    pending = sum(1 for v in by_id.values() if len(v) == 2) - len(pairs)
    if not pairs:
        return {"n": 0, "pending": pending, "agreement": None, "kappa": None,
                "disagreements": []}
    y1 = [str(a["label"]) for a, _ in pairs]
    y2 = [str(b["label"]) for _, b in pairs]
    disag = [{"point_id": a["id"], "block": a.get("block"),
              "member_a": f"{a.get('labeller')}:{a.get('label')}",
              "member_b": f"{b.get('labeller')}:{b.get('label')}"}
             for (a, b) in pairs for a, b in [(a, b)] if str(a["label"]) != str(b["label"])]
    return {"n": len(pairs), "agreement": percent_agreement(y1, y2),
            "kappa": cohen_kappa(y1, y2), "disagreements": disag}


def balance_report(rows: list[dict]) -> dict:
    """Class shares overall + per member. Flags < 5% classes for top-up."""
    rep: dict = {"overall": {c: 0 for c in CLASSES}, "per_member": {}, "top_up": []}
    for r in rows:
        if r.get("label") in (None, ""):
            continue
        name = CLASSES[int(r["label"])]
        rep["overall"][name] += 1
        rep["per_member"].setdefault(r.get("labeller", "?"), {}).setdefault(name, 0)
        rep["per_member"][r.get("labeller", "?")][name] += 1
    total = sum(rep["overall"].values()) or 1
    rep["top_up"] = [c for c, k in rep["overall"].items() if k / total < 0.05]
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate merged labelling.")
    ap.add_argument("--members-dir", default=str(REPO / "data" / "labels" / "members"))
    ap.add_argument("--overlap", default=str(REPO / "data" / "labels" / "overlap_index.csv"))
    args = ap.parse_args(argv)
    cfg = load_labelling_config()["labelling"]
    thr = float(cfg["agreement_threshold"])
    rows = load_member_rows(Path(args.members_dir))
    overlap_ids: set[str] = set()
    with open(args.overlap, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            overlap_ids.add(row["point_id"])
    errors = check_schema(rows, overlap_ids)
    train_blocks = [r["block"] for r in rows if r.get("split") == "train"]
    test_blocks = [r["block"] for r in rows if r.get("split") == "test"]
    try:
        assert_no_leakage(train_blocks, test_blocks)
        leak_ok = True
    except ValueError as e:
        errors.append(str(e))
        leak_ok = False
    oa = overlap_agreement(rows, overlap_ids)
    # Gate on the real pairwise vectors.
    from collections import defaultdict
    by_id: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["id"] in overlap_ids:
            by_id[r["id"]].append(r)
    vec_a = [str(v[0]["label"]) for v in by_id.values()
             if len(v) == 2 and v[0].get("label") not in (None, "")
             and v[1].get("label") not in (None, "")]
    vec_b = [str(v[1]["label"]) for v in by_id.values()
             if len(v) == 2 and v[0].get("label") not in (None, "")
             and v[1].get("label") not in (None, "")]
    gate_ok, ag, kp = (passes_gate(vec_a, vec_b, thr) if vec_a else (False, None, None))
    print(json.dumps({"schema_errors": errors, "leakage_ok": leak_ok,
                      "overlap_manifest": len(overlap_ids),
                      "overlap_labelled_pairs": oa["n"],
                      "overlap_pending": oa.get("pending", 0),
                      "agreement": ag, "kappa": kp,
                      "gate_passed": gate_ok and not errors,
                      "balance": balance_report(rows)}, indent=2))
    if oa["disagreements"]:
        print("--- disagreements (adjudication queue) ---")
        for d in oa["disagreements"]:
            print(d)
    if errors or not gate_ok:
        print(f"VALIDATION FAILED (gate>={thr}). Adjudication queue above -> guide.")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
