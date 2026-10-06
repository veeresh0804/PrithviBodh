"""Tests for the Hyderabad labelling pipeline (human labels stay human).

Covers: deterministic generation (fixed seed), block disjointness (same gate
as CI leakage test), overlap size/spread, duplicates only inside the overlap
set, agreement gate pass+fail on fixtures, .qgz validity (zip + XML + points
layer + no classification layer), prelabel refusals (test rows, missing
rasters), merge blocking on empty labels.
"""
import json
import xml.etree.ElementTree as ET
import zipfile

import pytest

from geoeco.features.sampling import assert_no_leakage
from geoeco.labels import pipeline as P


def _skeleton(n_test=200, n_train=150, seed=42):
    import numpy as np
    ctx = P.load_labelling_config()
    cfg = ctx["labelling"]
    # Small deterministic skeleton reusing pipeline geometry helpers.
    minlon, minlat, maxlon, maxlat = 78.20, 17.11, 78.76, 17.65
    folds = list(cfg["folds"])
    band = (maxlon - minlon) / len(folds)
    dlat = 5000 / 111_190.0
    dlon = 5000 / (111_320.0 * __import__("numpy").cos(__import__("numpy").radians(17.38)))
    rng = np.random.Generator(np.random.PCG64(seed))
    pts = P.sample_points(rng, n_test, minlon, minlon + band * 3, minlat, maxlat,
                          1, "test", folds, minlon, band, dlon, dlat, minlat)
    pts += P.sample_points(rng, n_train, minlon + band * 3, maxlon, minlat, maxlat,
                           1 + n_test, "train", folds, minlon, band, dlon, dlat, minlat)
    return pts


def test_generation_deterministic():
    a, b = _skeleton(), _skeleton()
    assert [(p["id"], p["block"]) for p in a] == [(p["id"], p["block"]) for p in b]


def test_blocks_disjoint_and_expected():
    pts = _skeleton()
    test_blocks = {p["block"] for p in pts if p["split"] == "test"}
    train_blocks = {p["block"] for p in pts if p["split"] == "train"}
    assert test_blocks == {"B0", "B1", "B2"}
    assert train_blocks == {"B3", "B4"}
    assert_no_leakage([p["block"] for p in pts if p["split"] == "train"],
                      [p["block"] for p in pts if p["split"] == "test"])


def test_labels_always_empty_from_pipeline():
    pts = _skeleton()
    assert all(p["label"] is None and p["label_name"] == "" for p in pts)


def test_overlap_size_and_spread(tmp_path):
    pts = _skeleton()
    chosen = P.select_overlap(pts, 0.10, 42)
    assert 30 <= len(chosen) <= 40  # 10% of 350
    assert {p["split"] for p in chosen} == {"test", "train"}
    assert len({p["block"] for p in chosen}) >= 4


def test_member_split_counts_and_overlap_dup(tmp_path):
    pts = _skeleton()
    P.select_overlap(pts, 0.10, 42)
    members = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    paths, counts = P.split_members(pts, members, 42, tmp_path)
    assert sum(counts.values()) == len(pts) + len([p for p in pts if p["overlap_id"]])
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert all(f["properties"]["label"] is None for f in data["features"])
        assert all("suggested_label" not in f["properties"] for f in data["features"])


def test_agreement_gate_pass_and_fail():
    from geoeco.labels.agreement import passes_gate
    ok, ag, _kp = passes_gate(["0"] * 90 + ["1"] * 10, ["0"] * 90 + ["2"] * 10, 0.85)
    assert ok and ag == pytest.approx(0.9)
    ok, ag, _ = passes_gate(["0"] * 70 + ["1"] * 30, ["0"] * 70 + ["2"] * 30, 0.85)
    assert not ok and ag == pytest.approx(0.7)


def test_qgz_valid_and_no_classification_layer(tmp_path):
    from geoeco.labels import qgis as Q
    dest = tmp_path / "member_A.qgz"
    Q.write_qgz("A", dest)
    assert dest.is_file()
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
        assert names == ["member_A.qgs"]
        xml = zf.read(names[0]).decode("utf-8")
    root = ET.fromstring(xml)
    texts = ET.tostring(root, encoding="unicode").lower()
    assert "member_a.geojson" in texts
    assert "arcgisonline" in texts and "google" in texts
    assert "value" in texts and "water (0)" in texts
    assert "classif" not in texts and "suggest" not in texts


def test_prelabel_missing_inputs_fail_loudly(tmp_path):
    from geoeco.labels import prelabel as PL
    with pytest.raises(FileNotFoundError):
        PL.run_pretrain_suggestions(tmp_path)


def test_merge_blocked_on_empty_labels(tmp_path, monkeypatch):
    from geoeco.labels import merge as M
    (tmp_path / "o.csv").write_text("overlap_id,point_id,block,split,member_a,member_b,status\n",
                                    encoding="utf-8")
    assert M.main(["--members-dir", str(tmp_path), "--overlap",
                   str(tmp_path / "o.csv"), "--adjudications",
                   str(tmp_path / "a.csv"), "--out", str(tmp_path / "o.geojson")]) == 2
