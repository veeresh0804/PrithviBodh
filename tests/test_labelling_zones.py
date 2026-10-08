"""Tests for zone-stratified cell assignment (core/mid/far).

All fixtures generated in tmp (never data/); thresholds/seed from
configs/labels/labelling.yaml. Fast, deterministic, no network.
"""
import pytest

from geoeco.labels import grid as G
from geoeco.labels.pipeline import REPO, load_labelling_config


def _ctx():
    ctx = load_labelling_config()
    cfg, aoi = ctx["labelling"], ctx["aoi"]
    minlon, minlat, maxlon, maxlat = (float(v) for v in aoi["bounds_wgs84"])
    cells = G.build_cells(minlon, minlat, maxlon, maxlat, float(cfg["fine_block_size_m"]))
    e1, e2 = (float(v) for v in cfg["zone_edges_km"])
    min_frac = float(cfg["min_cell_area_fraction"])
    return cfg, cells, e1, e2, min_frac


def test_zone_edges_cover_all_assignable_cells():
    _cfg, cells, e1, e2, min_frac = _ctx()
    for c in cells:
        if G.cell_area_fraction(c, 5000.0) >= min_frac:
            assert G.zone_of_cell(c, e1, e2) in ("core", "mid", "far")


def test_zoned_quotas_met_and_slivers_excluded():
    _cfg, cells, e1, e2, min_frac = _ctx()
    qpz = {"core": {"test": 4, "train": 3}, "mid": {"test": 23, "train": 17},
           "far": {"test": 21, "train": 16}}
    side = G.assign_cells_zoned(cells, 6, qpz, min_frac, (e1, e2))
    by_id = {c["id"]: c for c in cells}
    got = {}
    for cid, s in side.items():
        assert G.cell_area_fraction(by_id[cid], 5000.0) >= min_frac
        key = (G.zone_of_cell(by_id[cid], e1, e2), s)
        got[key] = got.get(key, 0) + 1
    for z, q in qpz.items():
        assert got[(z, "test")] == q["test"]
        assert got[(z, "train")] == q["train"]


def test_zoned_buffer_guarantee():
    from tests.test_labelling_buffer import assert_buffer_and_blocks
    cfg, cells, e1, e2, min_frac = _ctx()
    qpz = {"core": {"test": 4, "train": 3}, "mid": {"test": 23, "train": 17},
           "far": {"test": 21, "train": 16}}
    side = G.assign_cells_zoned(cells, 6, qpz, min_frac, (e1, e2))
    pts = G.sample_grid_points(side, cells, 200, 150, 7)
    test = [{"block": p["block"], "xy": tuple(p["geometry"]["coordinates"])}
            for p in pts if p["split"] == "test"]
    train = [{"block": p["block"], "xy": tuple(p["geometry"]["coordinates"])}
             for p in pts if p["split"] == "train"]
    assert assert_buffer_and_blocks(test, train, float(cfg["fine_block_size_m"]) / 1000.0) >= 5.0


def test_zoned_deterministic():
    _cfg, cells, e1, e2, min_frac = _ctx()
    qpz = {"core": {"test": 4, "train": 3}, "mid": {"test": 23, "train": 17},
           "far": {"test": 21, "train": 16}}
    a = G.assign_cells_zoned(cells, 6, qpz, min_frac, (e1, e2))
    b = G.assign_cells_zoned(cells, 6, qpz, min_frac, (e1, e2))
    assert a == b


def test_zoned_quota_failure_loud():
    _cfg, cells, e1, e2, min_frac = _ctx()
    with pytest.raises(ValueError, match="[Qq]uota"):
        G.assign_cells_zoned(cells, 7, {"core": {"test": 12, "train": 12},
                                        "mid": {"test": 0, "train": 0},
                                        "far": {"test": 0, "train": 0}},
                             min_frac, (e1, e2))


def test_zoned_stable_across_hash_seeds(tmp_path):
    """Regression: set-iteration order must not leak into the draw.

    Runs the assignment in two processes with different PYTHONHASHSEED and
    requires identical output (this is what caught the set-ordering bug).
    """
    import subprocess
    import sys
    helper = tmp_path / "draw.py"
    helper.write_text(
        "import json, sys; sys.path.insert(0, '.');"
        "from geoeco.labels import grid as G;"
        "from geoeco.labels.pipeline import load_labelling_config;"
        "ctx = load_labelling_config(); cfg = ctx['labelling'];"
        "aoi = ctx['aoi'];"
        "cells = G.build_cells(*[float(v) for v in aoi['bounds_wgs84']], 5000.0);"
        "qpz = {'core': {'test': 4, 'train': 3}, 'mid': {'test': 23, 'train': 17},"
        " 'far': {'test': 21, 'train': 16}};"
        "e1, e2 = (float(v) for v in cfg['zone_edges_km']);"
        "mfrac = float(cfg['min_cell_area_fraction']);"
        "side = G.assign_cells_zoned(cells, 6, qpz, mfrac, (e1, e2));"
        "print(json.dumps(side, sort_keys=True))",
        encoding="utf-8")
    outs = []
    for hs in ("0", "1"):
        import os
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, str(helper)], capture_output=True,
                           text=True, cwd=str(REPO), env=env, timeout=120, check=True)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert outs[0] == outs[1]

