"""Buffer-aware leakage tests for point-based spatial designs.

Stronger than block-membership alone: asserts (a) no shared block id between
splits AND (b) every test point is at least `buffer_km` from every train
point (haversine). Parameters come from configs/labels/labelling.yaml
(fine_block_size_m) — nothing hard-coded.

These tests CAN fail: test_buffer_catches_adjacent_design builds a violating
fixture (points 1 km apart across a shared boundary), and
test_current_band_design_violates_buffer documents that the committed B0-B4
band design has test/train points closer than one cell width.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from geoeco.labels.pipeline import REPO, load_labelling_config

R_KM = 6371.0


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Pairwise haversine distances for coordinate arrays."""
    lon1r = np.radians(np.asarray(lon1, dtype=float))[:, None]
    lat1r = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lon2r = np.radians(np.asarray(lon2, dtype=float))[None, :]
    lat2r = np.radians(np.asarray(lat2, dtype=float))[None, :]
    h = (np.sin((lat2r - lat1r) / 2.0) ** 2
         + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2.0) ** 2)
    return 2.0 * R_KM * np.arcsin(np.sqrt(h))


def min_test_train_km(test_xy: list, train_xy: list) -> float:
    """Minimum test-train distance in km."""
    tx = np.array([p[0] for p in test_xy]); ty = np.array([p[1] for p in test_xy])
    rx = np.array([p[0] for p in train_xy]); ry = np.array([p[1] for p in train_xy])
    return float(haversine_km(tx, ty, rx, ry).min())


def assert_buffer_and_blocks(test_pts: list, train_pts: list, buffer_km: float) -> float:
    """Assert block disjointness + buffer distance. Returns min distance.

    Raises:
        AssertionError: On shared blocks or any pair closer than buffer_km.
    """
    tb = {p["block"] for p in test_pts}
    rb = {p["block"] for p in train_pts}
    shared = tb & rb
    assert not shared, f"Shared blocks in both splits: {sorted(shared)}"
    d = min_test_train_km([p["xy"] for p in test_pts], [p["xy"] for p in train_pts])
    assert d >= buffer_km, f"Min test-train distance {d:.2f} km < buffer {buffer_km} km"
    return d


def _load_audit_grid() -> tuple[list, list]:
    data = json.loads((REPO / "data" / "labels" / "audit"
                       / "hyd_sampling_skeleton_grid.geojson").read_text(encoding="utf-8"))
    test, train = [], []
    for f in data["features"]:
        p = {"block": f["properties"]["block"], "xy": tuple(f["geometry"]["coordinates"])}
        (test if f["properties"]["split"] == "test" else train).append(p)
    return test, train


def test_grid_design_passes_buffer():
    cfg = load_labelling_config()["labelling"]
    buf = float(cfg["fine_block_size_m"]) / 1000.0
    test, train = _load_audit_grid()
    d = assert_buffer_and_blocks(test, train, buf)
    assert d >= buf


def test_buffer_catches_adjacent_design():
    test = [{"block": "B2", "xy": (78.53, 17.38)}]
    train = [{"block": "B3", "xy": (78.54, 17.38)}]  # ~1 km away
    with pytest.raises(AssertionError, match="[Bb]uffer|distance"):
        assert_buffer_and_blocks(test, train, 5.0)


def test_shared_block_caught():
    pts = [{"block": "g01_01", "xy": (78.3, 17.3)}]
    with pytest.raises(AssertionError, match="[Ss]hared blocks"):
        assert_buffer_and_blocks(pts, pts, 5.0)


def test_current_band_design_violates_buffer():
    """Documents the audit finding: the superseded band design had test/train
    neighbours across the B2/B3 boundary closer than one cell width.
    Reads the archived copy (adoption moved canonical paths to the grid)."""
    data = json.loads((REPO / "data" / "labels" / "audit" / "bands_superseded"
                       / "hyd_sampling_skeleton.geojson").read_text(encoding="utf-8"))
    test = [{"block": f["properties"]["block"], "xy": tuple(f["geometry"]["coordinates"])}
            for f in data["features"] if f["properties"]["split"] == "test"]
    train = [{"block": f["properties"]["block"], "xy": tuple(f["geometry"]["coordinates"])}
             for f in data["features"] if f["properties"]["split"] == "train"]
    with pytest.raises(AssertionError):
        assert_buffer_and_blocks(test, train, 5.0)
