"""Buffer-aware leakage tests for point-based spatial designs.

Data-independent: no reads under ``data/`` (gitignored/DVC). The passing
grid test generates its skeleton in-memory with the REAL design code
(geoeco.labels.grid build_cells + assign_cells + sample_grid_points,
deterministic via config seed) on small quotas so runtime stays < ~5 s.
Failing cases use tiny synthetic fixtures.

Asserts (a) no shared block id between splits AND (b) every test point is at
least ``buffer_km`` from every train point (haversine). Buffer comes from
configs/labels/labelling.yaml (fine_block_size_m) — nothing hard-coded.
"""
import numpy as np
import pytest

from geoeco.labels import grid
from geoeco.labels.pipeline import load_labelling_config

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
    tx = np.array([p[0] for p in test_xy])
    ty = np.array([p[1] for p in test_xy])
    rx = np.array([p[0] for p in train_xy])
    ry = np.array([p[1] for p in train_xy])
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


def _buffer_km_from_config() -> float:
    cfg = load_labelling_config()["labelling"]
    return float(cfg["fine_block_size_m"]) / 1000.0


def _design_seed_from_config() -> int:
    cfg = load_labelling_config()["labelling"]
    return int(cfg.get("grid_seed", cfg["seed"]))


def _generated_grid_points(tmp_path) -> tuple[list, list]:
    """Real design on small quotas; round-trips through tmp_path (no data/)."""
    import json

    ctx = load_labelling_config()
    cfg, aoi = ctx["labelling"], ctx["aoi"]
    seed = _design_seed_from_config()
    cell_m = float(cfg["fine_block_size_m"])
    minlon, minlat, maxlon, maxlat = (float(v) for v in aoi["bounds_wgs84"])
    cells = grid.build_cells(minlon, minlat, maxlon, maxlat, cell_m)
    side = grid.assign_cells(cells, seed, test_quota=8, train_quota=6)
    pts = grid.sample_grid_points(side, cells, n_test=8, n_train=6, seed=seed)
    skel = tmp_path / "skeleton.geojson"
    feats = [{"type": "Feature", "id": p["id"], "geometry": p["geometry"],
              "properties": {"block": p["block"], "split": p["split"]}} for p in pts]
    skel.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    data = json.loads(skel.read_text(encoding="utf-8"))
    test, train = [], []
    for f in data["features"]:
        p = {"block": f["properties"]["block"], "xy": tuple(f["geometry"]["coordinates"])}
        (test if f["properties"]["split"] == "test" else train).append(p)
    return test, train


def test_grid_design_passes_buffer(tmp_path):
    buf = _buffer_km_from_config()
    test, train = _generated_grid_points(tmp_path)
    d = assert_buffer_and_blocks(test, train, buf)
    assert d >= buf


def test_buffer_catches_adjacent_design():
    buf = _buffer_km_from_config()
    test = [{"block": "B2", "xy": (78.53, 17.38)}]
    train = [{"block": "B3", "xy": (78.54, 17.38)}]  # ~1 km away < buffer
    with pytest.raises(AssertionError, match=r"[Bb]uffer|distance"):
        assert_buffer_and_blocks(test, train, buf)


def test_shared_block_caught():
    buf = _buffer_km_from_config()
    pts = [{"block": "g01_01", "xy": (78.3, 17.3)}]
    with pytest.raises(AssertionError, match=r"[Ss]hared blocks"):
        assert_buffer_and_blocks(pts, pts, buf)


def test_band_style_boundary_adjacency_violates_buffer():
    """Synthetic band-style failure: test/train clusters ~0.4 km apart in
    different blocks across a boundary (no archived file read)."""
    buf = _buffer_km_from_config()
    # 0.004 deg lon at 17.38N ~= 0.42 km — well under the 5 km buffer.
    test = [{"block": "B2", "xy": (78.480, 17.38 + i * 0.001)} for i in range(3)]
    train = [{"block": "B3", "xy": (78.484, 17.38 + i * 0.001)} for i in range(3)]
    with pytest.raises(AssertionError):
        assert_buffer_and_blocks(test, train, buf)
