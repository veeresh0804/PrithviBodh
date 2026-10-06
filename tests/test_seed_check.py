"""Tests for the seed-acceptance checker (GATE 1 is a PROPORTION).

Fixtures are tiny synthetic skeletons (no network, no data files). Thresholds
are NOT hard-coded here — the required core counts are derived from
configs/labels/labelling.yaml (`seed_acceptance.min_core_share_per_side` x
`splits:` totals, via the checker's own config loader), so fixtures track the
rule instead of fitting it. Core points sit exactly on the core
(78.48, 17.38, dist 0 km); outer band points reuse the MIDS layout at
~11.9-23.8 km from the core (outside the 10 km radius) so band span holds
while core counts are controlled exactly.
"""
import json
import math

from geoeco.labels import seed_check as SC
from geoeco.labels.pipeline import load_labelling_config

MINLON = 78.20
BAND_W = 0.56 / 5  # same 5 equal bands as the acceptance rule (B0-B4 width)
MIDS = [MINLON + (b + 0.5) * BAND_W for b in range(5)]

CORE_LON, CORE_LAT = 78.48, 17.38
OUTER_MIDS = (MIDS[0], MIDS[1], MIDS[3], MIDS[4])  # every band but the core band


def _write_skeleton(path, rows):
    """rows: list of (split, lon, lat). Blocks are synthetic (checker only
    reads split + coordinates, so this works for any design)."""
    feats = []
    for i, (split, lon, lat) in enumerate(rows, 1):
        pid = f"SYN-{i:03d}"
        feats.append({"type": "Feature", "id": pid,
                      "geometry": {"type": "Point", "coordinates": [lon, lat]},
                      "properties": {"id": pid, "block": "s00", "label": None,
                                     "label_name": "", "labeller": "",
                                     "notes": "", "split": split,
                                     "fine_block": "s00",
                                     "overlap_id": None}})
    path.write_text(json.dumps({"type": "FeatureCollection",
                                "features": feats}), encoding="utf-8")
    return path


def _required_counts():
    """(share, n_test_cfg, n_train_cfg, req_test, req_train) from config."""
    ctx = load_labelling_config()
    labelling = ctx["labelling"]
    share = float(labelling["seed_acceptance"]["min_core_share_per_side"])
    n_test = int(labelling["splits"]["test"])
    n_train = int(labelling["splits"]["train"])
    return share, n_test, n_train, math.ceil(share * n_test), math.ceil(share * n_train)


def _core_share_rows(n_test_core, n_train_core):
    """Exact core counts per side + identical outer-band points (8/side, all
    outside the core radius) so band span holds and the mean-dist gap stays
    small — only the core gate varies across fixtures."""
    rows = []
    for _ in range(n_test_core):
        rows.append(("test", CORE_LON, CORE_LAT))
    for _ in range(n_train_core):
        rows.append(("train", CORE_LON, CORE_LAT))
    for split in ("test", "train"):
        for mid in OUTER_MIDS:
            rows.append((split, mid, 17.38))
            rows.append((split, mid, 17.39))
    return rows


def _balanced_rows():
    """Both sides share the same balanced layout: all 5 bands + core."""
    rows = []
    for split in ("test", "train"):
        for mid in MIDS:
            rows.append((split, mid, 17.38))
            rows.append((split, mid, 17.39))
    return rows


def test_core_share_pass(tmp_path):
    """Both sides at exactly the required 5% share → all gates pass."""
    _, _, _, req_test, req_train = _required_counts()
    p = _write_skeleton(tmp_path / "share_pass.geojson",
                        _core_share_rows(req_test, req_train))
    passed, table = SC.check_seed_acceptance(p)
    assert table["core"]["test_within"] == req_test
    assert table["core"]["train_within"] == req_train
    assert table["core"]["required_test"] == req_test
    assert table["core"]["required_train"] == req_train
    assert table["gates"]["core_both_sides"]["passed"] is True
    assert passed is True
    assert table["passed"] is True
    # Proxy-table shape matches audit A3 columns.
    for split in ("test", "train"):
        for key in ("n", "mean_dist_km", "mean_lon", "mean_lat"):
            assert key in table[split]
    assert all(g["passed"] for g in table["gates"].values())
    assert SC.main(["--skeleton", str(p)]) == 0


def test_core_share_fail_train_side(tmp_path):
    """Train one point below its required share (test exact) → core gate
    alone rejects; gap and band span still pass (isolation)."""
    _, _, _, req_test, req_train = _required_counts()
    p = _write_skeleton(tmp_path / "share_fail_train.geojson",
                        _core_share_rows(req_test, req_train - 1))
    passed, table = SC.check_seed_acceptance(p)
    assert passed is False
    assert table["core"]["test_within"] == req_test
    assert table["core"]["train_within"] == req_train - 1
    assert table["gates"]["core_both_sides"]["passed"] is False
    # Isolation: the other gates still pass on this fixture.
    assert table["gates"]["mean_dist_gap"]["passed"] is True
    assert table["gates"]["lon_band_span"]["passed"] is True


def test_core_share_fail_test_side(tmp_path):
    """Mirror: test one point below its required share → core gate rejects;
    gap and band span still pass (isolation)."""
    _, _, _, req_test, req_train = _required_counts()
    p = _write_skeleton(tmp_path / "share_fail_test.geojson",
                        _core_share_rows(req_test - 1, req_train))
    passed, table = SC.check_seed_acceptance(p)
    assert passed is False
    assert table["core"]["test_within"] == req_test - 1
    assert table["core"]["train_within"] == req_train
    assert table["gates"]["core_both_sides"]["passed"] is False
    # Isolation: the other gates still pass on this fixture.
    assert table["gates"]["mean_dist_gap"]["passed"] is True
    assert table["gates"]["lon_band_span"]["passed"] is True


def test_core_share_percentage_not_absolute(tmp_path):
    """2 core points per side passed the OLD absolute floor (>= 1) but must
    FAIL the proportional gate — the percentage, not the count, is enforced."""
    _, _, _, req_test, req_train = _required_counts()
    assert req_test > 2 and req_train > 2  # guard: fixture is below the share
    p = _write_skeleton(tmp_path / "share_old_floor.geojson",
                        _core_share_rows(2, 2))
    passed, table = SC.check_seed_acceptance(p)
    assert passed is False
    assert table["core"]["test_within"] == 2
    assert table["core"]["train_within"] == 2
    assert table["gates"]["core_both_sides"]["passed"] is False


def test_offset_fixture_fails_on_gap(tmp_path, capsys):
    rows = []
    for k in range(10):
        rows.append(("test", 78.48 + 0.001 * k, 17.38 + 0.001 * k))
    for k in range(10):
        rows.append(("train", 78.22 + 0.001 * k, 17.13 + 0.001 * k))
    p = _write_skeleton(tmp_path / "offset.geojson", rows)
    passed, table = SC.check_seed_acceptance(p)
    assert passed is False
    assert table["gates"]["mean_dist_gap"]["passed"] is False
    rc = SC.main(["--skeleton", str(p)])
    assert rc != 0
    out = capsys.readouterr().out
    assert "mean_dist_gap" in out  # failing gate is named


def test_core_missing_from_train_side_fails(tmp_path):
    """Train avoids the core while gap (3.08 km) and bands (4) still pass —
    the core gate alone rejects the draw."""
    rows = []
    for split, mid, lat in _balanced_rows():
        if split == "train" and abs(mid - 78.48) < 1e-9:
            mid = 78.63  # push train core points outside the 10 km radius
        rows.append((split, mid, lat))
    p = _write_skeleton(tmp_path / "nocore.geojson", rows)
    passed, table = SC.check_seed_acceptance(p)
    assert passed is False
    assert table["core"]["train_within"] == 0
    assert table["core"]["test_within"] >= 1
    assert table["gates"]["core_both_sides"]["passed"] is False
    # Isolation: the other gates still pass on this fixture.
    assert table["gates"]["mean_dist_gap"]["passed"] is True
    assert table["gates"]["lon_band_span"]["passed"] is True
