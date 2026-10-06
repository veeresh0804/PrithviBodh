"""Tests for the pre-registered seed-acceptance checker.

Fixtures are tiny synthetic skeletons (no network, no data files). Thresholds
are NOT hard-coded here — the checker reads them from
configs/labels/labelling.yaml; fixtures use large margins (gap 0.0 vs 37.8 km
against a 5.0 km gate; core 2 vs 0 points against a >= 1 gate) so outcomes are
robust, not threshold-fitted.
"""
import json

from geoeco.labels import seed_check as SC

MINLON = 78.20
BAND_W = 0.56 / 5  # same 5 equal bands as the acceptance rule (B0-B4 width)
MIDS = [MINLON + (b + 0.5) * BAND_W for b in range(5)]


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


def _balanced_rows():
    """Both sides share the same balanced layout: all 5 bands + core."""
    rows = []
    for split in ("test", "train"):
        for mid in MIDS:
            rows.append((split, mid, 17.38))
            rows.append((split, mid, 17.39))
    return rows


def test_balanced_fixture_passes(tmp_path):
    p = _write_skeleton(tmp_path / "balanced.geojson", _balanced_rows())
    passed, table = SC.check_seed_acceptance(p)
    assert passed is True
    assert table["passed"] is True
    # Proxy-table shape matches audit A3 columns.
    for split in ("test", "train"):
        for key in ("n", "mean_dist_km", "mean_lon", "mean_lat"):
            assert key in table[split]
    assert all(g["passed"] for g in table["gates"].values())
    assert SC.main(["--skeleton", str(p)]) == 0


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
