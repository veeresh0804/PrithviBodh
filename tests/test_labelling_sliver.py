"""Sliver-exclusion tests for the grid block design.

Data-independent: no reads under ``data/`` (gitignored/DVC). Builds the REAL
grid via geoeco.labels.grid build_cells on the config AOI, assigns with the
config grid_seed/quotas, and asserts the maxlat-clamped northern edge sliver
row (~42 m tall vs 5 km, area fraction ~0.008) never holds points.

Asserts (a) every assigned cell passes the config min_cell_area_fraction
threshold, (b) all sliver-row (top row) cells are excluded from assignment,
(c) assign_cells still meets the 48/36 quotas (else it raises ValueError
loudly). Fast, deterministic, no network, no data files.
"""
from geoeco.labels import grid
from geoeco.labels.pipeline import load_labelling_config


def _design_inputs():
    ctx = load_labelling_config()
    cfg, aoi = ctx["labelling"], ctx["aoi"]
    for key in ("grid_seed", "grid_cells", "min_cell_area_fraction",
                "fine_block_size_m", "splits"):
        assert key in cfg, f"labelling.yaml missing required key: {key}"
    minlon, minlat, maxlon, maxlat = (float(v) for v in aoi["bounds_wgs84"])
    cell_m = float(cfg["fine_block_size_m"])
    cells = grid.build_cells(minlon, minlat, maxlon, maxlat, cell_m)
    return cfg, cells, cell_m


def test_no_assigned_cell_below_area_threshold():
    cfg, cells, cell_m = _design_inputs()
    thresh = float(cfg["min_cell_area_fraction"])
    side = grid.assign_cells(cells, int(cfg["grid_seed"]),
                             int(cfg["grid_cells"]["test"]),
                             int(cfg["grid_cells"]["train"]), thresh)
    frac = {c["id"]: grid.cell_area_fraction(c, cell_m) for c in cells}
    bad = sorted(cid for cid in side if frac[cid] < thresh)
    assert not bad, f"assigned cells below area fraction {thresh}: {bad}"
    assert len(frac) == 156  # 12 cols x 13 rows over the Hyderabad AOI


def test_sliver_row_cells_all_excluded():
    cfg, cells, cell_m = _design_inputs()
    thresh = float(cfg["min_cell_area_fraction"])
    side = grid.assign_cells(cells, int(cfg["grid_seed"]),
                             int(cfg["grid_cells"]["test"]),
                             int(cfg["grid_cells"]["train"]), thresh)
    nrows = max(c["row"] for c in cells) + 1
    sliver = [c["id"] for c in cells if c["row"] == nrows - 1]
    assert len(sliver) == 12  # full northern edge row is the ~42 m sliver
    assert all(grid.cell_area_fraction(c, cell_m) < thresh
               for c in cells if c["row"] == nrows - 1)
    leaked = sorted(cid for cid in sliver if cid in side)
    assert not leaked, f"sliver-row cells assigned points: {leaked}"


def test_quotas_still_met_with_slivers_excluded():
    cfg, cells, _ = _design_inputs()
    thresh = float(cfg["min_cell_area_fraction"])
    # Raises ValueError loudly if the buffer + sliver exclusion cannot meet
    # quotas — that loud failure (not silent relaxation) is the assertion.
    side = grid.assign_cells(cells, int(cfg["grid_seed"]),
                             int(cfg["grid_cells"]["test"]),
                             int(cfg["grid_cells"]["train"]), thresh)
    n_test = sum(1 for s in side.values() if s == "test")
    n_train = sum(1 for s in side.values() if s == "train")
    assert n_test == int(cfg["grid_cells"]["test"]), n_test
    assert n_train == int(cfg["grid_cells"]["train"]), n_train
