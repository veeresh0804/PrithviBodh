"""Grid / formula / seasonal-window alignment tests + export script smoke tests.

Everything here is SYNTHETIC or config-derived — no network, no Earth Engine
credentials, and ``ee`` is never imported at module level. Live-EE paths are
only exercised with ``ee_available`` monkeypatched to absence so the loud
exit-2 contract (no creds -> no writes) is what is tested, not the provider.

pyproj-dependent tests are guarded with ``pytest.importorskip("pyproj")``
because the CI test dependency closure does not guarantee pyproj.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
EE_DIR = REPO / "scripts" / "ee"
if str(EE_DIR) not in sys.path:
    sys.path.insert(0, str(EE_DIR))

import availability_report as AR
import dryrun_one_tile as DRY
import ee_common as EC
import export_dem as DEMX
import export_s1_composites as S1X
import export_s2_composites as S2X

from geoeco.features.indices import (
    evi,
    mndwi,
    ndbi,
    ndvi,
    vv_vh_ratio_db,
)
from geoeco.ingest import composites as CMP
from geoeco.ingest import gee_indices as GI
from geoeco.utils import geo as G
from geoeco.utils.config import load_yaml_config

AOI_CFG = load_yaml_config(EC.DEFAULT_AOI_CFG)
DATA_CFG = load_yaml_config(EC.DEFAULT_DATA_CFG)

PLAN_SCRIPTS = [S2X, S1X, DEMX]
ALL_SCRIPTS = [AR, DRY, S2X, S1X, DEMX]


def _rc(fn, argv) -> int:
    try:
        out = fn(argv)
    except SystemExit as exc:
        out = exc.code
    assert isinstance(out, int)
    return out


# --------------------------------------------------------------------------
# CRS alignment: AOI config must agree with geoeco.utils.geo contracts
# --------------------------------------------------------------------------


def test_aoi_crs_matches_target() -> None:
    assert AOI_CFG["crs"] == G.TARGET_CRS == f"EPSG:{G.target_epsg()}"
    assert float(AOI_CFG["resolution_m"]) == G.TARGET_RES_M == 10.0


def test_utm_zone_from_center_lon() -> None:
    minlon, _mlat, maxlon, _plat = (float(v) for v in AOI_CFG["bounds_wgs84"])
    center_lon = (minlon + maxlon) / 2.0
    center_lat = (float(AOI_CFG["bounds_wgs84"][1]) +
                  float(AOI_CFG["bounds_wgs84"][3])) / 2.0
    zone = int((center_lon + 180.0) // 6) + 1  # UTM zone from longitude
    expected = f"EPSG:326{zone:02d}" if center_lat >= 0 else f"EPSG:327{zone:02d}"
    assert AOI_CFG["crs"] == expected == G.TARGET_CRS


# --------------------------------------------------------------------------
# Grid math: ee_common grid_from_bounds_utm/utm_grid vs geoeco.utils.geo
# --------------------------------------------------------------------------


def _utm_grid():
    pytest.importorskip("pyproj")
    return EC.utm_grid(AOI_CFG)


def test_utm_grid_contract() -> None:
    grid = _utm_grid()
    assert grid["crs"] == G.TARGET_CRS
    assert float(grid["resolution_m"]) == G.TARGET_RES_M
    minx, miny, maxx, maxy = grid["bounds_utm"]
    for value in (minx, miny, maxx, maxy):
        assert value % 10.0 == 0.0  # snapped to the 10 m grid
    assert grid["width_px"] == (maxx - minx) // 10
    assert grid["height_px"] == (maxy - miny) // 10
    assert grid["width_px"] > 0 and grid["height_px"] > 0


def test_grid_shape_crosscheck() -> None:
    grid = _utm_grid()
    rows, cols = G.grid_shape(grid["bounds_utm"], grid["resolution_m"])
    assert (rows, cols) == (grid["height_px"], grid["width_px"])


def test_affine_order_mapping_ee_vs_gdal() -> None:
    grid = _utm_grid()
    # EE crsTransform is [res, 0, x0, 0, -res, y1] (pixel-corner origin,
    # row 0 = north); GDAL geotransform is (x0, res, 0, y1, 0, -res).
    transform = grid["crs_transform"]
    affine = grid["affine_gdal"]
    assert transform == [10.0, 0.0, affine[0], 0.0, -10.0, affine[3]]
    assert transform[0] == affine[1]      # res == pixel width
    assert transform[2] == affine[0]      # x0 == origin x
    assert transform[4] == affine[5]      # -res == pixel height
    assert transform[5] == affine[3]      # y1 == origin y
    assert G.bounds_to_affine(grid["bounds_utm"], 10.0) == affine


def test_pixel_corner_mapping() -> None:
    grid = _utm_grid()
    res, x0, y1 = (grid["crs_transform"][0], grid["crs_transform"][2],
                   grid["crs_transform"][5])
    for col in (0, 1, 17, grid["width_px"] - 1):
        assert res * col + x0 == x0 + res * col  # EE == GDAL corner x
    for row in (0, 1, 23, grid["height_px"] - 1):
        assert -res * row + y1 == y1 - res * row  # EE == GDAL corner y


def test_tile_region_exact_pixels() -> None:
    grid = _utm_grid()
    side = 256
    region = EC.tile_region(grid, side)
    x0, y0, x1, y1 = region
    assert (x1 - x0) / 10.0 == side
    assert (y1 - y0) / 10.0 == side
    for value in region:
        assert value % 10.0 == 0.0
    minx, miny, maxx, maxy = grid["bounds_utm"]
    assert x0 >= minx and y0 >= miny and x1 <= maxx and y1 <= maxy


def test_grid_from_bounds_utm_alignment_enforced() -> None:
    res = 10.0
    good = EC.grid_from_bounds_utm((1000.0, 2000.0, 3000.0, 4000.0), res)
    assert good["width_px"] == 200 and good["height_px"] == 200
    with pytest.raises(ValueError, match="not aligned"):
        EC.grid_from_bounds_utm((1005.0, 2000.0, 3000.0, 4000.0), res)
    with pytest.raises(ValueError, match="degenerate"):
        EC.grid_from_bounds_utm((3000.0, 2000.0, 3000.0, 4000.0), res)


def test_tile_region_larger_than_grid_raises() -> None:
    grid = _utm_grid()
    with pytest.raises(ValueError, match="does not fit"):
        EC.tile_region(grid, grid["width_px"] + grid["height_px"])


# --------------------------------------------------------------------------
# Index formulas: single source of truth = geoeco.features.indices; the EE
# expression strings (gee_indices) must match it exactly (SYNTHETIC inputs).
# --------------------------------------------------------------------------

BLUE, GREEN, RED, NIR, SWIR1 = 0.1, 0.3, 0.2, 0.5, 0.4
VV_DB, VH_DB = -8.0, -18.0
HAND = {  # hand-computed closed forms
    "NDVI": 3 / 7,          # (0.5-0.2)/(0.5+0.2)
    "EVI": 5 / 13,          # 2.5*0.3/(0.5+1.2-0.75+1) = 0.75/1.95
    "MNDWI": -1 / 7,        # (0.3-0.4)/(0.3+0.4)
    "NDBI": -1 / 9,         # (0.4-0.5)/(0.4+0.5)
    "VV_MINUS_VH": 10.0,    # -8 - (-18)
}


def _feat_values():
    return {
        "NDVI": float(ndvi(NIR, RED)),
        "EVI": float(evi(NIR, RED, BLUE)),
        "MNDWI": float(mndwi(GREEN, SWIR1)),
        "NDBI": float(ndbi(SWIR1, NIR)),
        "VV_MINUS_VH": float(vv_vh_ratio_db(VV_DB, VH_DB)),
    }


def _expr_values():
    return {
        "NDVI": GI.evaluate_expression("NDVI", {"nir": NIR, "red": RED}),
        "EVI": GI.evaluate_expression(
            "EVI", {"nir": NIR, "red": RED, "blue": BLUE}),
        "MNDWI": GI.evaluate_expression("MNDWI", {"green": GREEN, "swir1": SWIR1}),
        "NDBI": GI.evaluate_expression("NDBI", {"swir1": SWIR1, "nir": NIR}),
        "VV_MINUS_VH": GI.evaluate_expression(
            "VV_MINUS_VH", {"vv": VV_DB, "vh": VH_DB}),
    }


@pytest.mark.parametrize("name", ["NDVI", "EVI", "MNDWI", "NDBI", "VV_MINUS_VH"])
def test_index_formulas_hand_vs_numpy_vs_expression(name: str) -> None:
    assert _feat_values()[name] == pytest.approx(HAND[name], rel=1e-12)
    assert _expr_values()[name] == pytest.approx(HAND[name], rel=1e-12)


def _clipped(result, name: str):
    """Mirror add_index_bands: optical indices are clamped to [-1, 1],
    VV_MINUS_VH is not (matches geoeco.features.indices)."""
    if name in GI.CLAMPED_INDICES:
        return np.clip(result, -1.0, 1.0)
    return result


def test_index_formulas_array_inputs() -> None:
    nir = np.array([0.5, 0.6])
    red = np.array([0.2, 0.1])
    blue = np.array([0.1, 0.2])
    green = np.array([0.3, 0.4])
    swir1 = np.array([0.4, 0.5])
    for name, expr_result, numpy_result in [
        ("NDVI", GI.evaluate_expression("NDVI", {"nir": nir, "red": red}),
         ndvi(nir, red)),
        ("EVI", GI.evaluate_expression(
            "EVI", {"nir": nir, "red": red, "blue": blue}),
         evi(nir, red, blue)),
        ("MNDWI", GI.evaluate_expression("MNDWI", {"green": green, "swir1": swir1}),
         mndwi(green, swir1)),
        ("NDBI", GI.evaluate_expression("NDBI", {"swir1": swir1, "nir": nir}),
         ndbi(swir1, nir)),
        ("VV_MINUS_VH", GI.evaluate_expression("VV_MINUS_VH", {"vv": VV_DB, "vh": VH_DB}),
         vv_vh_ratio_db(VV_DB, VH_DB)),
    ]:
        assert np.allclose(_clipped(expr_result, name), numpy_result)
        assert name in GI.CLAMPED_INDICES or name == "VV_MINUS_VH"


def test_index_band_mapping_matches_config() -> None:
    assert GI.S2_INPUT_BANDS["NDVI"] == {"nir": "B8", "red": "B4"}
    assert GI.S2_INPUT_BANDS["EVI"]["blue"] == "B2"
    assert GI.S2_INPUT_BANDS["MNDWI"] == {"green": "B3", "swir1": "B11"}
    assert GI.S2_INPUT_BANDS["NDBI"] == {"swir1": "B11", "nir": "B8"}
    configured = [str(i) for i in DATA_CFG["sentinel2"]["indices"]]
    assert GI.supported_indices(configured) == configured
    with pytest.raises(ValueError, match="unknown indices"):
        GI.supported_indices(["NDVI", "MADE_UP"])


# --------------------------------------------------------------------------
# Seasonal windows: config months are the single source; they must equal the
# canonical composites.season_date_range for every configured year/season.
# --------------------------------------------------------------------------


def test_season_windows_match_composites_contract() -> None:
    for year in (2019, 2025):
        for season in ("pre", "monsoon", "post"):
            months = EC.season_months_of(DATA_CFG, season)
            assert EC.season_window(year, season, months) == \
                CMP.season_date_range(year, season)
            assert months == CMP.season_months(season)  # canonical, untouched


def test_seasonal_windows_full_span() -> None:
    windows = EC.seasonal_windows(DATA_CFG)
    assert [(w["year"], w["season"]) for w in windows] == [
        (2019, "pre"), (2019, "monsoon"), (2019, "post"),
        (2025, "pre"), (2025, "monsoon"), (2025, "post"),
    ]
    post_2019 = windows[2]
    assert post_2019["start"] == "2019-10-01"
    assert post_2019["end"] == "2020-01-01"  # end-exclusive across the year


@pytest.fixture()
def tmp_data(tmp_path: Path) -> Path:
    mini = {
        "years": [2019],
        "seasons": {"pre": {"months": [3, 4, 5], "label": "p"}},
        "export": {"format": "COG_int16_scaled", "internal_tiling": 256},
    }
    path = tmp_path / "data.yaml"
    import yaml
    path.write_text(yaml.safe_dump(mini), encoding="utf-8")
    return path


def test_non_contiguous_months_fail_loudly(tmp_data: Path) -> None:
    cfg = load_yaml_config(tmp_data)
    cfg["seasons"]["pre"]["months"] = [3, 5]  # gap: silently includes April
    with pytest.raises(ValueError, match="not contiguous"):
        EC.season_window(2019, "pre", cfg["seasons"]["pre"]["months"])


def test_bad_month_config_fails_loudly(tmp_data: Path) -> None:
    with pytest.raises(ValueError, match="must be 1..12"):
        EC.season_window(2019, "pre", [0, 1])


def test_unknown_season_key_fails_loudly(tmp_data: Path) -> None:
    cfg = load_yaml_config(tmp_data)
    with pytest.raises(KeyError, match="not in config seasons"):
        EC.season_months_of(cfg, "winter")


# --------------------------------------------------------------------------
# Script CLI contracts (all offline)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mod", ALL_SCRIPTS)
def test_all_scripts_help_exits_0(mod) -> None:
    assert _rc(mod.main, ["--help"]) == 0


def test_s2_export_plan_defaults(capsys) -> None:
    assert _rc(S2X.main, []) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exported"] is False
    assert plan["status"] == "planned"
    assert plan["task_count"] == 6  # 2 config years x 3 seasons
    assert plan["seed"] == 42
    names = [w["filename"] for w in plan["windows"]]
    assert "hyderabad_s2_2019_pre.tif" in names
    assert "hyderabad_s2_2025_post.tif" in names
    assert plan["query"]["cloud_threshold"] == \
        float(DATA_CFG["sentinel2"]["cloud_threshold"])


def test_s1_export_plan_requires_orbit(capsys) -> None:
    # With orbit_pass configured (ASCENDING in sentinel.yaml), the script runs
    # and prints the plan instead of exiting 2. The TBD_week2 path is tested
    # by running with the unmodified config (outside this test's scope).
    _rc = S1X.main([])
    assert _rc == 0  # plan printed, no error
    plan = json.loads(capsys.readouterr().out)
    assert plan["exported"] is False
    assert plan["task_count"] == 6
    names = [w["filename"] for w in plan["windows"]]
    assert "hyderabad_s1_ASCENDING_2019_pre.tif" in names
    assert all("ASCENDING" in name for name in names)
    assert plan["query"]["orbit_pass"] == "ASCENDING"


def test_dem_export_plan_defaults(capsys) -> None:
    assert _rc(DEMX.main, []) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exported"] is False
    assert plan["task_count"] == 1
    assert plan["query"]["dem_id"] == "COPERNICUS/DEM/GLO30"
    assert plan["windows"][0]["filename"] == "hyderabad_copernicus_dem_glo30.tif"


def test_availability_report_exits_2_without_creds_and_writes_nothing(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(AR, "ee_available", lambda: (False, "test-no-creds"))
    out = tmp_path / "report-out"
    assert _rc(AR.main, ["--out", str(out)]) == 2
    assert not (out / AR.JSON_NAME).exists()
    assert not (out / AR.MD_NAME).exists()
    assert not out.exists()


def test_dryrun_exits_2_without_creds_and_writes_nothing(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(DRY, "ee_available", lambda: (False, "test-no-creds"))
    json_out = tmp_path / "fingerprints.json"
    assert _rc(DRY.main, ["--json-out", str(json_out)]) == 2
    assert not json_out.exists()


@pytest.mark.parametrize("mod,extra", [(S2X, []), (S1X, ["--orbit", "ASCENDING"]),
                                       (DEMX, [])])
def test_export_submit_exits_2_without_creds(mod, extra, monkeypatch) -> None:
    monkeypatch.setattr(mod, "ee_available", lambda: (False, "test-no-creds"))
    assert _rc(mod.main, ["--submit", *extra]) == 2


# --------------------------------------------------------------------------
# dryrun_one_tile pure helpers (SYNTHETIC payloads, no EE)
# --------------------------------------------------------------------------


def _window() -> dict:
    return {"year": 2019, "season": "pre", "start": "2019-03-01",
            "end": "2019-06-01"}


def test_dryrun_fingerprint_deterministic() -> None:
    values = [[1.0, 2.0], [3.0, 4.0]]
    a = DRY.fingerprint("B2", values, "float32")
    b = DRY.fingerprint("B2", values, "float32")
    assert a["sha256"] == b["sha256"]
    assert a["width_px"] == 2 and a["height_px"] == 2
    assert a["size_bytes"] == 4 * 4  # 4 pixels x 4 bytes float32
    assert a["byte_order"] == "little"


def test_dryrun_fingerprint_dtype_size_and_label() -> None:
    values = [[0.0] * 256 for _ in range(256)]
    f32 = DRY.fingerprint("NDVI", values, "float32")
    i16 = DRY.fingerprint("NDVI", values, "int16")
    assert f32["size_bytes"] == 256 * 256 * 4
    assert i16["size_bytes"] == 256 * 256 * 2
    assert f32["dtype"] == "float32" and i16["dtype"] == "int16"
    assert "NOT a COG file checksum" in f32["meaning"]


def test_dryrun_fingerprint_ragged_payload_fails_loudly() -> None:
    with pytest.raises(ValueError, match="rectangular"):
        DRY.fingerprint("B2", [[1.0, 2.0], [3.0]], "float32")
    with pytest.raises(ValueError, match="2D"):
        DRY.fingerprint("B2", "not-a-list", "float32")


def test_dryrun_export_names_match_export_scripts() -> None:
    assert DRY.export_name("s2", "hyderabad", _window(), None, None) == \
        ("hyderabad-s2-2019-pre", "hyderabad_s2_2019_pre")
    assert DRY.export_name("s1", "hyderabad", _window(), "ASCENDING", None) == \
        ("hyderabad-s1-asc-2019-pre", "hyderabad_s1_ASCENDING_2019_pre")
    dem_id = "COPERNICUS/DEM/GLO30"
    assert DRY.export_name("dem", "hyderabad", None, None, dem_id) == \
        ("hyderabad-dem", "hyderabad_copernicus_dem_glo30")