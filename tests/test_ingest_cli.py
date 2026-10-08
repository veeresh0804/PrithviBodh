"""CLI smoke tests for geoeco.ingest (offline: no network, no EE credentials)."""
import json
from pathlib import Path

import pytest
import yaml

from geoeco.ingest import composites as CMP
from geoeco.ingest import export as EXP
from geoeco.ingest import gee_s1 as S1
from geoeco.ingest import gee_s2 as S2
from geoeco.ingest import stac as STAC

MODULES = [EXP, CMP, S1, S2, STAC]

TINY_AOI = {
    "name": "tiny",
    "bounds_wgs84": [78.40, 17.35, 78.42, 17.37],
    "crs": "EPSG:32644",
    "resolution_m": 10,
}

TINY_DATA = {
    "years": [2019],
    "seasons": {
        "pre": {"months": [3, 4, 5], "label": "pre-monsoon"},
        "monsoon": {"months": [6, 7, 8, 9], "label": "monsoon"},
        "post": {"months": [10, 11, 12], "label": "post-monsoon"},
    },
    "sentinel2": {
        "source": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": ["B2", "B3", "B4"],
        "cloud_threshold": 0.6,
        "track_valid_obs_count": True,
    },
    "sentinel1": {
        "source": "COPERNICUS/S1_GRD",
        "mode": "IW",
        "polarizations": ["VV", "VH"],
        "orbit_pass": "ASCENDING",
        "features": ["VV_dB", "VH_dB"],
    },
    "export": {"format": "COG_int16_scaled", "internal_tiling": 256,
               "catalog": "STAC"},
}


def _rc(fn, argv) -> int:
    try:
        out = fn(argv)
    except SystemExit as exc:
        out = exc.code
    assert isinstance(out, int)
    return out


@pytest.fixture()
def tiny_cfgs(tmp_path: Path) -> dict[str, str]:
    aoi = tmp_path / "aoi.yaml"
    data = tmp_path / "data.yaml"
    aoi.write_text(yaml.safe_dump(TINY_AOI), encoding="utf-8")
    data.write_text(yaml.safe_dump(TINY_DATA), encoding="utf-8")
    return {"aoi": str(aoi), "data": str(data)}


@pytest.mark.parametrize("mod", MODULES)
def test_help_exits_0(mod) -> None:
    assert _rc(mod.main, ["--help"]) == 0


def test_missing_config_nonzero(tmp_path: Path) -> None:
    missing = str(tmp_path / "nope.yaml")
    assert _rc(EXP.main, ["--config", missing, "--data", missing,
                          "--out", str(tmp_path / "o")]) != 0
    assert _rc(CMP.main, ["--config", missing, "--data", missing]) != 0
    assert _rc(S1.main, ["--config", missing, "--data", missing]) != 0
    assert _rc(S2.main, ["--config", missing, "--data", missing]) != 0
    assert _rc(STAC.main, ["--out", str(tmp_path / "no-dir")]) != 0


def test_export_manifest_offline(tiny_cfgs, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(EXP, "ee_available", lambda: (False, "test-no-creds"))
    out = tmp_path / "composites"
    rc = _rc(EXP.main, ["--config", tiny_cfgs["aoi"], "--data", tiny_cfgs["data"],
                        "--out", str(out)])
    assert rc == 2  # creds absent: manifest + clear message, no fake outputs
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "needs-ee-auth"
    assert manifest["seed"] == 42
    assert manifest["sentinel2"]["bands"] == ["B2", "B3", "B4"]
    assert set(manifest["seasons"]) == {"pre", "monsoon", "post"}
    assert manifest["grid"]["crs"] == "EPSG:32644"
    assert manifest["grid"]["resolution_m"] == 10
    assert len(manifest["windows"]) == 3
    assert len(manifest["items"]) == 3


def test_composites_plan(tiny_cfgs, tmp_path: Path) -> None:
    out = tmp_path / "plan"
    rc = _rc(CMP.main, ["--config", tiny_cfgs["aoi"], "--data", tiny_cfgs["data"],
                        "--out", str(out), "--year", "2019"])
    assert rc == 0
    plan = json.loads((out / "valid_obs_plan.json").read_text(encoding="utf-8"))
    assert plan["seed"] == 42
    assert [w["season"] for w in plan["windows"]] == ["pre", "monsoon", "post"]
    assert plan["windows"][0]["start"] == "2019-03-01"


def test_composites_bad_season_fails(tiny_cfgs, tmp_path: Path) -> None:
    bad = dict(TINY_DATA)
    bad["seasons"] = {"pre": {"months": [1, 2], "label": "x"},
                      "monsoon": {"months": [6, 7, 8, 9], "label": "y"},
                      "post": {"months": [10, 11, 12], "label": "z"}}
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    rc = _rc(CMP.main, ["--config", tiny_cfgs["aoi"], "--data", str(bad_path)])
    assert rc != 0


def test_s1_s2_specs_offline(tiny_cfgs, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(S1, "ee_available", lambda: (False, "test-no-creds"))
    monkeypatch.setattr(S2, "ee_available", lambda: (False, "test-no-creds"))
    out = tmp_path / "specs"
    assert _rc(S1.main, ["--config", tiny_cfgs["aoi"], "--data", tiny_cfgs["data"],
                         "--year", "2019", "--out", str(out)]) == 2
    assert _rc(S2.main, ["--config", tiny_cfgs["aoi"], "--data", tiny_cfgs["data"],
                         "--year", "2019", "--out", str(out)]) == 2
    s1 = json.loads((out / "s1_collection_specs.json").read_text(encoding="utf-8"))
    s2 = json.loads((out / "s2_collection_specs.json").read_text(encoding="utf-8"))
    assert s1["windows"][0]["spec"]["collection"] == "COPERNICUS/S1_GRD"
    assert s1["orbit"] == "ASCENDING"
    assert s2["windows"][0]["spec"]["collection"] == "COPERNICUS/S2_SR_HARMONIZED"
    assert s2["bands"] == ["B2", "B3", "B4"]


def test_s1_tbd_orbit_and_s2_bad_thr_fail(tiny_cfgs) -> None:
    assert _rc(S1.main, ["--config", tiny_cfgs["aoi"], "--data", tiny_cfgs["data"],
                         "--orbit", "SIDEWAYS"]) != 0
    assert _rc(S2.main, ["--config", tiny_cfgs["aoi"], "--data", tiny_cfgs["data"],
                         "--cloud-thr", "5"]) != 0


def test_require_ee_mentions_authenticate(monkeypatch) -> None:
    monkeypatch.setattr(EXP, "ee_available", lambda: (False, "test-no-creds"))
    with pytest.raises(RuntimeError, match="needs earthengine authenticate"):
        EXP.require_ee()


def test_stac_catalog_from_out_dir(tiny_cfgs, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(EXP, "ee_available", lambda: (False, "test-no-creds"))
    out = tmp_path / "composites"
    assert _rc(EXP.main, ["--config", tiny_cfgs["aoi"], "--data", tiny_cfgs["data"],
                          "--out", str(out)]) == 2
    rc = _rc(STAC.main, ["--out", str(out), "--config", tiny_cfgs["aoi"],
                         "--data", tiny_cfgs["data"]])
    assert rc == 0
    catalog = json.loads((out / "catalog.json").read_text(encoding="utf-8"))
    items = catalog.get("items", catalog.get("features", []))
    if not items:
        # pystac static layout: items live in their own files, not inline.
        skip = {"catalog.json", "collection.json", "manifest.json"}
        items = [p for p in out.rglob("*.json") if p.name not in skip]
    assert len(items) == 3
