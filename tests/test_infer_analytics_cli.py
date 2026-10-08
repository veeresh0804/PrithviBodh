"""CLI coverage for inference + analytics (no weights/rasters/network here).

Covers: --help exits 0 for each CLI; AOI tiling math (~550 nominal tiles);
smoke demos run fast on synthetic data; missing inputs exit nonzero; GEE CLI
exits 2 without credentials and never imports ee (never fakes output).
"""
import sys
from pathlib import Path

import pytest

from geoeco.analytics import area_estimation as AREA
from geoeco.analytics import change as CH
from geoeco.analytics import degradation as DEG
from geoeco.analytics import indicators as IND
from geoeco.infer import gee_inference as GEE
from geoeco.infer import tiled_inference as TI
from geoeco.utils.config import load_yaml_config

REPO = Path(__file__).resolve().parents[1]
AOI = str(REPO / "configs" / "aoi" / "hyderabad.yaml")

CLIS = [TI, GEE, CH, IND, DEG, AREA]


@pytest.mark.parametrize("mod", CLIS, ids=["tiled", "gee", "change", "indicators",
                                           "degradation", "area"])
def test_help_exits_zero(mod):
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0


def test_tile_count_math_matches_plan():
    aoi = load_yaml_config(AOI)
    h, w = TI.aoi_pixel_dims(aoi)
    assert (h, w) == (6000, 6000)
    est = TI.estimate_tile_count(h, w, tile=256, overlap=32)
    assert 500 <= est["nominal_tiles"] <= 650  # plan ~550 (24x24 = 576)
    assert est["strided_tiles"] > est["nominal_tiles"]


def test_tiled_smoke_runs_overlap_blend():
    pytest.importorskip("torch")
    assert TI.main(["--aoi", AOI, "--smoke"]) == 0


def test_tiled_missing_model_and_stack_refuses():
    assert TI.main(["--aoi", AOI]) == 1


def test_tiled_missing_aoi_fails_loudly(tmp_path):
    assert TI.main(["--aoi", str(tmp_path / "missing.yaml")]) == 1


def test_gee_without_creds_exits_2(tmp_path, monkeypatch):
    monkeypatch.delenv("GEE_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert GEE.main([]) == 2
    assert "ee" not in sys.modules  # never imports/fakes Earth Engine


def test_gee_with_creds_prints_sequence(tmp_path, monkeypatch, capsys):
    key = tmp_path / "sa.json"
    key.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    assert GEE.main([]) == 0
    out = capsys.readouterr().out
    assert "smileRandomForest" in out and "classify" in out


def test_change_smoke(capsys):
    assert CH.main(["--from", "2019", "--to", "2025", "--smoke"]) == 0
    assert "SMOKE" in capsys.readouterr().out


def test_change_missing_inputs_refuses():
    assert CH.main(["--from", "2019", "--to", "2025"]) == 1


def test_change_bad_year_order_refuses():
    assert CH.main(["--from", "2025", "--to", "2019", "--smoke"]) == 1


def test_indicators_smoke(capsys):
    assert IND.main(["--smoke"]) == 0
    assert "SMOKE" in capsys.readouterr().out


def test_indicators_missing_inputs_refuses():
    assert IND.main([]) == 1


def test_degradation_smoke(capsys):
    assert DEG.main(["--smoke"]) == 0
    assert "SMOKE" in capsys.readouterr().out


def test_degradation_missing_inputs_refuses():
    assert DEG.main([]) == 1


def test_degradation_help_states_project_definition():
    assert "§6.6" in DEG.DEGRADATION_DEFINITION_EPILOG
    assert "tree_cover" in DEG.DEGRADATION_DEFINITION_EPILOG


def test_area_estimation_smoke(capsys):
    assert AREA.main(["--smoke"]) == 0
    out = capsys.readouterr().out
    assert "synthetic" in out and "ci_low_ha" in out


def test_area_estimation_missing_inputs_refuses():
    assert AREA.main([]) == 1
