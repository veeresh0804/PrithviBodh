"""CLI coverage for features + evaluation (no network, no data/ reads, <10 s)."""

import json
from pathlib import Path

import pytest

from geoeco.evaluation import spatial_cv as CV
from geoeco.features import build as B
from geoeco.features import sampling as S

REPO = Path(__file__).resolve().parents[1]
SENTINEL_CFG = REPO / "configs" / "data" / "sentinel.yaml"
SPATIAL_CV_CFG = REPO / "configs" / "eval" / "spatial_cv.yaml"


def _help_ok(main) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_build_help() -> None:
    _help_ok(B.main)


def test_sampling_help() -> None:
    _help_ok(S.main)


def test_spatial_cv_help() -> None:
    _help_ok(CV.main)


def test_stack_spec_numbers_from_config() -> None:
    cfg = B.load_sentinel_config(str(SENTINEL_CFG))
    assert len(cfg["sentinel2"]["bands"]) == 10
    assert len(cfg["sentinel2"]["indices"]) == 4
    assert len(cfg["sentinel1"]["features"]) == 4
    assert len(cfg["dem"]["derived"]) == 2
    spec = B.build_stack_spec(cfg)
    assert spec["breakdown"] == {"optical": 14, "sar": 4, "terrain": 2}
    assert spec["channels_per_season"] == 20 == cfg["stack_per_season"]
    assert spec["n_seasons"] == 3
    assert spec["total_channels"] == 60


def test_build_refuses_without_composites(tmp_path) -> None:
    rc = B.main(
        [
            "--config", str(SENTINEL_CFG),
            "--out", str(tmp_path / "stack"),
            "--composites-dir", str(tmp_path / "empty_composites"),
        ]
    )
    assert rc == 1
    assert not (tmp_path / "stack" / B.MANIFEST_NAME).exists()


def test_sampling_cli_prints_plan(capsys) -> None:
    assert S.main(["--config", str(SPATIAL_CV_CFG)]) == 0
    plan = json.loads(capsys.readouterr().out)["cv_plan"]
    assert plan["n_splits"] == 5
    assert plan["block_size_km"] == [3.0, 10.0]
    assert plan["block_size_m"] == [3000.0, 10000.0]
    assert plan["seed"] == 42


def test_spatial_cv_smoke_passes(capsys) -> None:
    assert CV.main(["--config", str(SPATIAL_CV_CFG)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["synthetic"] is True
    assert report["seed"] == 42
    assert len(report["folds"]) == 5
    assert report["all_disjoint"] is True
    assert all(f["disjoint"] for f in report["folds"])
    assert all("SYNTHETIC_" in b for f in report["folds"] for b in f["test_blocks"])


def test_missing_config_nonzero(tmp_path) -> None:
    missing = str(tmp_path / "nope.yaml")
    assert B.main(["--config", missing]) == 1
    assert S.main(["--config", missing]) == 1
    assert CV.main(["--config", missing]) == 1
