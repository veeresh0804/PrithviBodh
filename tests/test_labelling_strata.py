"""Tests for own-layer sampling strata + top-up protocol.

Synthetic numpy stacks only (the post-monsoon composite does not exist yet:
data/raw/composites/ is missing). Covers: strata mapping + priority,
threshold overrides (config-driven proof), counts, top-up needs/counts/
determinism/short-pool failure, missing-composite FileNotFoundError, seed +
composite-path config wiring, and the hard-rule gate that strata code never
writes `label` and never touches benchmarks.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from geoeco.labels import strata as S
from geoeco.labels.pipeline import load_labelling_config

STRATA_SRC = Path(__file__).resolve().parents[1] / "geoeco" / "labels" / "strata.py"


def _stack():
    # 4 pixels, one per stratum: water / veg / bright / other.
    ndvi = np.array([0.05, 0.60, 0.10, 0.30])
    mndwi = np.array([0.45, -0.20, -0.30, -0.25])
    bright = np.array([0.10, 0.20, 0.45, 0.15])
    return ndvi, mndwi, bright


def test_assign_strata_maps_each_rule():
    out = S.assign_strata(*_stack())
    assert list(out) == ["water_like", "vegetated", "bright_bare_built", "other"]


def test_assign_strata_priority_water_over_veg():
    ndvi = np.array([0.70])   # vegetated by NDVI...
    mndwi = np.array([0.50])  # ...but water wins (documented priority).
    bright = np.array([0.10])
    assert list(S.assign_strata(ndvi, mndwi, bright)) == ["water_like"]


def test_assign_strata_threshold_overrides_are_honoured():
    ndvi, mndwi, bright = _stack()
    out = S.assign_strata(ndvi, mndwi, bright, ndvi_veg=0.70)
    assert list(out)[1] == "other"  # veg pixel drops out under stricter thr
    out2 = S.assign_strata(ndvi, mndwi, bright, mndwi_water=0.90)
    assert list(out2)[0] == "other"


def test_assign_strata_rejects_shape_mismatch_and_empty():
    with pytest.raises(ValueError):
        S.assign_strata(np.array([0.1]), np.array([0.1, 0.2]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        S.assign_strata(np.array([]), np.array([]), np.array([]))


def test_strata_counts_covers_all_strata():
    out = S.assign_strata(*_stack())
    assert S.strata_counts(out) == {
        "water_like": 1, "vegetated": 1, "bright_bare_built": 1, "other": 1}


def test_seed_and_composite_path_come_from_config():
    cfg = load_labelling_config()["labelling"]
    assert S.default_seed() == 42 == cfg["seed"]
    assert S.default_composite_path() == Path(str(cfg["s2_post_monsoon_cog"]))
    assert S.default_composite_path().name == S.COMPOSITE_FILENAME


def test_missing_composite_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="[Ee]xport|[Cc]omposite|Earth Engine"):
        S.load_composite_indices(tmp_path / "does_not_exist.tif")
    # Repo state: the EE export has not been run (audit A5).
    with pytest.raises(FileNotFoundError):
        S.load_composite_indices()


def test_compute_topup_needs_share_and_floor():
    # 2000 labelled, water at 2% -> needs 100-40=60 (5% target == 100 floor).
    needs = S.compute_topup_needs({"water": 40, "tree_cover": 500}, 2000)
    assert needs["water"] == 60
    assert "tree_cover" not in needs
    # Larger total: 5% target (175) dominates the 100 floor.
    needs2 = S.compute_topup_needs({"water": 120}, 3500)
    assert needs2["water"] == 55
    with pytest.raises(ValueError):
        S.compute_topup_needs({"water": 1}, 0)


def _pool(n_per_stratum=20):
    pool = []
    blocks = {"water_like": "B0", "vegetated": "B1",
              "bright_bare_built": "B2", "other": "B3"}
    k = 0
    for s in S.STRATA:
        for _ in range(n_per_stratum):
            pool.append({"lon": 78.20 + k * 0.001, "lat": 17.30,
                         "stratum": s, "block": blocks[s],
                         "split": "test", "fine_block": f"fx{k}_fy0"})
            k += 1
    return pool


def test_plan_topup_counts_blocks_and_ids():
    pool = _pool()
    pts = S.plan_topup(pool, {"water_like": 8, "bright_bare_built": 4}, seed=42)
    assert len(pts) == 12
    assert sum("stratum=water_like" in p["notes"] for p in pts) == 8
    assert sum("stratum=bright_bare_built" in p["notes"] for p in pts) == 4
    assert [p["id"] for p in pts] == [f"HYD-T2-{3501 + k:04d}" for k in range(12)]
    assert {p["block"] for p in pts} <= {"B0", "B2"}  # blocks inherited, none new
    assert all(p["split"] == "test" for p in pts)
    assert all(p["label"] is None and p["label_name"] == "" for p in pts)
    assert all("suggested_label" not in p for p in pts)
    # 12 top-up pts at 0.10/group -> ~1-2 overlap rows with T2 ids.
    ov = [p for p in pts if p["overlap_id"]]
    assert all(o.startswith("OV-T2-") for o in [p["overlap_id"] for p in ov])


def test_plan_topup_deterministic_same_seed():
    pool = _pool()
    a = S.plan_topup(pool, {"vegetated": 5}, seed=42)
    b = S.plan_topup(pool, {"vegetated": 5}, seed=42)
    assert [p["id"] for p in a] == [p["id"] for p in b]
    assert [(p["geometry"], p["block"]) for p in a] == [(p["geometry"], p["block"]) for p in b]


def test_plan_topup_short_pool_raises_not_invents():
    pool = _pool(n_per_stratum=2)
    with pytest.raises(ValueError, match="[Pp]ool"):
        S.plan_topup(pool, {"water_like": 10}, seed=42)


def test_plan_topup_rejects_unknown_stratum_and_bad_rows():
    pool = _pool(n_per_stratum=2)
    with pytest.raises(ValueError, match="Unknown stratum"):
        S.plan_topup(pool, {"cloud": 1}, seed=42)
    bad = [{"lon": 78.3, "lat": 17.3, "stratum": "water_like"}]  # no block/split
    with pytest.raises(KeyError):
        S.plan_topup(bad, {"water_like": 1}, seed=42)


def _code_without_docstrings_and_comments(src: str) -> str:
    """Return source with docstrings and `#` comments blanked (code only)."""
    tree = ast.parse(src)
    blank: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                seg = body[0]
                blank.update(range(seg.lineno, (seg.end_lineno or seg.lineno) + 1))
    lines = src.splitlines()
    code = ["" if i + 1 in blank else ln.split("#", 1)[0] for i, ln in enumerate(lines)]
    return "\n".join(code)


def test_strata_code_never_writes_label_or_touches_benchmarks():
    """Hard-rule gate: no `label` value is ever written; no benchmark reads."""
    src = STRATA_SRC.read_text(encoding="utf-8")
    code = _code_without_docstrings_and_comments(src).lower()
    for token in ("dynamic_world", "worldcover", "crosswalk", "suggested_label",
                  "earthengine", "sentinelhub"):
        assert token not in code, f"forbidden token in strata.py code: {token}"
    tree = ast.parse(src)
    writes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value in (
                        "label", "label_name", "suggested_label"):
                    if not (isinstance(v, ast.Constant) and v.value in (None, "")):
                        writes.append(f"{k.value}={ast.dump(v)}")
    assert writes == [], f"strata.py writes label values: {writes}"
    # Runtime proof: every planned point carries an empty label.
    pts = S.plan_topup(_pool(), {"other": 3}, seed=42)
    assert all(p["label"] is None and p["label_name"] == "" for p in pts)
