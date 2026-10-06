"""Full-pipeline hash-seed reproducibility + canonical-checksum gate.

AUDIT (a) — unzoned stages and pipeline overlap/member stages: every
set/dict iteration feeding numpy RNG draws or output row order was checked.
All sites CLEAN (Python dicts are insertion-ordered; only set/frozenset
iteration is PYTHONHASHSEED-ordered, and no set feeds draws or row order):

- geoeco/labels/grid.py:160 `by_pos` dict — CLEAN (insertion-ordered, .get only).
- geoeco/labels/grid.py:161 `by_id` dict — CLEAN (insertion-ordered, lookup only).
- geoeco/labels/grid.py:167 `ys = sorted({...})` — CLEAN (sorted).
- geoeco/labels/grid.py:170 `assignable` list comp over build_cells order — CLEAN
  (deterministic row-major input, no set involved).
- geoeco/labels/grid.py:177 `rng.permutation` over a list — CLEAN (input order fixed).
- geoeco/labels/grid.py:182 `sorted(("test", "train"), ...)` — CLEAN (2-tuple, stable).
- geoeco/labels/grid.py:188 `neighbours` fixed dc/dr loops — CLEAN.
- geoeco/labels/grid.py:172 `side` dict — CLEAN (membership/get only, never iterated).
- geoeco/labels/grid.py:207 `by_id` lookups — CLEAN.
- geoeco/labels/grid.py:211 `pool` from `side.items()` — CLEAN (dict iteration is
  insertion-ordered, and insertion order here is seeded-RNG order, so it cannot
  vary with PYTHONHASHSEED; only set iteration could).
- geoeco/labels/pipeline.py:56 `set(...) & set(...)` — CLEAN (boolean gate only).
- geoeco/labels/pipeline.py:136-139 block sets — CLEAN (leak check is a boolean;
  only output goes through `sorted()` at line 143).
- geoeco/labels/pipeline.py:156-158 `groups.setdefault` over list order — CLEAN.
- geoeco/labels/pipeline.py:159 `for key in sorted(groups)` — CLEAN (sorted).
- geoeco/labels/pipeline.py:162 `rng.permutation(idx)` over a list — CLEAN.
- geoeco/labels/pipeline.py:198 `rng.permutation(len(pts))` int arg — CLEAN.
- geoeco/labels/pipeline.py:199-201 `primary` keyed by `mids` list, round-robin — CLEAN.
- geoeco/labels/pipeline.py:204 `sorted(overlap, key=id)` — CLEAN.
- geoeco/labels/pipeline.py:207 `next(m for m in (a, b) ...)` fixed tuple — CLEAN.
- geoeco/labels/pipeline.py:211-212 `for m in mids`, rows `sorted` by id — CLEAN.

CONCLUSION: the only ordering hazard was in `assign_cells_zoned` (fixed by
sorting; covered by `test_zoned_stable_across_hash_seeds`). This module covers
the FULL zoned pipeline (assignment + sampling + overlap + member split)
across hash seeds instead. No source files were changed for this audit.

All fixtures in tmp_path (never data/); seeds/counts from
configs/labels/labelling.yaml where they exist; no network.
"""

import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

from geoeco.labels import grid as G
from geoeco.labels.pipeline import (
    REPO,
    load_labelling_config,
)

SEED_LOG = REPO / "docs" / "seed_log.md"
CANON_RE = re.compile(r"canonical skeleton sha256:\s*([0-9a-fA-F]{64})")

# Tiny inline quotas for the two-process test ONLY (6 test / 4 train cells;
# keeps the full pipeline under 2 s). Production uses the full per-zone scheme
# (core 4/3, mid 23/17, far 21/16), exercised by tests/test_labelling_zones.py.
TINY_QPZ = {
    "core": {"test": 1, "train": 1},
    "mid": {"test": 3, "train": 2},
    "far": {"test": 2, "train": 1},
}
TINY_N_TEST, TINY_N_TRAIN = 6, 4

# Fallback full-scale zoned quotas from docs/seed_log.md (zoned candidate
# scheme; sums equal labelling.yaml grid_cells test/train). Used only when the
# config gains no explicit per-zone quota key (see _zoned_quotas).
FALLBACK_QPZ = {
    "core": {"test": 4, "train": 3},
    "mid": {"test": 23, "train": 17},
    "far": {"test": 21, "train": 16},
}

_HELPER_SRC = """\
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "@@REPO@@")
from geoeco.labels import grid as G
from geoeco.labels.pipeline import load_labelling_config, select_overlap, split_members

ctx = load_labelling_config()
cfg, aoi = ctx["labelling"], ctx["aoi"]
cells = G.build_cells(*[float(v) for v in aoi["bounds_wgs84"]],
                      float(cfg["fine_block_size_m"]))
edges = tuple(float(v) for v in cfg["zone_edges_km"])
# Tiny quotas inline for speed (see TINY_QPZ in the parent test module).
qpz = {"core": {"test": 1, "train": 1}, "mid": {"test": 3, "train": 2},
       "far": {"test": 2, "train": 1}}
side = G.assign_cells_zoned(cells, int(cfg["grid_seed"]), qpz,
                            float(cfg["min_cell_area_fraction"]), edges)
pts = G.sample_grid_points(side, cells, 6, 4, int(cfg["grid_seed"]))
select_overlap(pts, float(cfg["overlap_fraction"]), int(cfg["seed"]))
outdir = Path(sys.argv[1])
paths, _ = split_members(pts, list(cfg["members"]), int(cfg["seed"]), outdir)
rows = []
for p in sorted(paths, key=lambda x: x.name):
    data = json.loads(p.read_text(encoding="utf-8"))
    for f in data["features"]:
        rows.append({"member": p.stem, "id": f["id"], "geometry": f["geometry"],
                     "properties": f["properties"]})
rows.sort(key=lambda r: (r["member"], r["id"]))
print(hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest())
"""


def test_full_pipeline_stable_across_hash_seeds(tmp_path):
    """Two-process determinism over the FULL zoned pipeline.

    Runs assignment + sampling + overlap + member split in two subprocesses
    with PYTHONHASHSEED=0 vs 1 and requires identical member-row hashes, so
    any set/dict ordering leak in the overlap/member stages would fail.
    """
    helper = tmp_path / "full_pipe.py"
    helper.write_text(_HELPER_SRC.replace("@@REPO@@", REPO.as_posix()), encoding="utf-8")
    outs = []
    for hs in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run(
            [sys.executable, str(helper), str(tmp_path / f"m{hs}")],
            capture_output=True, text=True, cwd=str(REPO), env=env, timeout=120,
            check=False,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        h = r.stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{64}", h), f"helper did not print a hash: {h!r}"
        outs.append(h)
    assert outs[0] == outs[1]


def _zoned_quotas(cfg):
    """Per-zone quotas: explicit config key wins, else seed_log fallback.

    Fails loudly if the fallback no longer sums to the config grid_cells
    totals (promotion changed totals without recording a new scheme).
    """
    for key in ("grid_cells_zoned", "zoned_grid_cells", "grid_cells_by_zone"):
        if key in cfg:
            return {z: dict(v) for z, v in cfg[key].items()}
    qpz = {z: dict(v) for z, v in FALLBACK_QPZ.items()}
    if "grid_cells" in cfg:
        want = (int(cfg["grid_cells"]["test"]), int(cfg["grid_cells"]["train"]))
        got = (sum(v["test"] for v in qpz.values()), sum(v["train"] for v in qpz.values()))
        if got != want:
            raise ValueError(
                f"Zoned quota fallback sums to {got} but config grid_cells "
                f"requires {want}; record a new scheme, do not relax silently.")
    return qpz


def _canonical_zoned_points():
    """Regenerate the canonical zoned skeleton exactly as promotion does.

    grid_seed + per-zone quotas + sample counts from labelling.yaml splits,
    over the config AOI with config cell size / sliver fraction / zone edges.
    Mirrors grid.main but with assign_cells_zoned (no overlap/member files:
    the grid skeleton carries overlap_id None, as in grid.main).
    """
    ctx = load_labelling_config()
    cfg, aoi = ctx["labelling"], ctx["aoi"]
    cells = G.build_cells(*[float(v) for v in aoi["bounds_wgs84"]],
                          float(cfg["fine_block_size_m"]))
    edges = tuple(float(v) for v in cfg["zone_edges_km"])
    side = G.assign_cells_zoned(cells, int(cfg["grid_seed"]), _zoned_quotas(cfg),
                                float(cfg["min_cell_area_fraction"]), edges)
    return G.sample_grid_points(side, cells, int(cfg["splits"]["test"]),
                                int(cfg["splits"]["train"]), int(cfg["grid_seed"]))


def test_canonical_checksum(tmp_path):
    """Canonical zoned skeleton matches the checksum committed in seed_log.md.

    This test starts passing only after promotion: the orchestrator records a
    `canonical skeleton sha256: <64-hex>` line in docs/seed_log.md (computed
    with _canonical_zoned_points serialization below) when the zoned skeleton
    is promoted. Until that line exists, this test SKIPS (does not fail).
    """
    m = CANON_RE.search(SEED_LOG.read_text(encoding="utf-8"))
    if not m:
        pytest.skip("canonical checksum not yet recorded")
    pts = _canonical_zoned_points()
    blob = json.dumps(sorted(pts, key=lambda p: p["id"]), sort_keys=True)
    (tmp_path / "canonical_zoned_skeleton.json").write_text(blob, encoding="utf-8")
    assert hashlib.sha256(blob.encode("utf-8")).hexdigest() == m.group(1).lower()
