"""Own-layer sampling strata + top-up protocol (PRE-LABELLING, PROVISIONAL).

Why this module exists (audit finding A1): point placement is uniform-random
(seed 42) with no class strata, because true class stratification needs a
class map and the only available maps (Dynamic World / WorldCover) are
forbidden for test design (hard rule 2). This module adds independence-
preserving sampling guides computed ONLY from the project's own Sentinel-2
post-monsoon composite (per configs/data/sentinel.yaml).

HARD RULES enforced here:
1. Strata NEVER become labels. This module writes ``label=None`` /
   ``label_name=""`` only; the word ``label`` appears below solely in that
   empty assignment and in this prohibition. Test points are human-only.
2. No Dynamic World / WorldCover (or any classification product) is read,
   imported, or referenced in code paths. Strata come from our own
   NDVI / MNDWI / brightness stack.
3. Seed comes from configs/labels/labelling.yaml (asserted equal to
   configs/eval/spatial_cv.yaml by pipeline.load_labelling_config).
   Thresholds are explicit function parameters; the module-level
   PROVISIONAL_* values are fallbacks until the composite exists and they
   are calibrated + promoted into config. Nothing else hard-coded.
4. Missing composite -> FileNotFoundError with the exact EE export
   requirement (rule 4: never invent data).

PROVISIONAL strata rule table (priority order: first match wins):

| # | stratum            | rule (post-monsoon composite)              | provisional threshold |
|---|--------------------|--------------------------------------------|-----------------------|
| 1 | water_like         | MNDWI > mndwi_water                        | MNDWI > 0.2           |
| 2 | vegetated          | NDVI > ndvi_veg                            | NDVI > 0.4            |
| 3 | bright_bare_built  | NDVI < ndvi_bare_max AND brightness > ...  | NDVI < 0.2, bright > 0.25 |
| 4 | other              | anything else                              | --                    |

Index definitions (reflectance 0-1): NDVI = (B8-B4)/(B8+B4),
MNDWI = (B3-B11)/(B3+B11), brightness = mean(B2,B3,B4,B8).
Thresholds are PROVISIONAL: data/raw/composites/ does not exist yet, so
they cannot be calibrated. Re-calibrate after the EE export lands, then
promote the tuned values into configs/labels/labelling.yaml (`strata:`
block) which default_thresholds() will prefer over the fallbacks.

EXACT EE export requirement (blocked until done):
- Collections: COPERNICUS/S2_SR_HARMONIZED masked with
  GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED, keep pixels with cs <= 0.6
  (cloud_threshold from configs/data/sentinel.yaml).
- Period: post-monsoon months [10, 11, 12], year 2025 (repeat for 2019
  for the change pair), per configs/data/sentinel.yaml seasons.
- Bands: B2, B3, B4, B8, B11, B12 at 10 m (20 m resampled bilinear),
  median composite + valid-observation count.
- Grid: clip to configs/aoi/hyderabad.yaml bounds_wgs84
  [78.20, 17.11, 78.76, 17.65], EPSG:32644, 10 m.
- Destination: data/raw/composites/s2_post_2025.tif (path from
  configs/labels/labelling.yaml `s2_post_monsoon_cog`), multiband COG
  with band order (B2, B3, B4, B8, B11, B12), internal tiling 256,
  + STAC entry per configs/data/sentinel.yaml export block.

TWO-ROUND top-up protocol (strata are rough, so balance is enforced here):
- Round 1: label the first batch (skeleton ids HYD-0001.. / HYD-G-0001..),
  then run `python -m geoeco.labels.validate` (balance report: classes
  with < 5% share flagged in `top_up`).
- Round 2: compute_topup_needs() converts the balance report into extra
  point counts per deficit class (target: >= 5% share AND >= ~100 test
  points per class; first-order estimate, re-run validate.py after).
  Map each deficit class to its guide stratum via STRATUM_GUIDE_FOR_CLASS
  (sampling guide only, never a label), then plan_topup() draws that many
  candidate locations from the deficit strata using the same seed 42 on a
  fresh stream (seed + TOPUP_STREAM_OFFSET) inside the already-assigned
  grid cells / blocks (no new blocks, no benchmark reads).
- IDs: HYD-T2-<n> continuing the round-1 numeric sequence (first top-up
  is HYD-T2-3501 for the 3,500-point skeleton), so round-1 and round-2
  ids can never collide (grid skeletons: same numeric continuation with
  the T2 prefix). Blocks/split/fine_block are inherited from the sampled
  location (top-up test points stay in test blocks, train in train).
- Overlap: ~overlap_fraction (config default 0.10) of top-up points per
  (split, block) get OV-T2-<split>-<block>-<n> ids, appended as new rows
  to overlap_index.csv (existing OV- rows untouched). Member assignment
  reuses the pipeline A-B / B-C / C-A pair cycle at split time.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

STRATA: tuple[str, ...] = ("water_like", "vegetated", "bright_bare_built", "other")

# PROVISIONAL fallback thresholds (see module docstring). Used only when
# configs/labels/labelling.yaml has no `strata:` block; calibrate after
# the EE export lands and promote tuned values into config.
PROVISIONAL_MNDWI_WATER: float = 0.2
PROVISIONAL_NDVI_VEG: float = 0.4
PROVISIONAL_NDVI_BARE_MAX: float = 0.2
PROVISIONAL_BRIGHTNESS_BARE_MIN: float = 0.25

# Fresh RNG stream for round 2 (round 1 uses seed, seed+1 overlap,
# seed+2 member shuffle, seed+100 grid sampling). Offset is a named
# constant so the stream choice is explicit and reproducible.
TOPUP_STREAM_OFFSET: int = 200
TOPUP_ID_PREFIX: str = "HYD-T2-"

# Balance targets (mirror geoeco/labels/validate.py `< 5%` flag + task
# `~100 test points` floor). Explicit params on compute_topup_needs();
# constants keep the two call sites consistent.
MIN_CLASS_SHARE: float = 0.05
MIN_TEST_POINTS_PER_CLASS: int = 100

# Deficit class -> guide stratum (SAMPLING GUIDE ONLY, never a label).
# Provisional: re-check after round-1 balance (e.g. grass_shrub may draw
# better from vegetated in wet post-monsoon years).
STRATUM_GUIDE_FOR_CLASS: dict[str, str] = {
    "water": "water_like",
    "tree_cover": "vegetated",
    "cropland": "vegetated",
    "built_up": "bright_bare_built",
    "bare_rocky": "bright_bare_built",
    "grass_shrub": "other",
}

# Must stay equal to configs/labels/labelling.yaml `s2_post_monsoon_cog`
# (asserted by tests/test_labelling_strata.py).
COMPOSITE_FILENAME: str = "s2_post_2025.tif"

# Band order expected in the exported multiband COG (see EE requirement).
COMPOSITE_BAND_ORDER: tuple[str, ...] = ("B2", "B3", "B4", "B8", "B11", "B12")


def default_seed() -> int:
    """Return the labelling seed from config (single source of truth).

    Returns:
        Seed from configs/labels/labelling.yaml (asserted equal to the
        spatial-CV seed by pipeline.load_labelling_config).

    Raises:
        FileNotFoundError: If a config file is missing.
        KeyError: If required keys are missing.
        ValueError: On seed mismatch between configs.
    """
    from geoeco.labels.pipeline import load_labelling_config

    return int(load_labelling_config()["labelling"]["seed"])


def default_composite_path() -> Path:
    """Return the post-monsoon composite path from config.

    Returns:
        Path from configs/labels/labelling.yaml `s2_post_monsoon_cog`.
    """
    from geoeco.labels.pipeline import load_labelling_config

    cfg = load_labelling_config()["labelling"]
    return Path(str(cfg["s2_post_monsoon_cog"]))


def default_thresholds() -> dict[str, float]:
    """Return strata thresholds: config `strata:` block if present, else PROVISIONAL.

    Returns:
        Mapping with keys mndwi_water, ndvi_veg, ndvi_bare_max,
        brightness_bare_min.
    """
    from geoeco.labels.pipeline import load_labelling_config

    cfg = load_labelling_config()["labelling"]
    block = cfg.get("strata") or {}
    return {
        "mndwi_water": float(block.get("mndwi_water", PROVISIONAL_MNDWI_WATER)),
        "ndvi_veg": float(block.get("ndvi_veg", PROVISIONAL_NDVI_VEG)),
        "ndvi_bare_max": float(block.get("ndvi_bare_max", PROVISIONAL_NDVI_BARE_MAX)),
        "brightness_bare_min": float(
            block.get("brightness_bare_min", PROVISIONAL_BRIGHTNESS_BARE_MIN)
        ),
    }


def assign_strata(
    ndvi: np.ndarray,
    mndwi: np.ndarray,
    brightness: np.ndarray,
    *,
    mndwi_water: float = PROVISIONAL_MNDWI_WATER,
    ndvi_veg: float = PROVISIONAL_NDVI_VEG,
    ndvi_bare_max: float = PROVISIONAL_NDVI_BARE_MAX,
    brightness_bare_min: float = PROVISIONAL_BRIGHTNESS_BARE_MIN,
) -> np.ndarray:
    """Assign own-layer strata from index arrays (priority: water > veg > bright > other).

    Args:
        ndvi: NDVI array (any shape, reflectance-derived).
        mndwi: MNDWI array, same shape as ndvi.
        brightness: Mean(B2,B3,B4,B8) brightness array, same shape.
        mndwi_water: MNDWI threshold for water_like.
        ndvi_veg: NDVI threshold for vegetated.
        ndvi_bare_max: NDVI ceiling for bright_bare_built.
        brightness_bare_min: Brightness floor for bright_bare_built.

    Returns:
        String array of stratum names, same shape as inputs.

    Raises:
        ValueError: On shape mismatch or empty input.
    """
    ndvi_a = np.asarray(ndvi, dtype=float)
    mndwi_a = np.asarray(mndwi, dtype=float)
    bright_a = np.asarray(brightness, dtype=float)
    if not (ndvi_a.shape == mndwi_a.shape == bright_a.shape):
        raise ValueError(
            f"Shape mismatch: ndvi{ndvi_a.shape} vs mndwi{mndwi_a.shape} "
            f"vs brightness{bright_a.shape}"
        )
    if ndvi_a.size == 0:
        raise ValueError("Index arrays must be non-empty")
    out = np.full(ndvi_a.shape, "other", dtype="<U17")
    bright_mask = (ndvi_a < ndvi_bare_max) & (bright_a > brightness_bare_min)
    out[bright_mask] = "bright_bare_built"
    out[ndvi_a > ndvi_veg] = "vegetated"
    out[mndwi_a > mndwi_water] = "water_like"
    return out


def strata_counts(strata: np.ndarray) -> dict[str, int]:
    """Count pixels per stratum (all four strata always present as keys).

    Args:
        strata: Stratum-name array from assign_strata().

    Returns:
        Mapping stratum -> count.
    """
    flat = np.asarray(strata).ravel()
    return {s: int(np.sum(flat == s)) for s in STRATA}


def load_composite_indices(
    path: str | Path | None = None,
    *,
    band_order: tuple[str, ...] = COMPOSITE_BAND_ORDER,
) -> dict[str, np.ndarray]:
    """Load the post-monsoon composite and derive NDVI/MNDWI/brightness.

    Args:
        path: Composite COG path. Defaults to the
            `s2_post_monsoon_cog` config value.
        band_order: Band names in file order (1-based inside the reader).

    Returns:
        Mapping with float32 arrays ``ndvi``, ``mndwi``, ``brightness``
        plus the source ``path`` string.

    Raises:
        FileNotFoundError: If the composite does not exist (current
            state: data/raw/composites/ is missing — run the EE export
            from the module docstring first; nothing is invented here).
        ImportError: If rasterio is not installed.
    """
    target = Path(path) if path is not None else default_composite_path()
    if not target.is_file():
        raise FileNotFoundError(
            f"Missing post-monsoon composite {target}: export it from Earth "
            "Engine first (S2_SR_HARMONIZED post-monsoon median + Cloud Score+ "
            "cs<=0.6, bands B2,B3,B4,B8,B11,B12, 10 m EPSG:32644 clipped to the "
            "Hyderabad AOI). data/raw/composites/ does not exist yet -- "
            "rule 4: cannot invent it."
        )
    try:
        import rasterio  # type: ignore[import]
    except ImportError as e:
        raise ImportError("rasterio required to read the composite COG") from e
    idx = {b: i + 1 for i, b in enumerate(band_order)}
    with rasterio.open(target) as src:
        b2 = src.read(idx["B2"]).astype(np.float32)
        b3 = src.read(idx["B3"]).astype(np.float32)
        b4 = src.read(idx["B4"]).astype(np.float32)
        b8 = src.read(idx["B8"]).astype(np.float32)
        b11 = src.read(idx["B11"]).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b8 - b4) / (b8 + b4)
        mndwi = (b3 - b11) / (b3 + b11)
    ndvi = np.where(np.isfinite(ndvi), ndvi, 0.0).astype(np.float32)
    mndwi = np.where(np.isfinite(mndwi), mndwi, -1.0).astype(np.float32)
    brightness = ((b2 + b3 + b4 + b8) / 4.0).astype(np.float32)
    return {"ndvi": ndvi, "mndwi": mndwi, "brightness": brightness, "path": str(target)}


def compute_topup_needs(
    class_counts: dict[str, int],
    total_labelled: int,
    *,
    min_share: float = MIN_CLASS_SHARE,
    min_n: int = MIN_TEST_POINTS_PER_CLASS,
) -> dict[str, int]:
    """Convert a validate.py balance report into extra-point needs per class.

    Target per class is ``max(ceil(min_share * total), min_n)``; need is
    the shortfall vs the current count (first-order: total held fixed,
    re-run validate.py after round 2 for the exact shares).

    Args:
        class_counts: Labelled-point counts per class name.
        total_labelled: Total labelled points the shares divide by.
        min_share: Minimum class share (default mirrors the validate.py
            5% top-up flag).
        min_n: Minimum points per class (task floor of ~100 test points).

    Returns:
        Mapping class -> extra points needed (only deficit classes).

    Raises:
        ValueError: On negative counts or non-positive total.
    """
    if total_labelled <= 0:
        raise ValueError(f"total_labelled must be positive, got {total_labelled}")
    target = max(int(math.ceil(min_share * total_labelled)), int(min_n))
    needs: dict[str, int] = {}
    for cls in sorted(class_counts):
        n = int(class_counts[cls])
        if n < 0:
            raise ValueError(f"Negative count for class {cls!r}: {n}")
        if n < target:
            needs[cls] = target - n
    return needs


def needs_to_strata(needs: dict[str, int]) -> dict[str, int]:
    """Map per-class needs to per-stratum draw counts (guide only).

    Args:
        needs: Per-class extra-point needs from compute_topup_needs().

    Returns:
        Mapping stratum -> points to draw from that stratum.

    Raises:
        KeyError: On unknown class names.
    """
    out: dict[str, int] = {}
    for cls in sorted(needs):
        stratum = STRATUM_GUIDE_FOR_CLASS[cls]  # KeyError on unknown class
        out[stratum] = out.get(stratum, 0) + int(needs[cls])
    return out


def plan_topup(
    candidates: list[dict[str, Any]],
    need_by_stratum: dict[str, int],
    *,
    seed: int = 42,
    start_id: int = 3501,
    id_prefix: str = TOPUP_ID_PREFIX,
    overlap_fraction: float = 0.10,
    seed_stream_offset: int = TOPUP_STREAM_OFFSET,
) -> list[dict[str, Any]]:
    """Draw round-2 top-up points from deficit strata (deterministic).

    Points reuse the already-assigned blocks/cells: each candidate carries
    its round-1 ``block``/``split``/``fine_block`` and the draw only adds
    density inside deficit strata. No benchmark is read; no new blocks.

    Args:
        candidates: Pool of candidate locations, each with lon, lat,
            stratum (one of STRATA), block, split ("test"/"train"),
            fine_block. Coordinates are floats; stratum is the
            assign_strata() output at that location.
        need_by_stratum: Stratum -> number of extra points to draw.
        seed: Labelling seed (pass default_seed(); 42 per config).
        start_id: First numeric id (round-1 skeleton holds 1..start_id-1,
            so ids never collide; default 3501 for the 3,500-point
            skeleton, same for the grid HYD-G- variant with T2 prefix).
        id_prefix: Round-2 id prefix (default HYD-T2-).
        overlap_fraction: Share of top-up points per (split, block) given
            an OV-T2- double-labelling id (mirrors the config 0.10).
        seed_stream_offset: Fresh RNG stream offset (default keeps round 2
            disjoint from round-1 seed/seed+1/seed+2/seed+100 streams).

    Returns:
        New point dicts in pipeline schema (id, block, geometry, split,
        fine_block, overlap_id; ``label`` ALWAYS None). The guide stratum
        is recorded in ``notes`` (``topup;stratum=<s>``), never as a label.

    Raises:
        ValueError: On empty pool, unknown stratum, non-positive need, or
            a deficit stratum whose pool is smaller than its need (report
            and widen the candidate pool instead of inventing points).
        KeyError: On candidate dicts missing required keys.
    """
    required = ("lon", "lat", "stratum", "block", "split", "fine_block")
    for c in candidates:
        for k in required:
            if k not in c:
                raise KeyError(f"Candidate missing required key {k!r}: {c}")
        if c["stratum"] not in STRATA:
            raise ValueError(f"Unknown stratum {c['stratum']!r}; expected one of {STRATA}")
        if c["split"] not in ("test", "train"):
            raise ValueError(f"Bad split {c['split']!r} (expected test/train)")
    if not candidates:
        raise ValueError("Candidate pool is empty — report, do not invent points")
    for s, n in need_by_stratum.items():
        if s not in STRATA:
            raise ValueError(f"Unknown stratum {s!r}; expected one of {STRATA}")
        if int(n) <= 0:
            raise ValueError(f"Need for stratum {s!r} must be positive, got {n}")
    total_need = sum(int(n) for n in need_by_stratum.values())
    if total_need == 0:
        return []

    rng = np.random.Generator(np.random.PCG64(int(seed) + int(seed_stream_offset)))
    picked: list[dict[str, Any]] = []
    for stratum in sorted(need_by_stratum):
        need = int(need_by_stratum[stratum])
        pool = [c for c in candidates if c["stratum"] == stratum]
        if len(pool) < need:
            raise ValueError(
                f"Stratum {stratum!r}: need {need} points but pool has {len(pool)} "
                "candidates — widen the pool (same seed stream, same blocks), "
                "do not invent points."
            )
        order = rng.permutation(len(pool))
        picked.extend(pool[int(i)] for i in order[:need])

    pts: list[dict[str, Any]] = []
    for k, c in enumerate(picked):
        pts.append({
            "id": f"{id_prefix}{start_id + k:04d}",
            "block": c["block"],
            "geometry": {"type": "Point", "coordinates": [float(c["lon"]), float(c["lat"])]},
            "label": None,
            "label_name": "",
            "labeller": "",
            "notes": f"topup;stratum={c['stratum']}",
            "split": c["split"],
            "fine_block": c["fine_block"],
            "overlap_id": None,
        })

    # Overlap spread mirrors pipeline.select_overlap: ~fraction per
    # (split, block), drawn on the next stream so round-2 overlap ids are
    # reproducible without touching round-1 OV- rows.
    rng_ov = np.random.Generator(np.random.PCG64(int(seed) + int(seed_stream_offset) + 1))
    groups: dict[tuple[str, str], list[int]] = {}
    for i, p in enumerate(pts):
        groups.setdefault((p["split"], p["block"]), []).append(i)
    n_ov = 0
    for key in sorted(groups):
        idx = groups[key]
        k = int(round(len(idx) * float(overlap_fraction)))
        for j in rng_ov.permutation(idx)[:k]:
            pts[j]["overlap_id"] = f"OV-T2-{key[0]}-{key[1]}-{n_ov:03d}"
            n_ov += 1
    return pts


def main(argv: list[str] | None = None) -> int:
    """CLI: check composite presence / print provisional rules (no network)."""
    ap = argparse.ArgumentParser(description="Own-layer strata checks (no network).")
    ap.add_argument("--check-composite", action="store_true",
                    help="Fail loudly if the post-monsoon composite is missing.")
    ap.add_argument("--print-rules", action="store_true",
                    help="Print the provisional strata rule table.")
    args = ap.parse_args(argv)
    if args.print_rules:
        thr = default_thresholds()
        print(f"water_like: MNDWI > {thr['mndwi_water']} (PROVISIONAL)")
        print(f"vegetated: NDVI > {thr['ndvi_veg']} (PROVISIONAL)")
        print(f"bright_bare_built: NDVI < {thr['ndvi_bare_max']} "
              f"AND brightness > {thr['brightness_bare_min']} (PROVISIONAL)")
        print("other: anything else")
    if args.check_composite:
        load_composite_indices()  # raises FileNotFoundError when missing
        print(f"composite OK: {default_composite_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
