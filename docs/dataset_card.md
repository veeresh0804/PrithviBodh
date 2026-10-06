# Dataset Card — Hyderabad labels (Batch 58)

## Contents
- `data/labels/hyd_sampling_skeleton.geojson` — machine-generated point
  locations only (`label` always empty). Regenerable: `make labels-skeleton`.
- `data/labels/members/member_{A,B,C}.geojson` — per-member work files.
- `data/labels/overlap_index.csv` — 10% double-labelling manifest.
- `data/labels/qgis/member_{A,B,C}.qgz` — QGIS workspace per member.
- `data/labels/hyd_points.geojson` — frozen set, written ONLY by
  `make labels-merge` after validation passes (+ `VERSION.json` stamp).

## Protocol
`docs/labelling_protocol_v1.md` + code in `geoeco/labels/protocol.py`
(6 classes; R1 granite/rooftop, R2 seasonal tanks, R3 fallow vs bare).
Agreement gate ≥ 85% (`geoeco/labels/agreement.py`).

## Pre-labelling (train-only suggestions) — DEFAULT OFF
`--prelabel-train` writes `suggested_label` (never `label`) on TRAIN rows
where Dynamic World and WorldCover agree after the benchmark crosswalk.
KNOWN COUPLING: models trained on confirmed suggestions are partly coupled
to benchmarks B1/B2 — this is label noise by construction and MUST be
reported alongside those models' benchmark comparisons. Test points are
never suggested, never shown a suggestion, never derived from benchmarks.

## Licences / attribution
- Human labels: team-collected for this project.
- Dynamic World training tiles (if used for pretraining): CC BY-4.0,
  PANGAEA DOI 10.1594/PANGAEA.933475 — attribute in report + dashboard.
- Basemaps in QGIS projects: Esri World Imagery / Google Satellite (view-only).

## Known limitations
- Class balance verified post-labelling (`validate.py` balance report);
  classes under 5% trigger a top-up round.
- Fold bands (~11.9 km) exceed the spatial-CV 3-10 km range (provisional;
  see run report). Fine-block column (5 km) retained for analysis.

## Stratification decision (pre-labelling; composites pending)

Status: DECIDED, NOT YET APPLICABLE — `data/raw/composites/` does not exist
(audit A5), so no stratum has been computed and no point has been moved.
Placement stays uniform-random (seed 42) until the EE export below lands.

Why: the audit found point placement is uniform-random with no class strata,
because true class stratification needs a class map and the only available
maps (Dynamic World / WorldCover) are forbidden for test design (hard
rule 2). Fix: rough strata from the project's OWN Sentinel-2 post-monsoon
composite only (`geoeco/labels/strata.py`). Strata are sampling guides —
they NEVER become labels (`label` stays empty until human labelling).

Provisional strata rule table (priority: first match wins; all thresholds
PROVISIONAL until calibrated on the real composite):

| # | stratum | rule | provisional threshold |
|---|---------|------|-----------------------|
| 1 | water_like | MNDWI high | MNDWI > 0.2 |
| 2 | vegetated | NDVI high | NDVI > 0.4 |
| 3 | bright_bare_built | low NDVI + high brightness | NDVI < 0.2 AND mean(B2,B3,B4,B8) > 0.25 |
| 4 | other | anything else | -- |

Indices: NDVI=(B8-B4)/(B8+B4), MNDWI=(B3-B11)/(B3+B11),
brightness=mean(B2,B3,B4,B8), reflectance 0-1.

Exact EE export requirement (blocked): S2_SR_HARMONIZED post-monsoon
(months 10-12, 2025; repeat 2019) median + valid-obs count, Cloud Score+
mask cs<=0.6, bands B2,B3,B4,B8,B11,B12 at 10 m, clip to Hyderabad AOI,
EPSG:32644, to `data/raw/composites/s2_post_2025.tif` (config
`s2_post_monsoon_cog`) + STAC entry. Loader raises FileNotFoundError
until this exists — nothing invented.

Two-round top-up protocol (strata are rough, balance enforced here):
1. Label round 1, run `python -m geoeco.labels.validate` (flags classes
   < 5% in `top_up`).
2. `compute_topup_needs()` targets >= 5% share AND >= ~100 test points per
   class; `needs_to_strata()` maps deficit classes to guide strata
   (water->water_like, tree/crop->vegetated, built/bare->bright_bare_built,
   grass/shrub->other); `plan_topup()` draws from those strata with seed 42
   on stream seed+200 inside already-assigned blocks (no new blocks, no
   benchmark reads). Ids HYD-T2-3501+ (never collide with round 1);
   split/block/fine_block inherited (test top-up stays in test blocks).
   ~10% per (split, block) get OV-T2- double-labelling ids appended to
   overlap_index.csv (existing OV- rows untouched). Re-run validate.py
   after round 2.

Tests: `tests/test_labelling_strata.py` (synthetic stacks only; includes a
hard-rule gate that strata code never writes `label` and never reads
benchmarks).
