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
