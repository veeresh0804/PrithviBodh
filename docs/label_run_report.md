# Labelling Run Report — 2026-10-06 (automated part)

## Counts
- Skeleton: 3,500 points (test 2,000 in B0-B2; train 1,500 in B3-B4). Seed 42
  (asserted equal to `configs/eval/spatial_cv.yaml`).
- Overlap: 350 points (10%), selected per (split, block); pairs cycle A-B/B-C/C-A.
- Member files: A 1,292 / B 1,278 / C 1,280 rows (primary ~1,167 + overlap copies).
- All `label` fields empty (rule 1 verified by test). No `suggested_label`
  anywhere (prelabel default OFF; verified by test).
- QGIS projects: `data/labels/qgis/member_{A,B,C}.qgz` (valid zip + XML,
  points layer + Esri/Google XYZ + label 0-5 dropdown, no classification layer).

## Tests
- `tests/test_labelling.py`: 9 passed (determinism, disjoint B0-2/B3-4,
  empty labels, overlap size/spread, member counts, gate pass+fail fixtures,
  qgz validity, prelabel loud-fail, merge blocked when unlabelled).
- `tests/test_leakage.py`, `tests/test_indices.py`, `tests/test_api.py`: pass.
- `validate.py` on current members: FAILS as designed — 3,850 empty labels
  (humans pending), leakage_ok=true, agreement pending (no labelled pairs).

## Files written (code, committed)
- `configs/labels/labelling.yaml` (all pipeline params; members A/B/C)
- `geoeco/labels/{pipeline,qgis,prelabel,validate,merge}.py`
- `tests/test_labelling.py`, `Makefile`, `docs/dataset_card.md`
- DVC `labelling` stage + CI labelling-test step
- Generated data (local, gitignored): skeleton (REGENERATED — replaced the
  earlier B0-B9 helper output), `members/`, `overlap_index.csv`, `qgis/`.

## Assumptions / provisional deviations (logged, not hidden)
1. Class-stratified placement impossible pre-labelling without a class map;
   DW/WorldCover use forbidden for test design (rule 2) → uniform seeded
   sampling + post-hoc balance check with <5% top-up rule.
2. Fold bands ~11.9 km wide exceed spatial-CV 3-10 km range → provisional
   deviation; 5 km `fine_block` column retained (in-range).
3. Overlap spread across members+folds (class spread impossible pre-labelling).
4. QGIS projects omit S2 COG + admin boundary (files missing — see below).

## Missing files (rule 4 — stopped, not invented)
- `data/raw/composites/s2_post_2025.tif` (EE export) → re-run `qgis` module after.
- `data/admin/hyderabad_boundary.geojson` (SoI official) → same.
- `data/raw/benchmarks/{dynamic_world,worldcover}_hyd.tif` → `--prelabel-train`
  refuses until exported.

## Still needs a human
1. Labelling: open each `member_{A,B,C}.qgz`, fill label dropdown per
   `docs/labelling_protocol_v1.md` (no suggestions shown for any point).
2. Adjudication: `validate.py` prints the disagreement queue on failure;
   guide resolves into `adjudications.csv`, then `make labels-merge`.
3. Re-run `qgis` module after S2/admin exports so projects include them.
