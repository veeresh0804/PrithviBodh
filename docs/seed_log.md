# Seed log — grid acceptance draws (covariate gates only, never accuracy)
Rule (`configs/labels/labelling.yaml` seed_acceptance, checked by
`python -m geoeco.labels.seed_check`): core_both_sides (each side ≥5% of its
own points within 10 km of 17.38N 78.48E — set post-hoc, disclosed in config)
AND mean_dist_gap ≤ 5.0 km AND lon_band_span ≥ 4/5 per side. First ACCEPT in
scan order wins. Thresholds calibrated from a pilot scan, then frozen
(see labelling.yaml comments).

## INVALIDATED — pre-fix geometry (non-deterministic draws, must not be used)
A set-iteration-ordering bug in `assign_cells_zoned` made draws depend on
`PYTHONHASHSEED` (same seed, different skeletons per process; caught by
`test_zoned_stable_across_hash_seeds`, fixed by sorting). Every acceptance
below is VOID for promotion decisions:
- Seed 42 grid draw: REJECT (gap 8.39 km, train core 0).
- Scan [3, 999, 7, 11, 43–52]: seed 3 ACCEPTED (gap 3.33) — VOID (hash luck).
- 8-draw pilot (gaps 1.4–10.9 km, ~25% pass): calibration input only.

## SUPERSEDED — non-zoned grid, seed 5 (was canonical; archived 2026-10-06)
- Seed 3: REJECT (gap 7.34, train core 0). Seed 4: QUOTA-FAIL (train 34/36).
- Seed 5: ACCEPT — gap 1.32 km (23.43 vs 24.75), core test=226/train=16,
  bands 5/5 + 4/5, min test–train 5.053 km, cells 48/36/72.

## Zoned candidate — seed 6 (ACCEPT; promoted below, temp copies removed)
Zone quotas proportional to assignable cells (core 4/3, mid 23/17, far 21/16):
- Seed 5: QUOTA-FAIL (core test 3/4 — greedy order fills train first).
- Seed 6: ACCEPT — gap 1.05 km (22.94 vs 23.99), core test=177/train=123,
  bands 5/5 both sides, min test–train 5.055 km, cells per zone as quota'd.
- Promotion (replace canonical skeleton + members + QGIS) is a team decision;
  canonical `data/labels/*` still seed-5 non-zoned until then.

## PROMOTED 2026-10-06 — zoned seed 6 is canonical (`data/labels/*`)
- Team decision: adopt grid with zone stratification. `grid_seed: 5→6`,
  `grid_cells_zoned` quotas pinned in labelling.yaml (core 4/3, mid 23/17,
  far 21/16). Seed-5 artefacts archived to `data/labels/audit/superseded_seed5/`.
- Checker on the promoted file: ACCEPT — gap 1.05 km, core test=177 (≥100) /
  train=123 (≥75) under the 5% proportion gate, bands 5/5 both sides,
  min test–train 5.055 km. Members A 1,269 / B 1,297 / C 1,286, overlap 352.
- canonical skeleton sha256: 449ad05e8aa737dc8ec888c64a6a7e868bf2a822a6c529e8d66ec920655e2bd3
  (hash of the regeneration-canonical serialization defined in
  `tests/test_labelling_repro.py::test_canonical_checksum` — sorted point
  dicts, `sort_keys`, no indent — verified point-identical to the promoted
  file on 2026-10-06; `test_canonical_checksum` enforces it in CI).
- KNOWN LIMIT (3-cell core caveat): train core points (123) sit in ~3 cells of
  5 km (~41 pts/cell); within-cell points are spatially correlated, so the
  effective "dense urban" training sample is smaller than 123 suggests. The
  core cannot hold more balanced cells at this cell size (exhaustive search:
  max 3+3). Stated here, not a rejection reason. Revisit only with smaller
  cells + a new accepted seed, never by relabelling in place.
