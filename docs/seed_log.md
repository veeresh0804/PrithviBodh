# Seed log — grid acceptance draws (covariate gates only, never accuracy)

Rule (`configs/labels/labelling.yaml` seed_acceptance, checked by
`python -m geoeco.labels.seed_check`): core_both_sides (≥1 pt/side within
10 km of 17.38N 78.48E) AND mean_dist_gap ≤ 5.0 km AND lon_band_span ≥ 4/5
per side. First ACCEPT in scan order wins. Thresholds calibrated from a
pilot scan, then frozen (see labelling.yaml comments).

## INVALIDATED — pre-fix geometry (non-deterministic draws, must not be used)
A set-iteration-ordering bug in `assign_cells_zoned` made draws depend on
`PYTHONHASHSEED` (same seed, different skeletons per process; caught by
`test_zoned_stable_across_hash_seeds`, fixed by sorting). Every acceptance
below is VOID for promotion decisions:
- Seed 42 grid draw: REJECT (gap 8.39 km, train core 0).
- Scan [3, 999, 7, 11, 43–52]: seed 3 ACCEPTED (gap 3.33) — VOID (hash luck).
- 8-draw pilot (gaps 1.4–10.9 km, ~25% pass): calibration input only.

## Current canonical — non-zoned grid, seed 5 (ACCEPT, `data/labels/*`)
- Seed 3: REJECT (gap 7.34, train core 0). Seed 4: QUOTA-FAIL (train 34/36).
- Seed 5: ACCEPT — gap 1.32 km (23.43 vs 24.75), core test=226/train=16,
  bands 5/5 + 4/5, min test–train 5.053 km, cells 48/36/72.

## Zoned candidate — seed 6 (ACCEPT, NOT promoted; skeleton in temp only)
Zone quotas proportional to assignable cells (core 4/3, mid 23/17, far 21/16):
- Seed 5: QUOTA-FAIL (core test 3/4 — greedy order fills train first).
- Seed 6: ACCEPT — gap 1.05 km (22.94 vs 23.99), core test=177/train=123,
  bands 5/5 both sides, min test–train 5.055 km, cells per zone as quota'd.
- Promotion (replace canonical skeleton + members + QGIS) is a team decision;
  canonical `data/labels/*` still seed-5 non-zoned until then.
