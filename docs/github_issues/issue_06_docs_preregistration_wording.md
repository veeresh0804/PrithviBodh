TITLE: [docs] labelling.yaml comments overstated pre-registration (fixed)

**Finding**
- Comments claimed thresholds were "Pre-registered / fixed BEFORE" any look at
  results, which was not accurate: they were calibrated from an 8-draw pilot
  scan.

**Fix applied**
- Wording replaced with the honest note: thresholds set from the pilot-scan
  distribution and frozen before test-point selection (no tuning after selection).
  `configs/labels/labelling.yaml`, `docs/seed_log.md`.
