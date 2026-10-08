TITLE: [data-quality] Raise min_core_points_per_side 1 → 10 (fixed)

**Finding**
- An existence floor of 1 point lets a side's urban signal be noise in per-class
  F1 confidence intervals. 10 ≈ 0.5% of test / 0.67% of train — presence with mass.

**Fix applied**
- Threshold raised to 10 in `configs/labels/labelling.yaml`; seed acceptance
  gates updated. Canonical seed 6 passes: core 177 (train) / 123 (test) [V],
  recorded in `docs/seed_log.md`.
