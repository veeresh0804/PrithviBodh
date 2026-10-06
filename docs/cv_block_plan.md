# Spatial-CV block plan — grid design (36 train cells, 5-fold GroupKFold)

Scope: grid alternative from `data/labels/audit/` (48 test / 36 train /
72 buffer cells, 5 km, seed 42). Provisional until the semivariogram
revisit below. Companion: `configs/eval/spatial_cv.yaml`
(`GroupKFold`, `n_splits: 5`, `group_by: spatial_block_id`,
bootstrap 95% CI).

## 1. Fold arithmetic: ~7 cells per fold

5-fold GroupKFold over 36 train cells → `36 / 5 = 7.2`, i.e. folds of
7–8 cells each (~290–330 train points per validation fold at
1,500 train points / 36 cells ≈ 42 pts/cell). Each fit trains on
~28–29 cells.

## 2. What to expect: noisy folds, wide CIs

With only ~7 largely-independent spatial units per validation fold,
expect **noisy fold-to-fold macro-F1** (single cells with lakes, core
urban fabric, or rare classes swing their fold) and **wide bootstrap
CIs**. This is inherent to honest spatial CV at this sample size, not a
bug. It directly conditions NFR-01 (PROJECT_CONTEXT.md §3.3: fused beats
best single-sensor in macro-F1 on spatial holdout with the bootstrap
95% CI excluding zero — `ci: { method: bootstrap, level: 0.95 }` in
`configs/eval/spatial_cv.yaml`): a real fusion gain must clear a wide
interval, so small deltas will correctly read as inconclusive. Report
per-fold macro-F1 plus the bootstrap CI; never a point estimate alone.
Do NOT shrink CIs by switching to random splits
(`report_random_split_only_for_inflation_demo`, Kattenborn 2022).

## 3. Grouping rule: fold on cell IDs — never merged bands

`GroupKFold` groups MUST be the grid **cell IDs** (`fine_block` / cell
`id`, e.g. `g04_10`). Never merge cells back into the B0–B4 longitude
bands for CV: the current band design has only 2 train-side blocks
(B3, B4), on which 5-fold GroupKFold crashes
(`Need >= 5 distinct blocks, got 2` — audit §A4). Merging cells into
bands reintroduces exactly that failure and destroys the 5.03 km
buffer guarantee the grid was adopted for.

## 4. Semivariogram revisit trigger (before first-batch labelling or never)

The 5 km cell/buffer is provisional: `data/raw/composites/` does not
exist yet, so the range cannot be computed and was not guessed (audit
§A5; expected 3–10 km per `configs/eval/spatial_cv.yaml`).

- **Trigger:** once `data/raw/composites/` exists (post-monsoon NDVI,
  VH, elevation at 10 m over the AOI), recompute the empirical
  semivariogram range before any labelling batch is drawn.
- **If range > 5 km:** widen the buffer (larger cells and/or wider than
  one-cell 8-neighbourhood) so the test–train separation still exceeds
  the range.
- **If range well below 5 km:** consider smaller cells to reclaim some
  of the 47.1% buffer area currently unsampled.
- **WARNING — do not change blocks after labelling starts.** Any
  relabelling of cell boundaries/assignments after the first batch
  invalidates the held-out test set (points sampled under one geography
  scored under another; buffer guarantees void). The revisit must happen
  **BEFORE first-batch labelling, or never.** If labelling has started,
  keep the frozen 48/36/72 assignment and record the measured range as a
  limitation instead.
