# Model Card — <model name / version>

## 1. Identity
- Model ID / MLflow run ID:
- Family: [RF-optical | RF-SAR | RF-fused | decision-fusion | AlphaEarth-RF | U-Net | dual-branch | TerraMind | DOFA]
- Training config: `configs/model/<file>.yaml` (+ git commit):
- Data version (DVC): AOI / years / seasons:

## 2. Intended use + limits
- Use: seasonal land-cover mapping (10 m) for Hyderabad pilot; two-state only if §8 evaluated.
- Out of scope: crop type, flood nowcasting, non-Deccan regions without re-validation.

## 3. Training data
- Labels: own Hyd points/patches (train blocks) + DW tiles (pretrain, remapped 9→6).
- Features: S1 VV/VH + S2 10 bands + indices (NDVI/EVI/MNDWI/NDBI) + DEM elev/slope.
- Final test data (independent Hyd points/blocks) NEVER used here.

## 4. Evaluation (spatial block CV, 5-fold, seed 42)
| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Macro-F1 | | |
| OA / kappa / mIoU | | |
| Per-class F1 (6) | | |
- Fusion Δ macro-F1 (fused − best single): __ [CI: __] (NFR-01: CI lower > 0).
- Cloud-stress: monsoon drop fused __ vs optical-only __.
- Random-vs-spatial inflation: __ (demonstrates leakage control).
- Benchmark on same points: ours __ vs Dynamic World __ vs WorldCover __.

## 5. Known failures
- Granite/rocky vs rooftop (texture+context); seasonal tanks (majority state rule);
  fallow vs bare (multi-season NDVI). Confidence-masked change recommended.

## 6. Reproducibility
- Seed 42; env `requirements.lock`; rerun metrics ±0.5pp (NFR-03).
- Inference: 256px tiles, 32px overlap, Gaussian blend; Hyd <15 min on 1×T4.

## 7. Ethics / licences
- SoI boundaries only. DW-derived pretraining attributed CC BY-4.0 (see `docs/licence_register.md`).
