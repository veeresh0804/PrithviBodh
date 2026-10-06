# Dataset Card — <dataset name / DVC version>

## 1. Identity
- DVC data version / STAC catalog:
- AOI config: `configs/aoi/<file>.yaml`; data config: `configs/data/sentinel.yaml`
- Years / seasons: [2019, 2025] × [pre Mar-May, monsoon Jun-Sep, post Oct-Dec]

## 2. Sources + licences
- Sentinel-2 L2A `COPERNICUS/S2_SR_HARMONIZED` + cloud `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` (thr 0.6) — free/open (Copernicus).
- Sentinel-1 GRD IW VV+VH `COPERNICUS/S1_GRD` — free/open (Copernicus).
- DEM Copernicus GLO-30 (elevation+slope) — verify licence before redistributing.
- Dynamic World train tiles (PANGAEA DOI 10.1594/PANGAEA.933475) — **CC BY-4.0, attribution required**.
- Own Hyd labels (~2000 test pts, ~1500 train pts, ~100 256×256 patches, ~400 change pts).

## 3. Processing
- S2 cloud-masked median/season + valid-obs count; S1 edge-mask + Refined Lee (focal-median fallback),
  VV/VH dB + ratio + seasonal VH std; co-registered 10 m grid (`EPSG:32644` Hyd).
- Stack/season 20 ch (14 optical + 4 SAR + 2 terrain); export COG int16 + STAC.
- Classes (6): water, tree_cover, cropland, built_up, bare_rocky, grass_shrub (+ crosswalk DW/WC→6).

## 4. Splits + leakage control
- Spatial blocks 3–10 km, 5-fold GroupKFold, seed 42; test blocks disjoint (`tests/test_leakage.py`).
- Labelling protocol + 10% double-labelling agreement ≥85%.

## 5. Known limitations
- S1B failure Dec 2021 → reduced 2022–2024 coverage; primary years 2019/2025 only.
- Monsoon optical sparsity (valid-obs count provided); rock-vs-rooftop ambiguity documented.
