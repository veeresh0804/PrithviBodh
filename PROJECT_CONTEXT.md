# Batch 58 — Production-Level Geospatial ML Project Context
**Version:** 1.0 · 6 October 2026
**Program:** CSE (AI&ML) · CMR College of Engineering & Technology · A.Y. 2026-27
**Team:** V D S S M Veeresh (23H51A66K3), Sandhya Malavath (23H51A66P7), Sehwag Lavudiya (23H51A66CW)
**Guide:** Ms. A. Spandana
**Source:** Single reference doc — replaces earlier Phase 1 / Phase 2 split with one continuous milestone-based plan (32 pages).

> "Production-level" = config-driven, reproducible, tested, versioned, served via API + map UI, documented with honest accuracy. NOT expensive infra. Everything on free tiers. ₹0 running cost.

---
## 0. How To Use This Context
This file is the canonical context for all code, docs, and chat work. It covers:
1. Research basis (verified [V] / domain knowledge [K] / our analysis [A])
2. Requirements (FR + NFR + acceptance)
3. Desired outputs (data / software / research / dashboard)
4. Architecture (data, ML, serving, deployment)
5. Implementation (repo, pipeline, labels, models, training, inference, analytics, eval, QA, MLOps)
6. Plan of action (roles, milestones M0-M8, week-by-week 1-20, DoD, risks)

If anything conflicts, this file wins over ad-hoc chat assumptions.

---
## 1. Executive Summary
**Problem:** Ecosystems around Hyderabad and Telangana/AP changing fast: urban expansion, lake/tank loss, farmland/scrub shifts. Sentinel-2 optical blinded by monsoon cloud. Sentinel-1 SAR sees through cloud but noisy/hard to interpret. Global products exist but accuracy varies widely by region.

**Solution:** Fuse Sentinel-1 SAR + Sentinel-2 optical with ML/DL to produce:
- seasonal land-cover maps at 10 m
- change maps between years
- ecosystem indicators: vegetation condition, water extent, built-up growth, degradation index
All served via web map + API, validated with spatially independent test data.

**Study area:** Hyderabad + surroundings (~60×60 km, ~3,600 km², ~36M pixels, ~550 tiles) first. Config-driven design scales to Telangana + Andhra Pradesh (~275,000 km²) without code changes.

**Credibility claims:**
1. Controlled comparison: optical-only, SAR-only, 3 fusion strategies.
2. Benchmark vs Dynamic World + ESA WorldCover on same validation points.
3. Spatial block CV (avoids inflation common in literature).
4. Cloud-stress test.
5. Fine-tune multi-sensor foundation model (TerraMind).

**Production-grade claims:** reproducible pipeline, data+model versioning, automated tests, REST API, tile server, web dashboard, Docker, model cards.

---
## 2. Research Basis
Tags: [V]=verified vs source in §14, [K]=standard knowledge, [A]=our analysis.

### 2.1 Why multi-sensor fusion
- **Sentinel-2 optical (13 bands, 10-60m, ~5-day):** rich spectra, veg/water indices. Limit: cloud-blocked, needs daylight [K].
- **Sentinel-1 SAR (C-band VV+VH, 10m):** day/night, through cloud, structure/moisture sensitive. Limit: speckle, physically different signal [K].

**Published fusion benefit [V]:**
- Moharrami et al. 2024 RS 16(9):1566: S1+S2 98.25% (migrated samples) vs 87.68% S1, 96.82% S2, SVM/RF. Note: sample-migration accuracy, NOT map accuracy.
- RS 17(7):1298 2025: dual-branch on SEN12MS 88% acc /87% F1 early fusion vs 84%/82% late. SEN12MS labels coarse (MODIS-derived).
- Multi-source S1+S2 time-series DL, ISPRS J. 2019: benefit of both modalities.
- Multi-temporal W-Net on S1: multi-temporal beat single-date +0.18 acc +0.25 F1.

### 2.2 Existing global products (baselines) [V]
- **Dynamic World (Google/WRI):** S2, 9 classes, DL, near-real-time per-image probs.
- **ESA WorldCover:** S1+S2, 11 classes, annual 2020+2021 only.
- **Esri Land Cover:** S2, 9 classes, annual.
- Accuracy region-dependent: Venter et al. 2022 — Esri 75%, DW 72%, WC 65% on global GT, but WC 71%, DW 66%, Esri 63% on European LUCAS. Malawi: 65.6% / 46.9% / 61.9% [V].
- **Gap [A]:** global, mostly optical-only, not tuned/validated for Deccan. Local fused model *may* beat locally — hypothesis to test, not assumption.

### 2.3 Foundation models / pretrained [V]
- **TerraMind (ESA Φ-lab + IBM):** any-to-any multimodal EO FM. Inputs S2 L2A, S1 GRD/RTC, DEM, land cover, NDVI. In TerraTorch. Thinking-in-Modalities can generate missing modalities. Use: main deep model, fine-tuned for segmentation.
- **DOFA:** wavelength-conditioned multi-sensor, pretrained S1/S2+, in TorchGeo. Use: second FM comparison.
- **Prithvi-EO-2.0 (NASA/IBM):** pretrained HLS (optical only). Use: optical-only FM baseline.
- **AlphaEarth Satellite Embedding (Google DeepMind):** annual 64-d per-pixel embeddings in EE. Use: ready features for RF, strong no-GPU baseline.

### 2.4 Labelled data [V]
- **DW training dataset (PANGAEA DOI 10.1594/PANGAEA.933475, CC BY-4.0):** >5B human-labelled pixels, ~24k tiles 510×510 @10m, expert + crowd + validation holdout. No imagery included, only image IDs → re-pull from EE. Use: dense labels for pretraining segmentation.
- **DW expert-consensus test tiles (Zenodo):** global comparison.
- **Own labels (mandatory):** ~2,000 hand-labelled Hyderabad points (test) + ~100 hand-drawn 256×256 patches (local fine-tune). For independent local eval.

### 2.5 Indian data [V]
- **Bhoonidhi (ISRO/NRSC):** open ≥5m products. Resourcesat LISS-III/IV, EOS-04 C-band SAR (from Mar 2022), EOS-04 terrain-normalised ARD.
- **Bhuvan (NRSC):** Indian LULC maps; qualitative cross-check.
- **Use:** Sentinel primary (EE preprocessed). EOS-04 = extension path to fill 2022-2025 S1 gap.

### 2.6 Data availability constraint [V]
- S1B failed 23 Dec 2021 → reduced coverage until S1C (regular from 26 Mar 2025). S1D calibrated since 17 Apr 2026.
- **Decision:** primary years **2019 (S1A+B) and 2025**. In-between only where coverage check passes.

### 2.7 Validation pitfall [V]
- Kattenborn et al. 2022: random hold-out can overestimate CNN accuracy by up to 28% vs spatially blocked. **We use spatial block CV throughout; report random-split only to show inflation.**

---
## 3. Requirements

### 3.1 Stakeholders
- Env/urban planners: where/how much/what kind changed.
- Researchers: reproducible maps, stats, downloads.
- Panel/guide: method works, honest validation.
- Public/students: easy map.

### 3.2 Functional (FR-01 to FR-17)
- FR-01 Must: Ingest S1 GRD + S2 L2A for any AOI/date range in config.
- FR-02 Must: Cloud-mask S2, speckle-filter S1, seasonal composites (pre-monsoon, monsoon, post-monsoon).
- FR-03 Must: Co-register all layers on common 10m grid + CRS.
- FR-04 Must: Indices NDVI, EVI, MNDWI, NDBI, VV/VH ratio; +DEM elevation/slope.
- FR-05 Must: Classify into configured classes (default 6: water, tree cover, cropland, built-up, bare/rocky, grass/shrub).
- FR-06 Must: Train/compare optical-only, SAR-only, early-fusion, decision-fusion, feature-fusion.
- FR-07 Must: Fine-tune multi-sensor FM (TerraMind) + compare.
- FR-08 Must: Change maps + from-to transition matrix between dates.
- FR-09 Must: Indicators: veg trend, water extent, built-up growth, degradation index.
- FR-10 Must: Eval with spatial block CV, benchmark vs DW+WorldCover, cloud-stress test.
- FR-11 Should: Error-adjusted (Olofsson-style) area estimation + CIs.
- FR-12 Must: Serve maps as web tiles + stats via REST API.
- FR-13 Must: Dashboard: map viewer, year/season toggle, before/after slider, draw-polygon stats.
- FR-14 Should: Downloads (GeoTIFF, GeoJSON, CSV) + auto PDF summary per area.
- FR-15 Should: Per-pixel confidence map.
- FR-16 Should: Scale to Telangana+AP by AOI config only.
- FR-17 Could: Optional EOS-04 SAR ingestion.

### 3.3 Non-functional (NFR-01 to NFR-10)
- NFR-01 Accuracy: fused beats best single-sensor in macro-F1 on spatial holdout, bootstrap 95% CI excludes zero. Absolute targets set AFTER baseline (§3.4).
- NFR-02 Robustness: under monsoon cloud, fused loses less macro-F1 than optical-only.
- NFR-03 Reproducibility: rerun from config+data version+seed → same metrics ±0.5pp.
- NFR-04 Scalability: Hyderabad inference <15min on 1×T4. Design supports ~275k km² by tiling.
- NFR-05 Performance: tiles <500ms p95 cached; polygon stats <3s up to 100km².
- NFR-06 Cost: ₹0 — EE noncommercial, Colab/Kaggle GPUs, free hosting.
- NFR-07 Maintainability: modular Python, type hints, ≥70% unit coverage core, CI every push.
- NFR-08 Transparency: model cards + dataset licences.
- NFR-09 Security: API keys for write/admin, no secrets in repo, read-only public map.
- NFR-10 Compliance: Official Survey of India boundaries; attribution for CC BY (DW).

### 3.4 Acceptance criteria
- Fusion benefit: Δ macro-F1 (fused − best single), spatial CV, bootstrap CI → CI lower >0.
- Local vs global: macro-F1 best vs DW + WorldCover on our points → Report; target ≥ best global.
- Cloud robustness: monsoon-only drop vs all-season → Fused drop < optical drop.
- Change detection: change-class F1 on labelled change sample → set after baseline (prov ≥0.6).
- Leakage: train/test blocks disjoint (auto test) → 100% pass.
- System: API+dashboard e2e + deployed URL → Pass.
- **No absolute accuracy promised upfront (deliberate): globals ~50-75% depending on validation [V]; honest targets from own baseline.**

---
## 4. Desired Outputs

### 4.1 Data products (all 10m unless noted)
- Seasonal land-cover map: COG uint8, 3 seasons × {2019,2025} Hyd; annual for two-state.
- Class prob/confidence: COG uint8 0-100, same.
- Change map + transition matrix: COG + CSV, 2019→2025.
- Veg condition (NDVI/EVI seasonal mean+change): COG per season/year.
- Water extent (fused SAR+MNDWI): COG + GeoJSON polygons per season/year.
- Built-up growth: COG 2019→2025.
- Degradation index (§6.6): COG + ward/mandal CSV 2019→2025.
- Area stats + CIs: CSV/JSON per class per admin unit per map.

### 4.2 Software
- Python package `geoeco` (`pip install -e .`)
- REST API (FastAPI) + tile server (TiTiler)
- Web dashboard (React + MapLibre GL)
- Docker Compose + public demo URL
- MLflow history + registry
- Model + dataset cards

### 4.3 Research
- Tables: 8 configs × metrics, benchmark, ablations, cloud test, random-vs-spatial.
- Figures: maps, confusion matrices, feature importance, Sankey transition.
- Final report + conference/student-journal paper draft.

### 4.4 Dashboard UX
1. Hyderabad AOI map + layer switcher: land cover, confidence, NDVI, water, change.
2. Year/season selectors + before/after swipe 2019 vs 2025.
3. Draw-polygon → class areas, change, indicators, accuracy caveats.
4. Comparison mode: ours vs DW vs WorldCover side-by-side.
5. Download (GeoTIFF/CSV) + Generate report (PDF).
6. About panel: model card + validation metrics.

---
## 5. System Architecture

### 5.1 High-level layers
`Data sources → Processing (EE) → Data layer (COG+STAC+Parquet+Chips) → ML layer (Classical/Deep/Foundation + MLflow) → Analytics (inference+change+eval) → Serving (PostGIS+TiTiler+FastAPI+React)`

Data sources: S1 GRD, S2 L2A+Cloud Score+, AlphaEarth embeddings, DEM GLO-30, DW/WorldCover, labels (own points+patches, DW tiles).

### 5.2 Layer responsibilities + tech (why)
- Processing: heavy raster near data — GEE + earthengine-api, geemap (Sentinel calibrated/terrain-corrected, no local storage).
- Raster store: analysis-ready/output — COG on Drive/Cloud Storage; R2 or HF dataset for demo (streams over HTTP, no download).
- Catalog: index rasters — STAC (pystac) static JSON (standard EO).
- Tabular: point features/labels/blocks — Parquet + pandas/geopandas (small/fast/versionable).
- Versioning: DVC (ties result to exact data).
- Training: sklearn, LightGBM, PyTorch Lightning, segmentation_models_pytorch, TorchGeo, TerraTorch (standard geo-ML).
- Config: Hydra/YAML (AOI/dates/classes/model swappable no code edits).
- Tracking: MLflow local or DagsHub free (reproducible, production model).
- DB: PostgreSQL+PostGIS (spatial “stats inside polygon”).
- Tiles: TiTiler (dynamic from COGs).
- API: FastAPI (fast/typed/OpenAPI).
- Frontend: React+MapLibre GL JS (fallback Streamlit+leafmap) (smooth, free OSS).
- Deploy: Docker Compose; HF Spaces/Render/college VM (one-command, free).
- CI/CD: GitHub Actions (free public).
- Monitoring: structured logs, /health, Uptime Kuma; drift report per new season (lightweight but real).

### 5.3 Data flow for one map
Config(AOI,dates) → EE → COG+STAC → Training(Colab GPU) → MLflow registry → Inference job → PostGIS → Dashboard. Steps: Build S1/S2 seasonal composites → Export feature stack (COG)+STAC → Chips/points → Log metrics/register best → Load production model → Stream tiles → Write LC+confidence COGs → Write area stats/change matrix → Query polygon via API + tiles via TiTiler.

### 5.4 Model family
- Early fusion: stack all channels → single encoder-decoder. Inputs/season Optical 10 bands+indices, SAR VV/VH/ratio, Terrain elev/slope.
- Feature-level: Optical encoder + SAR encoder → fuse bottleneck+skips → shared decoder.
- Decision-level: Optical model + SAR model → average/stack probs.
- Foundation: TerraMind encoder S2+S1+DEM tokens → UPerNet/UNet decoder.

### 5.5 DB schema (PostGIS)
- REGION(id PK, name, aoi geometry) contains ADMIN_UNIT, has MAP_PRODUCT.
- ADMIN_UNIT(id PK, region_id FK, name, level, geom) for AREA_STAT.
- MODEL_VERSION(id PK, mlflow_run_id, name, test_macro_f1, model_card_url) produced MAP_PRODUCT.
- MAP_PRODUCT(id PK, region_id FK, model_version_id FK, product_type, year, season, cog_url) summarised_by AREA_STAT, compared_in CHANGE_STAT.
- AREA_STAT(id PK, map_product_id FK, admin_unit_id FK, class_name, area_ha, ci_low_ha, ci_high_ha).
- CHANGE_STAT(id PK, from_product_id FK, to_product_id FK, from_class, to_class, area_ha).

### 5.6 API
- GET /health, GET /regions, GET /products?region=&type=&year=&season=, GET /tiles/{product_id}/{z}/{x}/{y}.png (proxied TiTiler colormap), GET /stats/{product_id}?admin_unit=, POST /analyze {GeoJSON polygon+product IDs → areas/change/indicators}, GET /change?from=&to=&admin_unit=, GET /compare?product_id=&baseline=dynamic_world, GET /models/{id}/card, GET /download/{product_id}, POST /admin/jobs/inference (API key).

### 5.7 Deployment
Dev push → GitHub → GH Actions (lint/tests/leakage/build) → GHCR → Host (HF Spaces/Render/college VM) → Docker Compose stack (FastAPI, TiTiler, PostGIS, React static Nginx) + Object storage COGs (R2/HF). Training/inference in Colab/Kaggle calling same `geoeco` package; serving only reads finished outputs (cheap).

### 5.8 Security/governance
Secrets in env + GH Secrets never repo. Admin needs API key; public read-only. Rate limit /analyze (100km² cap). Licence register (Sentinel free/open; DW CC BY-4.0 attribution). SoI boundaries only.

---
## 6. Implementation Design

### 6.1 Repo structure
```
geoeco/
 configs/aoi/hyderabad.yaml       # bounds, CRS EPSG:32644, admin file
 configs/aoi/telangana_ap.yaml    # scale-up
 configs/data/sentinel.yaml       # years, seasons, bands, cloud thr
 configs/model/{rf,unet,dualbranch,terramind,dofa}.yaml
 configs/eval/spatial_cv.yaml
 geoeco/ingest/      # gee_s1.py, gee_s2.py, composites.py, export.py, stac.py
 geoeco/features/    # indices.py, terrain.py, texture.py, sampling.py
 geoeco/labels/      # protocol checks, agreement, dw_tiles.py, chips.py
 geoeco/models/      # classical.py, unet.py, dual_branch.py, foundation.py, fusion.py
 geoeco/train/       # train_classical.py, train_seg.py (Lightning)
 geoeco/infer/       # tiled_inference.py, gee_inference.py
 geoeco/analytics/   # change.py, indicators.py, degradation.py, area_estimation.py
 geoeco/evaluation/  # spatial_cv.py, metrics.py, benchmark.py, cloud_stress.py, bootstrap.py
 geoeco/utils/
 api/                # FastAPI app, routers, schemas, db models
 web/                # React + MapLibre
 notebooks/          # thin callers
 tests/              # pytest indices, CRS, leakage, API
 docs/               # model/dataset cards, protocol, report
 docker-compose.yml
 dvc.yaml            # pipeline stages
 README.md
```

### 6.2 Data pipeline spec
- **S2:** COPERNICUS/S2_SR_HARMONIZED, masked GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED (thr config default 0.6). Bands B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12; 20m→10m. Indices NDVI,EVI,MNDWI,NDBI. Median/season + valid-observation count (cloud analysis).
- **S1:** COPERNICUS/S1_GRD IW VV+VH, one orbit after availability check. Edge-noise mask. Speckle Refined Lee (focal median fallback). Features VV,VH dB, VV−VH ratio, seasonal std VH (crop/water dynamics).
- **Terrain:** Copernicus DEM GLO-30 → elevation+slope →10m.
- **Seasons:** pre Mar-May, monsoon Jun-Sep, post Oct-Dec.
- **Grid:** EPSG:32644 Hyd. Two-state: per UTM 43N/44N/45N, mosaic EPSG:4326 display.
- **Feature stack/season:** 14 optical +4 SAR +2 terrain =20. ×3 seasons ~60 for classical. Deep: 20/season as time dim or stacked.
- **Export:** COG int16 scale factors, 256 internal tiling, STAC.
- **Two-state scale [A]:** 10m two states ~2.75B px/band → full 60-ch float = hundreds GB. So classical inside EE (ee.Classifier.smileRandomForest); deep streams per tile writes uint8 only.

### 6.3 Labelling
- Test points Hyd ~2,000 stratified — final eval never train — photo-interp HR basemap+S2, written protocol, 10% double ≥85%.
- Training points Hyd ~1,500 more — classical — same protocol different blocks.
- Local patches ~100 256×256 — fine-tune deep/FM — QGIS polygon rasterised.
- DW training tiles thousands 510×510 — pretrain seg — PANGAEA download, remap 9→6, pull matching S2 (+nearest S1) from EE.
- Change sample ~400 stratified change/no-change — change acc — label 2019+2025 class.
- **Hyd rules:** granite vs rooftop (texture+context HR), seasonal tanks (majority state/season), fallow vs bare (multi-season NDVI).

### 6.4 Model suite
- M1 RF optical — optical baseline.
- M2 RF SAR — SAR baseline.
- M3 RF/LightGBM all — early pixel — classical fusion.
- M4 M1+M2 prob avg/stacking — decision — cheap comparison.
- M5 RF AlphaEarth 64-d (pre-fused Google) — foundation-feature baseline.
- M6 U-Net ResNet-34 all stacked — early — deep fusion.
- M7 Dual-branch U-Net optical+SAR — feature-level — deep late.
- M8 TerraMind-base+UPerNet S2,S1,DEM — native multimodal — main FM.
- M9 DOFA+decoder S2+S1 — wavelength-conditioned — second FM.
- B1-B2 DW, WorldCover — external benchmarks.

**Training:**
- Classical (M1-M5): sklearn/LightGBM, RF 500 trees tuned max_features/min_samples_leaf via inner spatial CV, point table, CPU Colab.
- Deep (M6-M9): Lightning, smp, TorchGeo, TerraTorch, AdamW 1e-4 enc /1e-3 dec cosine 50-100ep batch 8-16 mixed precision, Weighted CE+Dice, aug flips/90°/brightness jitter optical-only/random modality dropout (simulates cloud), Stage1 pretrain DW Stage2 fine-tune Hyd, FM frozen→full, Colab/Kaggle T4 16GB TerraMind-base batch ~8 224-256px.
- **Modality dropout [A]:** randomly blanks optical in training → fallback to SAR like monsoon deployment.

### 6.5 Inference
- Tiled 256px 32px overlap Gaussian blending (no seams). Outputs class argmax, confidence max prob, optionally per-class. Throughput [A]: Hyd ~36M px ~550 tiles few min/T4/season. Classical alt: train Python, re-train same config in EE RF, classify server-side large areas.

### 6.6 Change + indicators
- LC change: post-classification 2019 vs 2025 masked by confidence (both ≥thr) → less false change; matrix ha.
- Direct check: CVA on fused features second opinion; disagreements flagged.
- Veg: seasonal NDVI/EVI mean + Δ within veg classes only.
- Water: fused MNDWI>thr OR VV<thr (SAR covers monsoon); vs JRC GSW ref.
- Built-up: transitions into + NDBI + SAR-texture trend.
- **Degradation index (explicit project def, not universal):** degrading = tree→bare/shrub, water→bare/built-up, cropland/grass→bare, any natural→built-up. Weighted area per ward/mandal % unit area.

### 6.7 Evaluation protocol
1. Spatial blocks: size from semivariogram range (expect 3-10km Hyd), 5-fold GroupKFold, test blocks never train/tune.
2. Metrics: OA, macro-F1, per-class F1, kappa, confusion; mIoU patch deep; bootstrap 95% CIs all headlines.
3. Area: Olofsson error-adjusted + CIs from confusion + map proportions.
4. Benchmark: sample DW (same season mode) + WorldCover at test points, crosswalk to 6, same metrics.
5. Cloud-stress: (a) natural monsoon composites few clears, (b) synthetic mask 25/50/75/100% optical track macro-F1.
6. Ablations: optical→+SAR→+DEM→+texture/temporal; VV vs VV+VH; seasons count.
7. Generalisation (two-state): train Hyd, test held-out TG/AP, report drop.
8. Leakage proof: random vs spatial side-by-side.

### 6.8 Testing/QA
- Unit indices: NDVI/MNDWI vs hand calc. Alignment: S1/S2 share CRS/transform/shape. Leakage: no block both train+test. Crosswalk: every DW/WC → exactly one of 6. Integration: tiny AOI e2e in CI cached data. API contracts/limit/errors. Regression registered metrics don’t silently change. Visual QA checklist (seams/edges/misclass) before release.

### 6.9 MLOps/repro
- DVC: ingest→features→sample→train→infer→analytics→evaluate.
- MLflow: log config/data version/git commit/metrics/artefacts; promote best to Production; API reads production.
- Seeds fixed, env pinned (requirements.lock, Docker).
- Model card per production: data, classes, metrics CIs, benchmarks, failures (rock vs rooftop), use/limits.
- Drift: new season predicted distribution + mean confidence vs training; flag shifts.

---
## 7. Plan of Action

### 7.1 Roles
- **A Data & Platform:** EE pipeline, STAC/COG, DVC, inference, Docker/deploy. Secondary TiTiler/PostGIS.
- **B Ground Truth/Eval/Research:** protocol, QA, DW tiles, eval framework, area est, literature, report. Secondary model cards.
- **C ML & App:** classical+deep, FM fine-tune, MLflow, FastAPI, dashboard. Secondary change analytics.
- Rhythm: 2×30min syncs/week, 1 guide check-in, GitHub board 1 issue/task.

### 7.2 Milestones (20w)
- M0 Foundations 1-2: scope, repo CI configs, EE access, S1/S2 availability report, 15-paper matrix.
- M1 Data platform 3-5: seasonal S1/S2/DEM 2019+2025 COGs in STAC, alignment tests, cloud stats.
- M2 Ground truth 4-7: test+train pts agreement ≥85%, blocks, DW remapped+paired, 100 patches.
- M3 Classical 6-8: M1-M5 spatial CV, benchmark DW/WC, random-vs-spatial.
- M4 Deep+FM 8-13: M6-M9 pretrain→fine-tune, cloud-stress, ablations, best in MLflow.
- M5 Analytics 12-15: change, matrix, indicators, degradation, areas CIs, change acc.
- M6 Product 13-17: FastAPI/TiTiler/PostGIS/dashboard Docker live e2e pass.
- M7 Scale-up/hardening 16-18: two-state annual (classical EE + deep sample districts), generalisation, cards.
- M8 Final 18-20: report, paper draft, demo video, rehearsed viva, tagged release.
- Overlap intentional. If short calendar cut M7 first. If long add EOS-04 (FR-17).

### 7.3 Week-by-week
- 1: A EE reg repo CI Hydra AOI; B verify refs Zotero matrix; C Colab DW/WC AOI crosswalk.
- 2: A S1/S2 availability 2019/2025 orbit report; B protocol v1 stratified design; C scope package skeleton MLflow.
- 3: A S2 Cloud Score+ medians indices; B sprint1 test pts; C sampling histograms.
- 4: A S1 edge Refined Lee VV/VH/ratio/std DEM; B sprint2 10% double agreement; C spatial blocks semivariogram leakage test.
- 5: A co-reg COG STAC DVC; B finalise test DW tiles; C M1-M3 spatial CV.
- 6: A pair DW+S2/S1 chip gen; B training pts separate blocks patch start; C M4 decision M5 AlphaEarth metrics bootstrap.
- 7: A chip QA loaders 20-ch; B benchmark DW/WC test pts; C random-vs-spatial v1 table.
- 8: A tiled inference overlap blend; B finish 100 patches area-est; C U-Net M6 pretrain DW.
- 9: A EE classifier large areas; B cloud-stress framework natural+synthetic; C M6 fine-tune Hyd M7 dual.
- 10: A inference all seasons/years best; B cloud-stress M1-M7; C M7 dropout M8 TerraTorch setup.
- 11: A PostGIS schema SoI boundaries; B ablations v2 table; C M8 frozen→full.
- 12: A TiTiler COGs colormaps; B change sample ~400; C M9 DOFA pick best register.
- 13: A Compose API/TiTiler/PostGIS/web; B change acc matrix val; C change confidence-masked.
- 14: A load products/stats PostGIS; B indicator val water vs JRC GSW NDVI; C indicators+degradation.
- 15: A GH Actions build/push staging; B areas CIs all; C FastAPI /products /stats /analyze /change /compare.
- 16: A prod deploy uptime logs; B model+dataset cards; C React+MapLibre layers swipe polygon.
- 17: A two-state EE annual classical; B generalisation sample TG/AP; C comparison downloads PDF.
- 18: A deep 2-3 districts drift; B generalisation results final tables; C e2e perf tuning.
- 19: A release tag README rerun; B report paper draft; C demo video slides.
- 20: Buffer/fixes; viva prep all.

### 7.4 Definition of Done
- All Must FR done+demoed. NFR-01..03 evidenced numbers. M1-M9+B1-B2 table bootstrap CIs spatial CV. Cloud/ablation/random-vs-spatial. Change/matrix/indicators/degradation+def. Areas CIs. Dashboard+API URL. CI green unit/integration/leakage. Cards+licence register. Rerun ±0.5pp. Report/paper/video/viva.

---
## 8. Risk Register
- Label quality/time High/VHigh → protocol double DW pretrain keep local small.
- Leakage High/High → block CV day1 auto test.
- S1 gaps Med/Med → week2 check years 2019/2025.
- GPU limits Med/Med → TerraMind-base mixed precision Kaggle second checkpointing.
- EE quota Med/Med → export once cache COGs batches.
- FM harder Med/Med → U-Net M6/M7 fallback TerraTorch tutorials.
- Rock vs built-up High/Med → labelling rule SAR texture known failure.
- Fusion no gain Med/Med → monsoon/cloud tests where SAR matters report honestly.
- Hosting free limits Med/Low → small demo COGs static fallback college VM.
- Scope creep High/High → requirements table contract; new→Could.
- Wrong boundaries Med/Med → SoI/state official only.
- Licence misses Low/Med → register attribution footer+report.

---
## 9. Cost/Resources (all free, check current limits — they change)
- Data/proc: GEE noncommercial — quota-limited.
- GPU: Colab/Kaggle weekly — free.
- Storage: Drive/R2 free/HF datasets — within limits.
- Tracking: MLflow local/DagsHub — free.
- Hosting: HF Spaces Docker/Render/college VM — free.
- CI: GH Actions public — free.
- Labelling: QGIS+EE basemap — team time.

---
## 10. Standout (why this wins)
1. Controlled fusion: 5 configs + FMs one dataset same spatial val.
2. Honest val: block CV bootstrap random-vs-spatial.
3. Local vs global: fair DW/WorldCover Deccan benchmark.
4. Cloud robustness: natural + synthetic + dropout training.
5. FM on Indian data: TerraMind Hyd fine-tune.
6. Defined degradation: explicit per-ward/mandal.
7. Working product: API/dashboard/reports not notebooks.
8. Built to scale: TG+AP via config + generalisation test.

---
## 11. Viva Prep (10 core)
1. What does SAR add that optical can't, where did results show?
2. Which fusion best, why?
3. How avoid leakage, how big difference?
4. Why DW-train + DW-benchmark not circular? Ans: final test own independent Hyd points, DW only pretrain.
5. Degradation def + limits?
6. How confident area estimates?
7. Accuracy outside Hyd?
8. Known failure modes production model?
9. How add EOS-04/new season?
10. What does dynamic mean in system?

---
## 12. Future Extensions
- EOS-04 to fill 2022-2024 (FR-17). Monthly once S1C/1D steady. Crop-type agri. Flood SAR monsoon. NL query (“how much tree cover did Medchal lose?”) over stats API.

---
## 13. Glossary
SAR: active microwave through cloud. VV/VH: vertical send/receive vs vertical send horizontal receive. Speckle: SAR grain noise. Composite: median etc over period. COG: Cloud-Optimised GeoTIFF streams web. STAC: standard EO metadata. Spatial block CV: train/test whole geo blocks. Macro-F1: avg F1 rare=equal. FM: pretrained huge unlabelled adapted few labels.

---
## 14. Sources
**Abstract refs (verified):** Moharrami 2024 https://www.mdpi.com/2072-4292/16/9/1566, Seagrass Sci Rep 14:8360 2024 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11006664/, Almeida 2024 Applied Geog 165:103249 DOI 10.1016/j.apgeog.2024.103249.
**Products/val:** DW https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9184477/, Venter 2022 RS 14(16):4101 DOI 10.3390/rs14164101, Malawi https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13407716/, Kattenborn 2022 ISPRS Open J spatial vs random.
**Fusion/datasets/models:** W-Net S1 https://pmc.ncbi.nlm.nih.gov/articles/PMC7288459, Multi-source S1+S2 ISPRS 2019 https://agritrop.cirad.fr/597770, SAR-optical early vs late RS 17(7):1298 https://www.mdpi.com/2072-4292/17/7/1298/htm, SEN12MS https://arxiv.org/abs/1906.07789, Prithvi-EO-2.0 https://arxiv.org/abs/2412.02732, DOFA https://arxiv.org/abs/2403.15356, TerraMind https://arxiv.org/abs/2504.11171 + https://research.ibm.com/blog/thinking-in-modalities-terramind + https://sentiwiki.copernicus.eu/web/create-ai-applications, AlphaEarth https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/, DW train https://doi.pangaea.de/10.1594/PANGAEA.933475, DW test https://zenodo.org/records/4766451.
**Indian:** Bhoonidhi 2025 https://www.nrsc.gov.in/nrscnew/assets/pdf/brochures/Bhoonidhi_Brochure_2025.pdf, EOS-04 https://www.nrsc.gov.in/sites/default/files/pdf/EOS_04_writeup_modified.pdf.
**Not yet verified — check before citing:** BigEarthNet, OSCD, LoveDA, Esri details, JRC GSW, Copernicus DEM licence, free-tier limits, AlphaEarth 2025 coverage, 2026 slides refs (Krejcar & Namazi; Nigar et al.).

---
## Appendix: Machine-readable defaults (for configs)
```yaml
aoi_hyderabad: {size_km: 60x60, area_km2: 3600, crs: EPSG:32644, display_crs: EPSG:4326}
years_primary: [2019, 2025]  # 2019=S1A+B, 2025=S1C (+S1D from 2026-04-17)
seasons: {pre: Mar-May, monsoon: Jun-Sep, post: Oct-Dec}
classes_default6: [water, tree_cover, cropland, built_up, bare_rocky, grass_shrub]
s2: {source: COPERNICUS/S2_SR_HARMONIZED, cloud: GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED, thr: 0.6, bands: [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12], indices: [NDVI,EVI,MNDWI,NDBI]}
s1: {source: COPERNICUS/S1_GRD, mode: IW, pol: [VV,VH], orbit: TBD_week2, speckle: RefinedLee_fallback_focal_median, feats: [VV_dB,VH_dB,VV-VH_ratio,VH_std_seasonal]}
dem: {source: Copernicus_GLO30, derived: [elevation,slope]}
stack_per_season: 20  # 14+4+2, x3=60 classical
export: {format: COG_int16_scaled, tile: 256, catalog: STAC}
labels: {test_pts: 2000, train_pts: 1500, patches: 100x256x256, change_pts: 400, agreement: 0.85}
inference: {tile: 256, overlap: 32, blend: gaussian, device: T4}
api_limits: {polygon_km2: 100, tiles_p95_ms: 500, stats_s: 3}
```
