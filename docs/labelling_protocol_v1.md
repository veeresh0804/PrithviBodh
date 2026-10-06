# Labelling Protocol v1 — Hyderabad (M2)
Stratified by 6 classes: water, tree_cover, cropland, built_up, bare_rocky, grass_shrub.

## Sampling
- Test: ~2000 points, stratified, in blocks NEVER used for train (see `configs/eval/spatial_cv.yaml`, 3-10km blocks, 5-fold GroupKFold).
- Train: ~1500 more points, different blocks.
- Change: ~400 points stratified change/no-change, label 2019 + 2025 class.
- Patches: ~100 256x256 polygons in QGIS, rasterised, for M6-M9 fine-tune.

## Source
HR basemap (Google/Esri in QGIS) + S2 median (per `configs/data/sentinel.yaml`) + S1 VV/VH for water/built-up sanity.

## Hyderabad rules (enforced in `geoeco/labels/protocol.py`)
- R1 granite vs rooftop: check texture + context in HR; rooftops rectangular + road adjacency, outcrops irregular + no shadow grid.
- R2 seasonal tanks: majority state in season; if vegetated >50% of season → vegetated class, else water.
- R3 fallow vs bare: multi-season NDVI — peak NDVI >0.35 → cropland (fallow), else bare_rocky.

## QA
10% double-labelled, agreement ≥85% (`geoeco/labels/agreement.py`). Kappa reported. Disagreements adjudicated by guide.
Export: `data/labels/hyd_points.geojson` (points + label + block_id + season + year), patches → `data/labels/patches/`.
