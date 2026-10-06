# DATA + Free-GPU Training Guide (Batch 58)

## 1. Best free site to train? Use Kaggle (main) + Colab (debug)
- **Kaggle (recommended main):** 30h free GPU/week (T4 x2 / P100), 9h max run, 20GB datasets + 20GB output, persistent. Best for M1-M7 + M8 TerraMind-base (batch 8, 224px fits T4 16GB). No random disconnect like Colab.
- **Colab Free:** T4 ~12h/day but idle disconnects, Drive needed for COGs. Good for quick debug + inference demo.
- **Earth Engine:** free noncommercial for S1/S2/DEM/AlphaEarth/WorldCover processing (quota-limited). Imagery NOT downloaded as bulk — export seasonal composites as COGs.
- **Hugging Face Datasets / R2:** free hosting for demo COGs for TiTiler.
- **Do NOT train full DW 5B-px locally.** Pretrain on subset, fine-tune on Hyderabad.

Verdict: **Train on Kaggle, debug on Colab, serve demo from HF.**

## 2. What you actually need (from PROJECT_CONTEXT §2.4/§6.3)
- `configs/data/sentinel.yaml`: years [2019,2025], seasons pre Mar-May / monsoon Jun-Sep / post Oct-Dec, S2 B2-B12 + NDVI/EVI/MNDWI/NDBI, S1 VV/VH + ratio + VH-std, DEM GLO-30.
- Public labels:
  - DW training (PANGAEA DOI 10.1594/PANGAEA.933475, CC BY-4.0): ~24k tiles 510x510. Imagery NOT included — pull S2 + nearest S1 from EE via `geoeco/labels/dw_tiles.py`.
  - DW expert test (Zenodo 4766451): global comparison only.
- Mandatory own labels (cannot download):
  - ~2000 Hyderabad test points (stratified, never train) + ~1500 train points (different blocks) — photo-interp HR basemap + S2, 10% double ≥85%.
  - ~100 patches 256x256 QGIS rasterised — fine-tune deep/FM.
  - ~400 change points (2019+2025 class) — change accuracy.
- S1/S2/DEM/AlphaEarth via EE (`earthengine authenticate`), not bulk download.

## 3. Minimal download (fits free tier)
```powershell
python scripts/download_datasets.py --out data --n_dw_test 5
# creates data/raw/dw_test/ + data/raw/pangaea/README.txt + data/labels/
```
- DW test sample: https://zenodo.org/records/4766451
- DW train manifest: https://doi.pangaea.de/10.1594/PANGAEA.933475
- DEM Hyderabad tile: AWS `s3://copernicus-dem-30m` (or EE `COPERNICUS/DEM/GLO-30`) — export via EE, don't scrape.
- SEN12MS (optional small pretrain alt): via TorchGeo `SEN12MS` auto-fetch, 256x256 S1/S2/MODIS labels.

## 4. Kaggle setup (5 min)
1. Kaggle → Create Dataset → upload `data/` zip (dw_test sample + your QGIS labels).
2. New Notebook → GPU T4 → Internet ON → Add dataset.
3. Install: `pip install -r requirements.txt` (torch + lightning + smp + torchgeo + terratorch stub).
4. Run:
```python
# classical first (CPU ok)
python -m geoeco.train.train_classical --config configs/model/rf.yaml --aoi configs/aoi/hyderabad.yaml
# deep (GPU)
python -m geoeco.train.train_seg --config configs/model/unet.yaml --pretrain dw --finetune data/labels/
```
5. Save `mlruns/` + best COG as Output → Version dataset for API/demo.

Colab same but mount Drive for COGs, timeout 12h → checkpoint often (`ModelCheckpoint` already in `train_seg.py`).

## 5. What NOT to do
- Don't download full 24k DW tiles locally (hundreds GB). Stream per-chip.
- Don't expect 2019-2024 S1 dense: S1B failed 2021-12-23, S1C regular 2025-03-26. Primary years 2019 + 2025 only.
- Don't train without spatial blocks (GroupKFold 5-fold, 3-10km) — random split inflates up to +28% (Kattenborn 2022).

## 6. Hand-labelling checklist (QGIS)
- Classes 6: water, tree_cover, cropland, built_up, bare_rocky, grass_shrub.
- Rules: granite vs rooftop (texture+context), tanks by majority state/season, fallow vs bare by multi-season NDVI.
- Export GeoJSON → `data/labels/hyd_points.geojson`, patches → `data/labels/patches/`.
