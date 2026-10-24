# PrithviBodh

**AI-Based Dynamic Geo-Ecosystem Assessment Using Multi-Sensor Remote Sensing**
Batch 58 · CSE (AI&ML) · CMR College of Engineering & Technology · Guide: Ms. A. Spandana

Land-cover, change and ecosystem indicators for Hyderabad and its surroundings (~60 x 60 km),
built in two steps:

1. **Existing solutions** - reproduce Dynamic World, ESA WorldCover, Esri Land Cover and
   single-sensor Random Forests, and score them on our own test points.
2. **Hybrid architecture** - fuse Sentinel-1, Sentinel-2 (with texture), Landsat LST, MODIS and DEM, train
   RF, SVM, CNN, Transformer and U-Net, and combine them by weighted voting.
3. **Assessment** - change and vegetation condition, flood / drought / deforestation risk, a 2031
   land-cover prediction and a ranked list of priority areas with suggested actions.

See [`docs/architecture.md`](docs/architecture.md) for both architectures.

## Free tools only

Google Earth Engine (noncommercial) · Google Colab / Kaggle · Google Drive · GitHub · QGIS · Streamlit.
No Docker, no paid cloud, no credit card.

## How to run (in order)

| Notebook | What it does | Who |
|---|---|---|
| `00_setup_and_data_check` | Sentinel-1/2 availability for 2019-2025; verify S1 descending orbit | Data |
| `01_labelling` | Candidate points -> label in QGIS -> validate + agreement check | All three |
| `02_existing_solutions` | Step 1: Baselines E1-E5 on the spatial test points | Evaluation |
| `03_hybrid_model` | Step 2: H1-H4/H5 + hybrid ensemble, comparative benchmark table | ML |
| `04_maps_change_indicators` | Annual maps 2019-2025 (D7), change map + matrix, multi-year persistence | Data + ML |
| `05_risk_prediction_decision` | Flood / drought / deforestation risk, 2031 prediction (validated on 2025), priority cells | Data + ML |

Open any notebook in Colab: `File -> Open notebook -> GitHub -> veeresh0804/PrithviBodh`.
Set your Earth Engine project ID in the setup cell.

**Benchmark Results (Evaluated on Identical 5 km Test Points)**:
* **H\* Hybrid Ensemble**: **OA 95.8%**, **Macro-F1 0.963**, **mIoU 0.930**, **Kappa 0.948**
* **Best Baseline (E4 RF Optical)**: OA 88.5%, Macro-F1 0.899
* **Paired Test Gain**: $\Delta \text{Macro-F1 } 95\% \text{ CI} = [0.0304, 0.1041] > 0$ (Statistically significant improvement)
* **Multi-Year Persistence Filter**: Suppresses 21.4% (58,637 ha) of transient single-year classification noise.

**Interactive Web Demo App**
```bash
pip install -r app/requirements.txt
python -m streamlit run app/streamlit_app.py
```
View locally at: `http://localhost:8501`.
Free cloud deployment: push to GitHub, then deploy `app/streamlit_app.py` on share.streamlit.io.

**Tests**: `python -m pytest -q` (all 20 tests pass)

## Project structure
```
config.yaml                 every setting (AOI, years, seasons, classes, models)
prithvibodh/
  gee_data.py               Earth Engine: sensors, stack, sampling, exports
  baselines.py              existing products + class crosswalks
  features.py               feature names, groups, scale factors
  labels.py                 load/validate labels, agreement
  spatial_cv.py             spatial blocks, leakage check
  classical.py              Random Forest, SVM
  deep.py                   CNN, Transformer, U-Net (PyTorch)
  hybrid.py                 weighted soft voting
  experiment.py             runs Step 1 and Step 2 on the same test points
  evaluate.py               metrics, bootstrap CIs, confusion matrix
  inference.py              full-map hybrid prediction
  change.py, indicators.py  change map + matrix, degradation, areas, NDVI change
  risk.py                   flood, drought, deforestation risk indices
  prediction.py             CA-Markov future land cover
  decision.py               priority table with suggested actions
  rasters.py                read / write / match raster grids
notebooks/                  00-05, run in Colab
app/streamlit_app.py        demo
docs/                       architecture, labelling protocol
tests/                      unit tests
```

## Things to know
* **Storage:** one exported stack (70 bands, 10 m) is about 5 GB, so two years use most of the free 15 GB Drive.
  If space is tight, export the stack at `scale=20` (about 1.3 GB). The indicator stacks are small (8 bands).
* **Years:** 2019 and 2025 avoid the Sentinel-1 gap (Dec 2021 - early 2025).
* **Monsoon gaps:** cloudy seasons are filled from the annual composite so no point is lost;
  the clear-observation count is reported separately.
* **2019 map:** the model is trained on 2025 labels and applied to 2019. Check some 2019 points by eye
  before trusting change numbers.
* **Risk and prediction:** screening indices and a CA-Markov projection, not calibrated hazard models or
  validated forecasts. Say this in the report.
* **Metrics reported:** OA, precision, recall, macro-F1, mIoU, kappa (+ per-class table), with 95% CIs.
* **Data sources to cite:** Sentinel-1/2 (Copernicus), Landsat 8/9 (USGS), MODIS MOD13Q1 (NASA),
  Copernicus DEM GLO-30, Dynamic World (CC BY 4.0), ESA WorldCover, Esri Land Cover.
