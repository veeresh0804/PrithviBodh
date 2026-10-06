# Kaggle Runbook — M1-M3 Classical (PrithviBodh)
All secrets stay in C:\Users\manoh\.kaggle\ (never in repo). Use `python -m kaggle` (exe not on PATH).

## 1. Version real labels (once QGIS done)
Local file must be `data/labels/hyd_points.geojson` (NOT synthetic).
```powershell
python scripts/make_dummy_labels.py  # smoke only, skip for real run
python -m kaggle datasets version -p data/labels --dir-mode zip -m "v2 real Hyd labels"
# or via site: qgis-hyd dataset -> New Version -> upload hyd_points.geojson
```

## 2. New Notebook (site)
GPU T4, Internet ON, Add input: vennamanoharaveeresh/qgis-hyd (v2).

## 3. Cells (paste in order)
```bash
!git clone https://github.com/veeresh0804/PrithviBodh.git
%cd PrithviBodh
!pip install -q -r requirements.txt
```
```python
import json, geopandas as gpd
pts = gpd.read_file('/kaggle/input/qgis-hyd/hyd_points.geojson')
assert 'synthetic' not in pts.columns or not pts.get('synthetic', False).any(), 'STOP: synthetic labels, upload v2 real'
print(len(pts), pts['label_name'].value_counts().to_dict())
```
```python
# M1-M3 spatial CV (GroupKFold blocks) -> mlruns/ + docs/results_table_v1.md
!python -m geoeco.train.train_classical --config configs/model/rf.yaml --aoi configs/aoi/hyderabad.yaml
```
Save Output -> Version dataset for demo/API.

Limits: 9h/run, 30h/week. Checkpoint often. Classical first, M6 UNet only after M1-M3 table filled.
