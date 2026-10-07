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

## 4. Post-labelling command sequence (M1-M5 classical -> M6 UNet, the day labels land)
Dataset input: `vennamanoharaveeresh/qgis-hyd` **v2** (real QGIS labels only).
Mounts at `/kaggle/input/qgis-hyd/hyd_points.geojson`. Re-run the §3 guard cell first
— it aborts on any `synthetic` column/row, so smoke tables can never leak into the real run.

> CLI correction: the §3 cell above (`--config ... --aoi ...`) predates the current
> `geoeco/train/train_classical.py` argparse, which requires `--points` (Parquet point
> table) + `--model` (a model ID, not a yaml path). Use the sequence below, which matches
> the real CLIs (also mirrored in `dvc.yaml` stage `train`).

```bash
# 0. Point table + chips must exist first (from the geoeco/ stream sampling CLI):
#      data/processed/points.parquet   (label + block_id + feature cols per
#                                       geoeco/models/classical.py:feature_columns)
#      data/processed/chips/           (img_*.pt / mask_*.pt pairs for train_seg.py)
#    NOTE 2026-10-07: geoeco/features/sampling.py is still library-only (no CLI;
#    dvc stage `sample` is a TODO-M1 no-op). If the sampling CLI has not landed when
#    labels land, build the table via the notebook-02 recipe (feature_columns + block_id)
#    but with REAL features/labels, or ask the geoeco/ stream owner. Do NOT train on
#    the smoke parquet (it carries synthetic=True for exactly this reason).
```

```bash
# 1. M1-M5 classical on Kaggle CPU (each ~minutes on the real point table; seed 42 fixed):
for M in M1 M2 M3 M3-lgbm M5; do
  python -m geoeco.train.train_classical --points data/processed/points.parquet --model $M --seed 42
done
# M1 = optical RF, M2 = SAR RF, M3 = all-60ch RF, M3-lgbm = LightGBM variant, M5 = AlphaEarth-64d RF.
# M3-lgbm needs lightgbm installed (in requirements.txt).
```

```bash
# 2. M6 UNet on Kaggle GPU T4 (needs GPU notebook; CPU will crawl):
#    Hyperparams below are FROM configs/model/unet.yaml — do not invent others:
#      epochs [50, 100] = pretrain DW -> fine-tune Hyd; batch [8, 16] (T4 16GB:
#      8 at 256px, 16 at 224px); AdamW lr_enc 1e-4 / lr_dec 1e-3; weighted_ce+dice.
python -m geoeco.train.train_seg --stage dw_pretrain --data data/processed/chips/ --model M6 --epochs 50 --batch 8
python -m geoeco.train.train_seg --stage hyd_finetune --data data/processed/chips/ --model M6 --epochs 100 --batch 8 --ckpt <dw_pretrain.ckpt>
# 224px variant if OOM: --batch 16 still fits T4 16GB per unet.yaml.
```

mlruns/ outputs: local `./mlruns/` in the Kaggle working dir (notebooks use
`/kaggle/working/mlruns`). After each stage: Save Output -> Version dataset for demo/API.
Classical metrics also feed `docs/results_table_v1.md` (M1-M3 headline table gate).
