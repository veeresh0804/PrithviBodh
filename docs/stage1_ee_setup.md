# Stage 1 — Earth Engine Data Platform Setup & Contradictions

Stage 1 (M1) of the Hyderabad multi-sensor study builds the data platform:
seasonal Sentinel-2 composites, Sentinel-1 features, and Copernicus DEM as
co-registered COGs on the AOI grid (EPSG:32644, 10 m). This page covers
setup, verification, the exact human steps, and the documented deltas
between the brief, the configs, and verified Earth Engine facts.

Everything here was either verified on this machine (offline) or is
explicitly marked as unverifiable without an Earth Engine login — no claim
about exported files or scene counts is made anywhere in this repo.

## 1. Environment & authentication

```powershell
pip install earthengine-api        # present here: 1.7.47
python -m ee.cli.eecli authenticate
```

Notes (verified on this machine):

- The `earthengine` console script is **not on PATH**; use
  `python -m ee.cli.eecli`. Its subcommands cover `authenticate` and
  `task list` **but have no script-run subcommand** — exports are submitted
  as EE tasks (`Export.image.toDrive`/`toCloudStorage`) by the Python
  scripts, not executed through a CLI runner.
- Credentials live under `%USERPROFILE%\.config\earthengine\credentials`;
  never print them.

## 2. What runs where

| Layer | Deliverable |
|---|---|
| Shared helpers | `scripts/ee/ee_common.py` — sys.path bootstrap, configs, UTM grid, season windows, COG export submission (`cloudOptimized: true`, exact `crsTransform`) |
| S2 composites | `scripts/ee/export_s2_composites.py` — CS+ masked median + indices + `valid_count`, 6 tasks |
| S1 features | `scripts/ee/export_s1_composites.py` — orbit-resolved, edge-masked, speckle-filtered `VV/VH/VVVH/VH_std`, 6 tasks |
| DEM | `scripts/ee/export_dem.py` — GLO-30 mosaic → elevation/slope, 1 task |
| Availability | `scripts/ee/availability_report.py` — live counts/parity/orbit recommendation → `docs/ee_availability/` |
| Fingerprint | `scripts/ee/dryrun_one_tile.py` — one-tile SHA-256 per sensor on the export grid |
| EE index expressions | `geoeco/ingest/gee_indices.py` — formulas aligned to `geoeco/features/indices.py` (optical indices clamped to [-1, 1] like the NumPy reference) |
| Alignment tests | `tests/test_gee_alignment.py` — synthetic, no network |
| Human procedure | `scripts/ee/README.md` — exact command sequence |

Key semantics baked into the pipeline:

- Cloud Score+ `cs` is **1 = clear, 0 = occluded**: pixels are kept where
  `cs >= sentinel2.cloud_threshold` (config). The scene-level pre-filter
  `CLOUDY_PIXEL_PERCENTAGE < 80` mirrors the read-only
  `geoeco/ingest/gee_s2.py` exactly so report counts match export scenes.
- Seasons are built purely from `configs/data/sentinel.yaml` month ranges —
  they equal the canonical `geoeco/ingest/composites.py` contract
  (verified by tests for 2019 and 2025).
- The export grid is the AOI's `EPSG:32644` @ 10 m; EE `crsTransform`
  `[res, 0, x0, 0, -res, y1]` is order-equal to the GDAL geotransform
  `(x0, res, 0, y1, 0, -res)` (tested).

## 3. Verified without Earth Engine credentials

```powershell
python -m pytest tests/test_gee_alignment.py -q      # 38 passed
python -m flake8 --select=F scripts/ee/*.py geoeco/ingest/gee_indices.py tests/test_gee_alignment.py
python scripts/ee/availability_report.py --help      # rc 0 (and the other 4 scripts)
python scripts/ee/availability_report.py             # rc 2, no files written (no creds)
python scripts/ee/dryrun_one_tile.py                 # rc 2, nothing written (no creds)
python scripts/ee/export_*.py --submit               # rc 2 with `earthengine authenticate`
python scripts/ee/export_s1_composites.py            # rc 2: orbit TBD, never guessed
```

Plus the three export scripts' default plan mode: `exported: false`, 6+6+1
tasks, exact filenames.

## 4. Human steps only a real login can complete

1. `python -m ee.cli.eecli authenticate`
2. `python scripts/ee/availability_report.py` → review `docs/ee_availability/`
3. Set `sentinel1.orbit_pass` (or pass `--orbit`)
4. Dry-run plans (optional, offline)
5. `python scripts/ee/dryrun_one_tile.py` → fingerprint sanity
6. `python scripts/ee/export_*.py --submit` → 13 tasks
7. `python -m ee.cli.eecli task list` until COMPLETED
8. `Get-FileHash -Algorithm SHA256` on the downloaded COGs

Full sequence and filenames: `scripts/ee/README.md`.

## 5. Contradictions & deltas (reported, not silently patched)

1. **DEM id in the brief does not exist.** `COPERNICUS/DEM_GLO30` is not a
   catalog id; the verified catalog id is `COPERNICUS/DEM/GLO30` (mapped from
   config `dem.source: Copernicus_GLO30`). It is superseded by
   `GLO30_2024_1` — override with `--dem-asset` if the project prefers it.
2. **No `earthengine` CLI script-run.** The prompt's "runnable via
   earthengine CLI" is not available: `python -m ee.cli.eecli` only offers
   `authenticate` / `task list`. Submissions are Python-driven EE tasks.
3. **`COG_int16_scaled` has no scale factors.** `export.format` requests
   int16 scaling but no per-band factors exist in config; scripts export
   native float32 and say so loudly (`scaling_note`). Add `export.scales`
   to enable true int16 output; factors are never invented.
4. **`RefinedLee` has no GEE operator.** The config asks for Refined Lee;
   GEE has no native implementation, so the pipeline applies the documented
   in-EE focal-median approximation on linear power and prints an explicit
   note (`SPECKLE_REFINED_LEE_NOTE`). True Refined Lee requires
   SNAP/external pre-processing.
5. **S1 orbit is a blocker until week 2.** `sentinel1.orbit_pass` is
   `TBD_week2`; `export_s1` and `dryrun_one_tile` exit 2 with guidance until
   the availability report supplies an orbit.
6. **S1 feature-name mismatch (read-only files).** `geoeco/ingest/gee_s1.py`
   produces `VV/VH/VVVH/VH_std` while `models/classical.py` expects
   `VV_dB/VH_dB/VV_VH_ratio/VH_std`; config `sentinel1.features` lists the
   `_dB` names. Reported — not patched (out of scope here).
7. **Not-in-config constants.** The 80 % scene-cloud filter
   (`geoeco/ingest/gee_s2.py`), the IW angle window 31–46°, the 500 m edge
   buffer and the 70 m speckle kernel (docstrings in
   `geoeco/ingest/gee_s1.py`) are constants with configurable fallbacks only
   for buffer/kernel. They should be promoted to `configs/data/sentinel.yaml`.
8. **Superseded artefacts left untouched.** `scripts/check_ee_availability.py`
   and the `docs/availability_report.md` placeholder predate this suite and
   are not owned by it; the active report lives in `docs/ee_availability/`.