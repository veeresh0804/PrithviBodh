# Earth Engine Export Suite — Stage 1 (M1)

Config-driven pipeline that assembles and exports the Hyderabad multi-sensor
study layers as cloud-optimised GeoTIFFs (COG) on the AOI grid
(`EPSG:32644`, 10 m) — **nothing is hard-coded**: AOI, years, seasons,
thresholds and sources all come from `configs/aoi/hyderabad.yaml` +
`configs/data/sentinel.yaml`.

A human must only authenticate once, copy-paste the commands below, and
(optionally) verify the 13 exported COG files. Every offline-checkable step
is covered by `tests/test_gee_alignment.py` (synthetic, no network).

## Scripts

| Script | Purpose | EE auth? | Exit codes |
|---|---|---|---|
| `ee_common.py` | Shared: config loading, UTM grid, season windows, export submission | import-only | n/a |
| `export_s2_composites.py` | Seasonal S2 composites + indices + `valid_count` → 6 COG exports | `--submit` only | 0 plan / 2 no-creds-config / 1 EE fail |
| `export_s1_composites.py` | Seasonal S1 features (edge mask + speckle filter) → 6 COG exports | `--submit` only | 0 / 2 (incl. TBD orbit) / 1 |
| `export_dem.py` | Copernicus DEM GLO-30 elevation + slope → 1 COG export | `--submit` only | 0 / 2 / 1 |
| `availability_report.py` | Live scene counts, CS+ parity, S1 platform/orbit stats → JSON + markdown | yes (always) | 0 / 2 `/ 1` |
| `dryrun_one_tile.py` | Sample one 10 m tile per sensor, print per-band SHA-256 fingerprints | yes (always) | 0 / 2 / 1 |

Exit codes: **0** = success/plan, **1** = live EE query/export failure,
**2** = no credentials / bad config / auth needed. Missing credentials always
fail loudly with `python -m ee.cli.eecli authenticate` and write nothing.

## Human command sequence (PowerShell)

### 1. Authenticate (one-time)

```powershell
python -m ee.cli.eecli authenticate
```

> The `earthengine` console script is **not** on PATH on this machine and its
> CLI has no script-run subcommand anyway — use `python -m ee.cli.eecli`
> (subcommands: `authenticate`, `task list`, ...). Never print credentials.

### 2. Availability report (picks the S1 orbit)

```powershell
python scripts/ee/availability_report.py
```

Writes `docs/ee_availability/availability_report.json` +
`availability_report.md` with real scene counts per config window, Cloud
Score+ join parity, S1 platforms/relative orbits, and an orbit
recommendation (criterion printed; a tie reports `NONE`, never a guessed
orbit). Exit 2 without authentication, writing nothing.

### 3. Set the S1 orbit

Copy the recommended orbit into `configs/data/sentinel.yaml`:

```yaml
sentinel1:
  orbit_pass: "ASCENDING"   # from docs/ee_availability/availability_report.md
```

(Or pass `--orbit ASCENDING|DESCENDING` to the S1 commands below — the orbit
is never guessed.)

### 4. Dry-run plans (no auth needed)

```powershell
python scripts/ee/export_s2_composites.py
python scripts/ee/export_s1_composites.py
python scripts/ee/export_dem.py
```

Each prints the JSON plan with `"exported": false` — verify windows,
filenames and grid before spending any quota.

### 5. One-tile fingerprint dry-run (needs auth)

```powershell
python scripts/ee/dryrun_one_tile.py --year 2019 --season pre
```

Samples `export.internal_tiling`-sized (256 px) 10 m tiles on the **exact
export grid** and prints per-band SHA-256 of the raw band bytes. These are
pipeline fingerprints (reproducibility), **not** COG checksums — no file is
written. Re-run later to prove the pipeline is unchanged. Optional
`--export` also submits the full-size export tasks (same as step 6).

### 6. Submit the 13 export tasks (needs auth)

```powershell
python scripts/ee/export_s2_composites.py --submit
python scripts/ee/export_s1_composites.py --submit          # reads config orbit
python scripts/ee/export_dem.py --submit
```

Each queues Earth Engine tasks (Google Drive root by default; `--drive-folder`
or `--bucket` to redirect). Output prints task ids — files exist only after
the tasks finish.

### 7. Monitor until all COMPLETED

```powershell
python -m ee.cli.eecli task list
```

### 8. Verify the COGs locally

The delivered filenames (13 total) served after task completion:

```
hyderabad_s2_2019_pre.tif        hyderabad_s2_2019_monsoon.tif
hyderabad_s2_2019_post.tif       hyderabad_s2_2025_pre.tif
hyderabad_s2_2025_monsoon.tif    hyderabad_s2_2025_post.tif
hyderabad_s1_ASCENDING_2019_pre.tif    hyderabad_s1_ASCENDING_2019_monsoon.tif
hyderabad_s1_ASCENDING_2019_post.tif   hyderabad_s1_ASCENDING_2025_pre.tif
hyderabad_s1_ASCENDING_2025_monsoon.tif  hyderabad_s1_ASCENDING_2025_post.tif
hyderabad_copernicus_dem_glo30.tif
```

(S1 filenames embed the chosen orbit; DEM id maps to `COPERNICUS/DEM/GLO30`
— see `docs/stage1_ee_setup.md`.)

```powershell
Get-FileHash -Algorithm SHA256 hyderabad_s2_2019_pre.tif
```

The hash of the COG is computed *after* download; it differs from the step-5
fingerprint by design (headers/compression).

## What only a real EE login can verify

- Real scene counts, cloudy-sky statistics, CS+ parity, orbit recommendation
- The submitted tasks actually complete and files land in Drive/Cloud Storage
- Final COG bytes/checksums and band alignment on disk
- S1 orbit effects on coverage

Everything else (plans, config alignment, formulas, exit-2 no-auth behaviour)
is verified offline by `tests/test_gee_alignment.py`.