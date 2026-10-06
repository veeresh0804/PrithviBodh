# Issue drafts — repo review findings (gh CLI unavailable locally, file for manual filing)

File these at https://github.com/veeresh0804/PrithviBodh/issues (one per finding).
Suggested labels noted per issue. Never paste secret values into issues.

## 1. [security] ADMIN_API_KEY fails open + default Postgres password [FIXED, needs review]
- `api/auth.py:18` defaulted to `"changeme-local-only"`; `docker-compose.yml`
  defaulted `ADMIN_API_KEY` the same way and set `POSTGRES_PASSWORD: geoeco`
  with Postgres bound to all interfaces.
- Fix: `expected_admin_key()` raises `RuntimeError` when unset; compose requires
  `${ADMIN_API_KEY:?…}` / `${POSTGRES_PASSWORD:?…}` and binds `127.0.0.1:5432`.
- Human action: rotate any exposed key out-of-band; set real values in deploy env.

## 2. [security] Add gitleaks secret-scan to CI
- No secret scanning. Add a `gitleaks` step (pinned version) to `.github/workflows/ci.yml`.
- Note: a Kaggle API token was pasted in chat during setup — already rotated
  (confirm), and `~/.kaggle/` lives outside the repo.

## 3. [data-quality] Sliver cells in grid design
- `build_cells` ceil produces a 42 m northern sliver row assignable for points.
- Fix: cells under `min_cell_area_fraction` (0.5, config) are never assigned/sampled;
  test fails if any assigned cell is below it. Seed re-accepted after the fix.

## 4. [data-quality] Seed 42 rejected → seed 3 accepted (grid)
- Checker gates (distance gap ≤5 km, core both sides, band span ≥4/5):
  seed 42 REJECT (gap 8.39, train core 0); seed 3 ACCEPT (gap 3.33,
  core 151/48). Canonical skeleton regenerated under seed 3.
- Human action: guide sign-off on the regenerated skeleton before labelling.

## 5. [data-quality] Raise min_core_points_per_side 1 → 10
- Existence floor (1 pt) lets a side's urban signal be noise in per-class F1 CIs.
  10 ≈ 0.5% of test / 0.67% of train — presence with mass. Seed 3 passes (151/48).

## 6. [docs] labelling.yaml comments overstated pre-registration
- "Pre-registered / fixed BEFORE" replaced with honest pilot-scan calibration
  note (thresholds set from 8-draw pilot distribution, frozen before selection).

## 7. [tests] CI must test the real spatial design (not missing data files)
- Buffer tests read gitignored `data/` files → CI failed with FileNotFoundError.
- Fix: tests generate skeletons in `tmp_path` via real grid code; shared
  haversine/proxy math vendored from gitignored `collect.py` into
  `geoeco/labels/geo_stats.py`. See AGENTS.md rule: CI tests must never read `data/`.
