# PROGRESS — engineering log (resumable)

Convention: one row per stage. Status is one of `DONE`, `PARTIAL`, `BLOCKED-HUMAN`,
`BLOCKED-DECISION`, `IN PROGRESS`. Never mark DONE without evidence in the
"Verification" column. Claims tagged [V] verified, [K] standard knowledge, [A] our analysis.

Last updated: 2026-10-08 (orchestrator).

---

## Stage 0 — Carry-over (agent only) — RESOLVED (all deliverables in; see notes)

| Item | Status | Evidence |
|---|---|---|
| CI green on main | DONE [V] | Main run **#41** (`bba870b`): **Status Success** — `140 passed, 7 skipped` (the +4 vs #37 are the new config-tracking tests), lint "All checks passed!", gitleaks/install/labelling/leakage all success. Full history: #37 Success (a790187), #38 Success (d354fa0 docs-only), #39–#40 failed on ONE error (B904 in the new test file — fixed the same way CI demanded), #41 green. Root cause, corrected: BLE001 is ✅ default-enabled (ruff docs) — the #30 failures were NEW-CODE violations (E501×23, RUF002×2, B905×8, B007×2, RUF005, RUF043×6, E702, E306, BLE001×2, RUF100, B904), NOT toolchain drift; "59→413" withdrawn as unverified. Pin: ruff `==0.16.10` + explicit select `[E,F,B,UP,BLE,RUF,I]` (+documented E402 ignore, ISC out with measured reason). Prior green: #26 (`b25def1`). |
| Lint errors fixed | DONE [V] | Every CI-reported error fixed across runs #32–#41 (see CI-green row), plus one real test-collection bug unblocked by green lint (`@torch.no_grad()` on possibly-None torch → context manager inside guarded body; verified with-torch and torch-absent). Branch `fix/lint-pin` merged to main (fast-forward). |
| Issues from `docs/issues_draft.md` filed on GitHub | BLOCKED-HUMAN | **Ready-to-file copies exist**: `docs/github_issues/issue_01..07.md` + `file_issues.ps1` script. `gh` CLI not installed; no GITHUB_TOKEN. Human runs the .ps1 script to copy each issue body to clipboard, then pastes into https://github.com/veeresh0804/PrithviBodh/issues/new. Seed-3 wording corrected to canonical seed 6 in all 7 copies. |
| Gitleaks step passing | DONE [V] | Run #30 job step: "Gitleaks scan (secrets)" = success. |
| No default admin key anywhere | DONE [V] | `docker-compose.yml` uses `${ADMIN_API_KEY:?…}` (fail-closed); `api/auth.py` default removed. |
| Postgres bound to localhost | DONE [V] | `docker-compose.yml:11` → `127.0.0.1:5432:5432`. |
| Postgres default password fallback | FLAG | `api/db.py:101` still defaults `DATABASE_URL` to dev creds when env var unset (compose always sets it). Not an admin key; hardening deferred to Stage 6. |
| Linux/Colab-compatible lock file | DONE [V] | Stream C produced `requirements.linux.lock.txt` (293 pins), `requirements.colab.lock.txt` (296 pins with test extra), and `docs/lockfile_policy.md`. Both locks were **resolved for Linux/CPython 3.10 on a Windows host** with pip 26.2.1, marker-audited (DROPPED = pywin32, waitress; MISSING = none). **Not yet executed on real Linux/Colab** — pending one CI run and one Colab GPU session (see `docs/lockfile_policy.md` Status section, lines 165–177). |

Known environment incidents:

- **ruff binary blocked by Windows Application Control** (WinError 4551) on this
  dev machine, as are `pip.exe`/`pip` shims (use `python -m pip`). Local lint runs
  are impossible; **CI is the lint gate** until this is resolved. Recorded
  2026-10-08. Proxy used locally: `python -m flake8 --select=F` (pure Python).

---

## Stage 1 — Data platform (M1) — DONE (agent) / BLOCKED-HUMAN (export)

- **Stream B delivered**: 8 new/verified files:
  - `scripts/ee/ee_common.py` — bootstraps repo root, loads configs, resolves seed, builds seasons windows from config months only, projects AOI → EPSG:32644/10 m grid, `tile_region`, scaling provenance, `submit_image_export` (COG cloudOptimized: true, exact crsTransform, exists_on_disk: false)
  - `scripts/ee/export_s2_composites.py` — Seasonal S2: CS+ join (config threshold), resample, median composite + valid_count, reflectance /0.0001, indices NDVI/EVI/MNDWI/NDBI, 6 COG export tasks; plan mode (exported:false); `--submit` = tasks; exits 2 without EE creds; never fakes outputs
  - `scripts/ee/export_s1_composites.py` — Seasonal S1: orbit from config/`--orbit` (TBD → exit 2, never guessed), IW+VV/VH config filters, edge-noise mask, Refined Lee noted as focal-median approximation (no native EE operator), features VV_dB/VH_dB/VV_minus_VH_ratio/VH_std, 6 tasks; exits 2 without creds
  - `scripts/ee/export_dem.py` — DEM COPERNICUS/DEM/GLO30 mosaic + setDefaultProjection → elevation/slope, 1 task; `--dem-asset` override; exits 2 without creds
  - `scripts/ee/availability_report.py` — Auth-first live EE queries → per-window S2 scene counts + cloudy stats + CS+ join parity, S1 pass/platform/relative-orbit counts → orbit recommendation (criterion printed, ties = NONE) → JSON+MD into `docs/ee_availability/` (written only after every query succeeds); exits 2 `"earthengine not installed"` or `"needs earthengine authenticate"`; never fakes numbers
  - `scripts/ee/dryrun_one_tile.py` — With EE auth, exports ONE 10m tile per sensor and prints SHA-256 fingerprints; without auth, exits 2 with clear instructions. Never writes fake outputs.
  - `tests/test_gee_alignment.py` — **38 passed** (synthetic, no network, CRS EPSG:32644, transform/shape, index formulas, config seasons vs composite season_date_range, all five scripts --help rc 0, plan exported:false, S1 TBD-orbit rc 2, no-auth rc 2 writing nothing, fingerprint determinism). `python -m pytest tests/test_gee_alignment.py -q` → green; flake8 clean on all new .py files
  - `geoeco/ingest/gee_indices.py` — pre-existing (verified, not modified); pure-Python index formulas (NDVI/EVI/MNDWI/NDBI/VV-VH ratio) testable without EE import
- Human: `earthengine authenticate`, then `scripts/ee/` dry-run → availability report → set `sentinel1.orbit_pass` → dry-runs → `--submit` 13 tasks → monitor → verify COGs.
- Exit criteria met: all non-auth-dependent code verified; auth-required scripts clearly labelled (exit 2 without creds); 38 synthetic alignment tests pass.

## Stage 2 — Ground truth (M2) — BLOCKED (D5 not signed off; pilot rows not started)

- **D1** Imagery year: user stated **2019** in chat (2026-10-08) — provisional,
  NOT team-confirmed; labelling still gated on D5 regardless.
- **D2** WorldCover handling: user stated **stable-points comparison** in chat
  (2026-10-08) — provisional, NOT team-confirmed.
- **D3** AlphaEarth 2025 layer: **OPEN (unverified)**. The chat value
  "checked-not-exists" was never verified against the catalog — per ground
  rule 4 it is flagged unverified, not recorded as decided. Must check
  availability before M5.
- **D4** Cell size: user stated **revisit after semivariogram** (2026-10-08) —
  a deferral, open by definition; any change goes through rule 2.
- **D5** Guide sign-off: user stated **"no"** (2026-10-08) — explicitly NOT
  signed off; blocks Stage 2 per ground rule (stop at human gate).

- Skeleton frozen: zoned seed 6, checksum `449ad05e…` (docs/seed_log.md).
- Pilot timing rows: user confirmed "Not started" (2026-10-08). D5 = "no" blocks real labelling per ground rule.

## Stages 3–8 — NOT STARTED (Option B: independent workstreams; D5 gates Stage 2)

- Stage 3 (Classical M1-M5) cannot start until Stage 2 labels exist — **on hold**
- Stage 4 (Deep M6-M9) — **HELD per reviewer 2026-10-08** (needs labels + GPU session)
- Stage 5 (Analytics M5) needs change sample from labelled data — **on hold**
- Stage 6 (Product M6) — **HELD per reviewer 2026-10-08**
- Stage 7 (M7) and Stage 8 (M8) — downstream, **HELD per reviewer 2026-10-08**

## Reviewer follow-ups done (2026-10-08)

- `tests/test_config_tracked.py` (4 tests): every `configs/*.yaml` referenced
  by `dvc.yaml` + `labelling.yaml` must exist, be git-tracked, and not
  git-ignored; `.gitignore` must anchor `/data/`. Negative control verified
  (bare `data/` fails the anchor test). Guards the sentinel.yaml incident.
- Skip audit: CI's 7 skips were 1× `importorskip("torch")`
  (`test_tiled_smoke_runs_overlap_blend`) + 6× `importorskip("pyproj")` via
  `_utm_grid()` (both absent from CI's light closure by design). Skeleton
  guards all RUN: `test_canonical_checksum` passes (checksum line present in
  seed_log.md), leakage + buffer tests have no skip paths. Local `-rs` on the
  runnable subset: 0 skips.
- Reviewer round 2 applied (commit `6f31713`, main run **#43 Success**):
  `pyproj` added to the CI install line + `[test]` extra + colab lock input
  (torch smoke stays Colab-only), `-rs` added to CI's pytest flags. Result:
  `146 passed, 1 skipped` with the reason in the log
  (`SKIPPED [1] test_infer_analytics_cli.py:44: could not import 'torch'`).
  The six UTM-grid alignment tests now run on every push. Colab/linux `.lock`
  regeneration (for the new input pin) is pending per `docs/lockfile_policy.md`.

## Stages 3–8 — NOT STARTED

---

## Pending human actions (consolidated)

1. **File the 7 issues on GitHub** — run `docs/github_issues/file_issues.ps1` (PowerShell) to copy each issue body to the clipboard, then paste into https://github.com/veeresh0804/PrithviBodh/issues/new. Seven copies are in `docs/github_issues/issue_01..07.md`. Seed-3→seed-6 correction applied in all 7.
2. **Earth Engine: `earthengine authenticate`**, then run `scripts/ee/` dry-run → full export. Scripts verified: availability_report exits 2 without creds; dryrun prints checksum/size; alignment tests all pass synthetic.
3. **Run the 50-point pilot**; paste the 3 timing rows in chat → schedule draft. User confirmed "Not started" (2026-10-08).
4. **D5: guide sign-off meeting** on the split-design pack. Blocks Stage 2.
5. **Kaggle token rotation** — CONFIRMED DONE (user, 2026-10-08).
6. **Decide D1–D4** when their gates arrive (blocks stage 2/3/5 respectively). D1 (imagery year), D2 (WorldCover), D3 (AlphaEarth), D4 (cell size).

## Deviations from the brief

- `make check` cannot run locally (ruff blocked by App Control). Bare `pytest`
  from a foreign cwd still runs and passes; lint verification is CI-only.
  Disclosed here rather than silently skipped.
