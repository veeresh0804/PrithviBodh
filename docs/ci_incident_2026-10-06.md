# CI incident 2026-10-06 — unit-test collection failure (resolved)

## Symptom
`ci / lint-test` failed at "Unit tests" ~2 s in (exit code 2 = collection
error), runs #12–#24. Install, ruff, and gitleaks steps passed throughout.

## Investigation (all dead ends recorded)
Ruled out with evidence, NOT by assumption: ruff (clean), test logic (62 pass
locally, in minimal venvs, in fresh clones, across 5 hash seeds), Python 3.10
(vermin-clean; full suite green under real 3.10.11), Linux (reviewer ran the
suite green on Linux), missing admin key, light-vs-full installs (full 3.10
install green), dependency versions (frozen set recorded below).

## Root cause (verified, not guessed)
`tests/` had no `__init__.py`. Locally everything ran via `python -m pytest`,
which puts the repo root on `sys.path`, so `import api` and
`from tests.… import …` worked. CI runs bare `pytest`, which only inserts the
test file's own directory — so collection died with
`ModuleNotFoundError: No module named 'api'`. Found via the pytest-tail
annotation added to CI (readable through the public annotations API).

## Fix
- `tests/__init__.py` (empty): makes `tests` a package, so pytest inserts the
  repo root on `sys.path` under both `pytest` and `python -m pytest`.
- Hardened the hash-seed subprocess test to use absolute `REPO` instead of
  `cwd="."` (same class of CWD assumption).
- Verified from a foreign working directory (the exact CI condition): 62 pass.
- `requirements.lock` (611 lines, full runtime freeze) committed for the
  Starlette/httpx deprecation risk noted in review; `test` extra + `make check`
  document the light test closure.

## Honesty note
Intermediate hypotheses (httpx, heavy-install interaction) were recorded as
unconfirmed at the time and are now superseded — httpx stays as a genuine
missing dep regardless. No green run was ever claimed before evidence.

---

# CI incident 2026-10-08 — ruff lint regression on unpinned ruff (run #30)

## Symptom
`ci / lint-test` "Lint (ruff)" failed with 13 errors spread across
`geoeco/api/tests` (runs #29–#30) after being clean for weeks. Every error
belonged to rule families that had never fired in this repo (`BLE001`,
`RUF*`, `ISC004`, `I001`).

## Root cause (verified, not guessed)
CI installs ruff unpinned (`pip install ruff`), so it silently tracks PyPI.
ruff 0.16.0 (2026-07-23) expanded its default rule set from 59 → 413 rules,
enabling `BLE001`, `RUF100`, `ISC004`, `I001`, `RUF046`, … for the first
time. The failing run pulled an in-flight `ruff==0.16.10`; that exact release
was re-verified in a throwaway Linux container against this repo. No repo
change caused the failure — the toolchain changed under us.

## Why the annotation hid it
The lint step mirrored only the last 40 lines of `ruff.log`; the new default
rule families sort after pyflakes errors, so 10 of the 13 errors fell outside
the visible annotation.

## Fixes
- Lint, all real fixes with no test-behavior change: removed dead `# noqa`
  and unused imports (RUF100/F401), dropped redundant `int()` wraps (RUF046),
  parenthesized implicit string concatenations (ISC004), narrowed
  `ee_available()` try blocks to `ImportError` (BLE001) with a justified
  `# noqa: BLE001` only at the genuine EE/network boundary in each `main()`,
  re-sorted one import block (I001), dropped an unused local (F841).
- CI: uncapped the ruff annotation — the full log is printed to the step
  console and mirrored as one annotation when it fits the ~60 KB cap, else
  the last 200 lines.
- Re-verified green with the exact ruff release CI uses (docker
  python:3.10-slim + `ruff==0.16.10`): 0 lint errors, 143 tests pass.

## Recommendation (not applied)
Pin ruff to a known-good version (verified `ruff==0.16.10`) and/or add an
explicit `select` under `[tool.ruff]` so default-set drift cannot break CI
silently again.
