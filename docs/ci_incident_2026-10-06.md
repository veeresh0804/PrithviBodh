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
