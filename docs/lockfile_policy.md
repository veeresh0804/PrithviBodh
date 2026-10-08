# Lockfile Policy

Status: **Stage 0 deliverable — RESOLVED on Windows for Linux/CPython 3.10, NOT
yet executed on Linux/Colab** (see [Status](#status) below).

## What these files are

| File | Target | Contents |
| --- | --- | --- |
| `requirements.linux.lock.txt` | Linux x86_64, CPython 3.10 (e.g. ubuntu-latest CI, or any image with glibc ≥ 2.18) | Runtime dependency graph (all transitive deps) from `pyproject.toml` `[project].dependencies`, fully pinned `name==version`. |
| `requirements.colab.lock.txt` | Google Colab (GPU or CPU kernel), linux x86_64 / CPython 3.10 | Same closure as the linux lock **plus** the `[test]` extra (`pytest`, `pluggy`, `iniconfig`; `pydantic` was already in the base closure via `mlflow-skinny`). Includes `[V]/[K]/[A]`-tagged claims in its header. |
| `requirements.linux.in.txt`, `requirements.colab.in.txt` | — | Resolution **inputs** (lower bounds only, not a lock). Mirror `pyproject.toml` by hand and carry a "target-only" section (see below). |
| `requirements.lock` | Windows / CPython 3.14 | **Dev-only snapshot** of the author's Windows environment. Not usable for Linux/Colab deployment; see below. |

## Which lock for which platform

- **CI / production / any Linux container → `requirements.linux.lock.txt`.** It is
  the single install set for `pip install -r requirements.linux.lock.txt`
  (Python 3.10, linux x86_64, glibc ≥ 2.18).
- **Colab notebooks → `requirements.colab.lock.txt`.** Run
  `!pip install -r requirements.colab.lock.txt` at the top of the notebook. It
  deliberately includes the test extra so the same environment can run both
  training and the test suite. GPU and CPU kernels are covered by the same file
  (torch's default PyPI Linux wheel is the CUDA build; on a CPU kernel the
  CUDA libraries install but are simply unused).
- **Windows development → `requirements.lock` (dev-only).** Do not use it to
  provision CI, Linux, or Colab.

## Why the Windows snapshot (`requirements.lock`) is dev-only

`requirements.lock` was generated on a Windows machine with CPython 3.14. It is
an *environment snapshot*, not a *resolved target lock*:

- It reflects the **host interpreter** (3.14), not the CI/Colab target (3.10).
  Pins chosen under 3.14 can be wrong or absent for 3.10 (e.g. packages that
  dropped 3.10 support, or 3.14-only builds).
- It contains **win32-only** packages (`pywin32`, `waitress`) and misses
  **POSIX-only** ones (`gunicorn`, `pexpect`), because it was snapshotted where
  it was produced, on Windows.
- It is untracked-by-policy: regenerated ad hoc on the author's machine, so it
  can drift from `pyproject.toml`.

It stays in the repo only as a convenience for the author's local dev loop.

## How the linux/colab locks were generated (mechanism)

pip resolves against the **host** interpreter and evaluates environment markers
on the host (Windows / 3.14). To produce a Linux/3.10 lock from Windows, the
resolution is run with target platform constraints and the result is then
**re-audited** for the real target:

### 1. Resolve (cross-platform)

```powershell
python -m pip install --dry-run --ignore-installed `
  --report <temp>\lock_report_linux_cp310.json `
  --find-links <temp>\local_wheels `
  --python-version 3.10 --implementation cp --abi cp310 --only-binary=:all: `
  --platform manylinux2014_x86_64 `
  --platform manylinux_2_17_x86_64 --platform manylinux_2_18_x86_64 `
  --platform manylinux_2_19_x86_64 --platform manylinux_2_20_x86_64 `
  --platform manylinux_2_21_x86_64 --platform manylinux_2_22_x86_64 `
  --platform manylinux_2_23_x86_64 --platform manylinux_2_24_x86_64 `
  --platform manylinux_2_25_x86_64 --platform manylinux_2_26_x86_64 `
  --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64 `
  --platform win_amd64 `
  -r requirements.linux.in.txt
```

Repeat with `requirements.colab.in.txt` → `lock_report_colab_cp310.json`.

Why the flag set:

- `--platform manylinux_2_17…2_28_x86_64` (continuous, 2_17→2_28) covers the
  glibc range pip must consider for Ubuntu 20.04/22.04 CI and Colab images.
  `manylinux2014_x86_64` is listed explicitly as the legacy alias of `2_17`.
- `--platform win_amd64` is **required** for the host-Windows marker
  evaluation to succeed: with only linux platforms present, Windows-only edges
  (`docker → pywin32`, `mlflow → waitress`, `dvc → pywin32`) cannot be matched,
  and the resolver explodes in backtracking. The win-arm class of results is
  removed afterwards by the audit step — never shipped.
- `--find-links <temp>\local_wheels` feeds the locally built
  `antlr4_python3_runtime-4.9.3-py3-none-any.whl`. That version is **sdist-only
  on PyPI**, and pip refuses `--no-binary` resolution while platform
  restrictions are active, so the sdist was built into a wheel once (on Windows)
  and re-used for every resolution.
- `--only-binary=:all:` guarantees every pin corresponds to a real wheel (no
  source builds sneaking into a "lock").

### 2. Audit for the real target

```powershell
python <temp>\audit_report.py <temp>\lock_report_linux_cp310.json --out <temp>\closure_linux.json
python <temp>\audit_report.py <temp>\lock_report_colab_cp310.json --out <temp>\closure_colab.json
```

The audit re-evaluates the resolved graph's markers under a linux/3.10
environment and reports:

- **DROPPED** — present in the report but not needed on linux/3.10
  (win32-only edges). Expected: exactly `pywin32` and `waitress`.
- **MISSING** — needed on linux/3.10 but invisible to host-Windows pip
  (target-only markers). These are appended to the `.in` "target-only" section
  and the resolution is re-run until MISSING is empty. Caught this way:
  `exceptiongroup`, `tomli`, `async-timeout` (`python_version < "3.11"`),
  `pexpect`, `gunicorn` (`sys_platform != "win32"`), torch's Linux CUDA closure
  (`cuda-toolkit[cu*]`, `cuda-bindings`, `nvidia-*`, `triton`), and
  `nvidia-cufile` (attached to the `cufile` extra behind a **linux-only**
  marker — torch requests `cuda-toolkit[cufile]`, but host Windows pip cannot
  see the dependency).
- **WINWHEEL** — closure packages whose *selected* wheel filename is a win32
  wheel (an artifact of `--platform win_amd64`; pip picked the win copy). Each
  is verified against PyPI to have a compatible linux wheel for the same
  version. Currently: `protobuf==6.33.6` (has
  `cp39-abi3-manylinux2014_x86_64`) and `psutil==7.2.2` (has
  `cp36-abi3-manylinux2010/2_12_x86_64…`). Harmless; the linux wheels are what
  installs on the target.
- **BADREQPY** — Requires-Python not satisfied by 3.10.12. Currently: none.

The `--out` closure JSON is the definitive lock content: the audited target
closure minus nothing further (pywin32/waitress are already excluded by the
audit). The lock files are generated from it, sorted by name.

### 3. (Later) execute on the target

The locks are only *resolved*. The remaining step is to actually run
`pip install -r requirements.linux.lock.txt` on ubuntu-latest (CI) and
`!pip install -r requirements.colab.lock.txt` in one Colab GPU session, and fix
anything that fails there (see [Status](#status)).

## Known special cases (evidence)

- **antlr4-python3-runtime==4.9.3** — sdist-only on PyPI (only `.tar.gz`
  release files, verified 2026-10-08 via `https://pypi.org/pypi/antlr4-python3-runtime/4.9.3/json`).
  Required by `hydra-core`. On Linux it builds from source at install time
  (setuptools); on Windows-resolution the local wheel is used instead.
- **nvidia-nccl-cu13==2.30.7** — publishes only `manylinux_2_18+` x86_64
  wheels (PyPI JSON), hence the glibc ≥ 2.18 floor for the whole lock. Ubuntu
  20.04 (glibc 2.31) and current Colab images (glibc ≫ 2.18) are fine.
- **nvidia-cufile==1.15.1.6** — `py3-none-manylinux_2_17_x86_64` wheel exists;
  pinned explicitly because torch requests `cuda-toolkit[cufile]` but the
  dependency's marker is linux-only and invisible to host pip.
- **protobuf==6.33.6 / psutil==7.2.2** — see WINWHEEL above.
- **Colab preinstalls** torch/torchvision on GPU kernels; installing the colab
  lock upgrades/replaces them with the pinned PyPI versions. `[K]`/`[A]`-tagged
  claims about Colab runtime behaviour live in `requirements.colab.lock.txt`.

## Regeneration checklist

1. Edit `pyproject.toml` → mirror into `requirements.linux.in.txt`
   (`requirements.colab.in.txt` = same base + `[test]` + target-only section).
2. Run the resolution command above for both inputs.
3. Run the audit; resolve any MISSING by adding to the target-only section and
   re-resolving (loop until MISSING = none); confirm DROPPED = `pywin32`,
   `waitress` only; verify every WINWHEEL version against PyPI.
4. Regenerate the two lock files from the audited closures.
5. Commit `requirements.linux.lock.txt`, `requirements.colab.lock.txt`,
   `requirements.linux.in.txt`, `requirements.colab.in.txt`.
6. Run the locks once on ubuntu-latest CI and in one Colab GPU session before
   relying on them (see status).

Do **not** regenerate the Windows `requirements.lock` from these commands; that
file stays a dev-only host snapshot.

## Status

As of 2026-10-08, `requirements.linux.lock.txt` (293 pins) and
`requirements.colab.lock.txt` (296 pins) were **resolved** for
Linux x86_64 / CPython 3.10 on a Windows host with pip 26.2.1, and the
resolved graph was marker-audited for that target (MISSING = none; DROPPED =
pywin32, waitress; BADREQPY = none). They have **not yet been executed** on a
real Linux/3.10 environment. Pending before these locks are trusted:

- one CI run (`pip install -r requirements.linux.lock.txt` on ubuntu-latest);
- one Colab GPU run (`!pip install -r requirements.colab.lock.txt`);
- confirmation that the antlr4 source build and the pinned torch/CUDA wheels
  install cleanly on those images.

Evidence (resolution reports, closures, this audit tooling) live in the temp
workspace used to build this deliverable (`%LOCALAPPDATA%\Temp\opencode\`):
`lock_report_linux_cp310.json`, `lock_report_colab_cp310.json`,
`closure_linux.json`, `closure_colab.json`, `audit_report.py`. They are not
committed; delete them once the locks are validated on Linux/Colab, or keep as
local audit evidence.