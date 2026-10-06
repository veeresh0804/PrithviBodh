# AGENTS.md — standing instructions for AI agents in this repo

## Collaboration preference (user standing instruction)
- **Default to parallel execution.** Whenever a task splits into independent
  workstreams, deploy multiple subagents in parallel instead of working
  sequentially. Ask only if the streams genuinely share files.
- **Partition by file ownership.** Each parallel agent owns a disjoint set of
  files/directories (e.g. one agent owns `tests/test_x.py`, another owns
  `geoeco/labels/y.py`). Agents must not write to files owned by another
  stream. Shared contracts (function names, column names) are agreed up front
  in the dispatching prompt.
- This applies to implementation, audits, fixes, and verification alike.

## Repo facts agents must respect
- `data/` is gitignored (DVC-tracked, never git). Tests running in CI (fresh
  clone) MUST NOT read files under `data/` — generate fixtures in `tmp_path`
  or commit small fixtures under `tests/fixtures/`.
- Hard rules: no test-label autofill, no DW/WorldCover use in test design,
  seed 42 from configs, config-driven params, fail loudly on missing inputs.
- Windows (PowerShell) environment. No `head`, no `ls ~/.x` — use
  `Select-Object`, `$env:USERPROFILE`. Long one-liners get mangled: prefer
  script files over `python -c`.
- `kaggle` exe is not on PATH — use `python -m kaggle`. Never print secrets.

## Note on skills/memory
System skills are load-only for agents (no skill-creation tool exists), so
this file is the persistent memory mechanism: it is read automatically in
future sessions working in this directory.
