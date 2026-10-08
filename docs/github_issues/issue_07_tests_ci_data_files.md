TITLE: [tests] CI must test the real spatial design, not gitignored data files (fixed)

**Finding**
- Buffer tests read gitignored `data/` files → CI failed with FileNotFoundError
  on fresh clones (AGENTS.md rule: CI tests must never read `data/`).

**Fix applied**
- Tests generate skeletons in `tmp_path` through the real grid code; shared
  haversine/proxy math vendored from gitignored `collect.py` into
  `geoeco/labels/geo_stats.py`.
- Repro test pins the canonical seed-6 checksum `449ad05e…`.
- Related incident record: `docs/ci_incident_2026-10-06.md` (bare `pytest`
  vs `python -m pytest` sys.path difference; fixed with `tests/__init__.py` +
  `pythonpath = ["."]`).
