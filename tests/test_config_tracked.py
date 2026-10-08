"""Regression test: referenced configs must exist, be git-tracked, not ignored.

Guards the 2026-10-08 incident: `configs/data/sentinel.yaml` (read by ~8
modules) was silently ignored by a bare `data/` .gitignore line and absent
from fresh clones, while CI kept passing because the file existed locally.
Any future .gitignore change that re-hides a referenced config fails here.

What it checks:
1. Every `configs/*.yaml` path mentioned in `dvc.yaml` (cmds, deps, params)
   and `configs/labels/labelling.yaml` exists on disk.
2. Each is tracked by git (`git ls-files --error-unmatch`).
3. None is ignored (`git check-ignore -q` must NOT match — the direct
   regression guard for the `data/` vs `/data/` bug).
4. `.gitignore` anchors top-level data as `/data/`, not bare `data/`.

Only `configs/` paths are checked: `data/` outputs are DVC-tracked and
gitignored by design (AGENTS.md), as are secrets/local files.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

# A configs/ path ending in .yaml/.yml inside dvc stage commands, deps, params.
_DVC_REF = re.compile(r"configs/[^\s\"':]+\.ya?ml")
# A configs/ mention inside labelling.yaml text (may carry a :key suffix).
_LAB_REF = re.compile(r"configs/[A-Za-z0-9_./-]+")


def _git(*args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as e:
        raise AssertionError("git binary not found: this test requires git") from e


def _dvc_config_refs() -> set[str]:
    with open(REPO / "dvc.yaml", encoding="utf-8") as fh:
        dvc = yaml.safe_load(fh)
    refs: set[str] = set()
    for stage in dvc["stages"].values():
        for key in ("cmd", "deps", "params"):
            val = stage.get(key)
            texts = [val] if isinstance(val, str) else list(val or [])
            for text in texts:
                refs.update(_DVC_REF.findall(str(text)))
    return refs


def _labelling_refs() -> set[str]:
    text = (REPO / "configs" / "labels" / "labelling.yaml").read_text(encoding="utf-8")
    return {m.split(":")[0] for m in _LAB_REF.findall(text)}


def _referenced_configs() -> list[str]:
    refs = sorted(_dvc_config_refs() | _labelling_refs())
    assert refs, "ref harvest found nothing — regexes no longer match dvc.yaml?"
    return refs


def test_referenced_configs_exist():
    missing = [p for p in _referenced_configs() if not (REPO / p).is_file()]
    assert not missing, f"referenced configs missing from disk: {missing}"


def test_referenced_configs_tracked_by_git():
    untracked = [
        p
        for p in _referenced_configs()
        if _git("ls-files", "--error-unmatch", "--", p).returncode != 0
    ]
    assert not untracked, (
        f"referenced configs NOT tracked by git (fresh clones break): {untracked}"
    )


def test_referenced_configs_not_ignored():
    ignored = [
        p
        for p in _referenced_configs()
        if _git("check-ignore", "-q", "--", p).returncode == 0
    ]
    assert not ignored, (
        f"referenced configs matched by .gitignore (the sentinel.yaml bug): {ignored}"
    )


def test_gitignore_anchors_top_level_data():
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/data/" in lines, ".gitignore must anchor top-level data as /data/"
    bare = [ln for ln in lines if ln.strip() == "data/"]
    assert not bare, (
        "bare `data/` also matches configs/data/ — use the anchored `/data/` only"
    )
