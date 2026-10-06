"""YAML config loader.

No secrets are read here; callers pass file paths explicitly.
Environment-variable expansion of the form ``${VAR}`` is supported
so CI/hosting can inject non-secret paths without editing YAML.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}:\s]+)(?::([^}]*))?\}")


def _expand_env(value: str) -> str:
    """Expand ${VAR} / ${VAR:default} occurrences in a string."""

    def _repl(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        return os.environ.get(var, default if default is not None else "")

    return _ENV_PATTERN.sub(_repl, value)


def _expand_obj(obj: Any) -> Any:
    """Recursively expand env vars in strings inside nested structures."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, Mapping):
        return {k: _expand_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_expand_obj(v) for v in obj]
    return obj


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict with env-var expansion.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` file.

    Returns:
        Parsed config mapping.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the top-level YAML node is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML node must be a mapping: {path}")
    return _expand_obj(data)


def get_nested(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Get a nested value via ``"a.b.c"`` dotted key.

    Args:
        config: Config mapping.
        dotted: Dotted key path.
        default: Returned when any level is missing.

    Returns:
        The nested value or ``default``.
    """
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def require_keys(config: Mapping[str, Any], keys: list[str], name: str = "config") -> None:
    """Validate that top-level keys exist.

    Args:
        config: Config mapping.
        keys: Required top-level keys.
        name: Label used in the error message.

    Raises:
        KeyError: If any key is missing.
    """
    missing = [k for k in keys if k not in config]
    if missing:
        raise KeyError(f"{name} missing required keys: {missing}")
