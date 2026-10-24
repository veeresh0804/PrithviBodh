"""Load the single project config."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("aoi", "seasons", "classes", "years", "cv", "models"):
        if key not in cfg:
            raise KeyError(f"config.yaml is missing '{key}'")
    return cfg
