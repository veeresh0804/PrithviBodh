"""Feature-stack spec builder: validate the 20-channel/season spec from config.

Stack contract (configs/data/sentinel.yaml): per season 14 optical
(10 S2 bands + 4 indices) + 4 SAR + 2 terrain = 20 channels; classical
models stack 3 seasons (~60 channels).

The CLI validates the spec purely from config, then refuses (exit 1)
when the EE composite inputs (data/raw/composites/, the ingest-stage
output per dvc.yaml) are absent — it never invents raster data. On
success it writes a stack-spec JSON manifest to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from geoeco.utils.config import load_yaml_config, require_keys

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "data" / "sentinel.yaml"
DEFAULT_OUT = REPO / "data" / "processed" / "feature_stack"
DEFAULT_COMPOSITES = REPO / "data" / "raw" / "composites"
MANIFEST_NAME = "stack_spec.json"

OPTICAL_CHANNELS = 14  # 10 S2 bands + 4 indices
SAR_CHANNELS = 4
TERRAIN_CHANNELS = 2
CHANNELS_PER_SEASON = 20


def load_sentinel_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and structural-check the Sentinel feature config."""
    cfg = load_yaml_config(path)
    require_keys(
        cfg,
        ["seasons", "sentinel2", "sentinel1", "dem", "stack_per_season"],
        name="sentinel.yaml",
    )
    require_keys(cfg["sentinel2"], ["bands", "indices"], name="sentinel.yaml:sentinel2")
    require_keys(cfg["sentinel1"], ["features"], name="sentinel.yaml:sentinel1")
    require_keys(cfg["dem"], ["derived"], name="sentinel.yaml:dem")
    return cfg


def build_stack_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate channel counts and return the per-season stack spec.

    Raises:
        ValueError: If any group count (or the total) breaks the
            14 + 4 + 2 = 20 contract.
    """
    seasons = list(cfg["seasons"])
    bands = list(cfg["sentinel2"]["bands"])
    indices = list(cfg["sentinel2"]["indices"])
    sar = list(cfg["sentinel1"]["features"])
    terrain = list(cfg["dem"]["derived"])
    n_optical = len(bands) + len(indices)
    per_season = n_optical + len(sar) + len(terrain)
    if n_optical != OPTICAL_CHANNELS:
        raise ValueError(
            f"Optical channels must be {OPTICAL_CHANNELS} "
            f"({len(bands)} bands + {len(indices)} indices), got {n_optical}"
        )
    if len(sar) != SAR_CHANNELS:
        raise ValueError(f"SAR features must be {SAR_CHANNELS}, got {len(sar)}: {sar}")
    if len(terrain) != TERRAIN_CHANNELS:
        raise ValueError(
            f"Terrain derived must be {TERRAIN_CHANNELS}, got {len(terrain)}: {terrain}"
        )
    if per_season != CHANNELS_PER_SEASON or per_season != int(cfg["stack_per_season"]):
        raise ValueError(
            f"Channels/season must be {CHANNELS_PER_SEASON} "
            f"(config stack_per_season={cfg['stack_per_season']}), got {per_season}"
        )
    return {
        "seasons": seasons,
        "n_seasons": len(seasons),
        "optical_bands": bands,
        "optical_indices": indices,
        "sar_features": sar,
        "terrain_derived": terrain,
        "breakdown": {"optical": n_optical, "sar": len(sar), "terrain": len(terrain)},
        "channels_per_season": per_season,
        "total_channels": per_season * len(seasons),
    }


def find_missing_inputs(composites_dir: str | Path = DEFAULT_COMPOSITES) -> list[str]:
    """List missing composite inputs; empty means the stack can be built."""
    d = Path(composites_dir)
    if not d.is_dir():
        return [f"{d}: directory absent (run the ingest stage first)"]
    files = [p for p in d.iterdir() if p.is_file()]
    if not files:
        return [f"{d}: directory empty (no seasonal composites)"]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate feature-stack spec; refuse without composites."
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--composites-dir", default=str(DEFAULT_COMPOSITES))
    args = ap.parse_args(argv)
    try:
        spec = build_stack_spec(load_sentinel_config(args.config))
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f"ERROR: invalid feature config: {exc}", file=sys.stderr)
        return 1
    missing = find_missing_inputs(args.composites_dir)
    if missing:
        print(f"ERROR: composite inputs absent: {missing[0]}", file=sys.stderr)
        return 1
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / MANIFEST_NAME
    manifest.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "spec": spec}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
