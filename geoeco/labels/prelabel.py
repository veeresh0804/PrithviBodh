"""OPTIONAL train-only label suggestions (default OFF).

When enabled with --prelabel-train, points in TRAIN blocks where Dynamic
World and WorldCover agree (after the benchmark crosswalk) get a
`suggested_label` column. Rules:

- NEVER touches test points: any test row present raises immediately.
- NEVER writes `label` — only `suggested_label` + `suggested_by`.
- Humans must review/confirm every suggestion; unconfirmed suggestions are
  dropped at merge. Enabling is recorded in docs/dataset_card.md as label
  noise + benchmark coupling (models trained on these suggestions are partly
  coupled to B1/B2 — report it, do not hide it).
- Requires local rasters DW + WorldCover over Hyderabad (EE export).
  Missing inputs fail loudly (rule 4) — nothing is invented.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geoeco.evaluation.benchmark import DW_CROSSWALK, WC_CROSSWALK, crosswalk

DW_RASTER = Path("data/raw/benchmarks/dynamic_world_hyd.tif")
WC_RASTER = Path("data/raw/benchmarks/worldcover_hyd.tif")


def run_pretrain_suggestions(members_dir: str | Path) -> int:
    """Add suggested_label to TRAIN member rows where DW == WorldCover."""
    members_dir = Path(members_dir)
    if not DW_RASTER.is_file():
        raise FileNotFoundError(
            f"Missing {DW_RASTER}: export Dynamic World over the Hyderabad AOI "
            "from Earth Engine first (rule 4: cannot invent benchmark inputs).")
    if not WC_RASTER.is_file():
        raise FileNotFoundError(
            f"Missing {WC_RASTER}: export ESA WorldCover over the Hyderabad AOI "
            "from Earth Engine first.")
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("rasterio required for pre-labelling") from e
    n = 0
    for f in sorted(members_dir.glob("member_*.geojson")):
        data = json.loads(f.read_text(encoding="utf-8"))
        changed = False
        for feat in data["features"]:
            p = feat["properties"]
            if p.get("split") != "train":
                if "suggested_label" in p:
                    raise ValueError(f"Refusing: suggestion target on test row {p.get('id')}")
                continue
            lon, lat = feat["geometry"]["coordinates"]
            with rasterio.open(DW_RASTER) as src:
                dw = int(next(iter(src.sample([(lon, lat)])))[0])
            with rasterio.open(WC_RASTER) as src:
                wc = int(next(iter(src.sample([(lon, lat)])))[0])
            dw6 = int(crosswalk([dw], DW_CROSSWALK, "DW")[0])
            wc6 = int(crosswalk([wc], WC_CROSSWALK, "WorldCover")[0])
            if dw6 == wc6:
                p["suggested_label"] = dw6
                p["suggested_by"] = "DW+WC-agree"
                changed, n = True, n + 1
        if changed:
            f.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"suggested_label written on {n} train rows (human review required).")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train-only suggestions (default off).")
    ap.add_argument("--members-dir", default="data/labels/members")
    args = ap.parse_args(argv)
    run_pretrain_suggestions(args.members_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
