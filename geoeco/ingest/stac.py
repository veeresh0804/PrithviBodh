"""Static STAC catalog creation (pystac).

Produces a self-contained static JSON catalog over exported COGs so the
data layer is standard EO metadata even on free hosting (no STAC API
server needed). ``pystac`` is imported lazily; :func:`item_dict` works
without it for offline tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def item_dict(
    item_id: str,
    cog_href: str,
    bounds_wgs84: tuple[float, float, float, float],
    start: str,
    end: str,
    epsg: int = 32644,
) -> dict[str, Any]:
    """Build a minimal STAC-item-like dict without pystac.

    Args:
        item_id: STAC item id.
        cog_href: HREF of the COG asset.
        bounds_wgs84: (minlon, minlat, maxlon, maxlat).
        start: ISO start datetime.
        end: ISO end datetime.
        epsg: Native CRS code (``projection:epsg`` extension field).

    Returns:
        Plain-dict STAC item (JSON-serialisable).
    """
    minlon, minlat, maxlon, maxlat = bounds_wgs84
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [minlon, minlat],
                    [maxlon, minlat],
                    [maxlon, maxlat],
                    [minlon, maxlat],
                    [minlon, minlat],
                ]
            ],
        },
        "bbox": [minlon, minlat, maxlon, maxlat],
        "properties": {"start_datetime": start, "end_datetime": end, "proj:epsg": epsg},
        "assets": {"image": {"href": cog_href, "type": "image/tiff; application=geotiff"}},
        "links": [],
    }


def create_collection(collection_id: str, description: str = "") -> Any:
    """Create a ``pystac.Collection`` for Hyderabad seasonal products.

    Args:
        collection_id: Collection id.
        description: Human-readable description.

    Returns:
        ``pystac.Collection``.

    Raises:
        ImportError: If ``pystac`` is not installed.
    """
    try:
        import pystac
    except ImportError as exc:
        raise ImportError("STAC helpers need pystac: pip install pystac") from exc
    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
        pystac.TemporalExtent([[datetime(2019, 1, 1, tzinfo=timezone.utc), None]]),
    )
    return pystac.Collection(
        id=collection_id, description=description or collection_id, extent=extent
    )


def create_item(
    item_id: str,
    cog_href: str,
    bounds_wgs84: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
) -> Any:
    """Create a ``pystac.Item`` pointing at a COG asset.

    Args:
        item_id: Item id.
        cog_href: COG HREF.
        bounds_wgs84: (minlon, minlat, maxlon, maxlat).
        start: Start datetime (tz-aware preferred).
        end: End datetime.

    Returns:
        ``pystac.Item`` with a single ``image`` asset.

    Raises:
        ImportError: If ``pystac`` is not installed.
    """
    try:
        import pystac
    except ImportError as exc:
        raise ImportError("STAC helpers need pystac: pip install pystac") from exc
    minlon, minlat, maxlon, maxlat = bounds_wgs84
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat], [minlon, minlat]]
        ],
    }
    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=[minlon, minlat, maxlon, maxlat],
        datetime=start,
        properties={"start_datetime": start.isoformat(), "end_datetime": end.isoformat()},
    )
    item.add_asset("image", pystac.Asset(href=cog_href, media_type="image/tiff; application=geotiff"))
    return item


def write_static_catalog(catalog: Any, out_dir: str) -> str:
    """Write a static (self-contained) catalog to disk.

    Args:
        catalog: ``pystac.Catalog``.
        out_dir: Destination directory.

    Returns:
        Path of the written ``catalog.json``.
    """
    catalog.normalize_hrefs(out_dir)
    catalog.save(catalog_type="SELF_CONTAINED")  # type: ignore[attr-defined]
    return f"{out_dir.rstrip('/')}/catalog.json"


REPO = Path(__file__).resolve().parents[2]
DEFAULT_AOI_CFG = REPO / "configs" / "aoi" / "hyderabad.yaml"
DEFAULT_DATA_CFG = REPO / "configs" / "data" / "sentinel.yaml"
DEFAULT_OUT = REPO / "data" / "raw" / "composites"
SPATIAL_CV_CFG = REPO / "configs" / "eval" / "spatial_cv.yaml"


def resolve_seed(data_cfg: dict[str, Any], explicit: int | None) -> int:
    """Seed from CLI flag, else data config, else spatial_cv.yaml, else 42."""
    if explicit is not None:
        return int(explicit)
    seed = data_cfg.get("seed")
    if isinstance(seed, int):
        return int(seed)
    try:
        from geoeco.utils.config import load_yaml_config

        cv = load_yaml_config(SPATIAL_CV_CFG)
        if isinstance(cv.get("seed"), int):
            return int(cv["seed"])
    except (FileNotFoundError, ValueError, TypeError):
        pass
    return 42


def collect_items(outdir: Path, aoi: dict[str, Any],
                  manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Items from the export manifest plus any unmatched COGs in the out dir."""
    items = [dict(it) for it in manifest.get("items", [])]
    known = {it.get("assets", {}).get("image", {}).get("href") for it in items}
    bounds = [float(v) for v in aoi["bounds_wgs84"]]
    windows = manifest.get("windows", [])
    fallback_start = windows[0]["start"] if windows else "2019-01-01"
    fallback_end = windows[0]["end"] if windows else "2019-02-01"
    for tif in sorted(outdir.rglob("*.tif")):
        href = tif.relative_to(outdir).as_posix()
        if href in known:
            continue
        items.append(item_dict(
            item_id=tif.stem.replace(".", "-").replace("/", "-"),
            cog_href=href,
            bounds_wgs84=(bounds[0], bounds[1], bounds[2], bounds[3]),
            start=fallback_start,
            end=fallback_end,
        ))
    return items


def write_catalog_json(outdir: Path, collection_id: str, description: str,
                        items: list[dict[str, Any]], seed: int) -> Path:
    """Write a static catalog with pystac when available, else plain JSON."""
    try:
        import pystac  # noqa: F401

        has_pystac = True
    except ImportError:
        has_pystac = False
    if has_pystac:
        import pystac

        catalog = pystac.Catalog(id=f"{collection_id}-catalog",
                                 description=description)
        collection = create_collection(collection_id, description)
        catalog.add_child(collection)
        for it in items:
            props = it.get("properties", {})
            start = datetime.fromisoformat(str(props.get("start_datetime")))
            end = datetime.fromisoformat(str(props.get("end_datetime")))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            asset_href = it.get("assets", {}).get("image", {}).get("href", "")
            collection.add_item(create_item(
                it.get("id", "item"), asset_href,
                tuple(it.get("bbox", [0, 0, 0, 0])),  # type: ignore[arg-type]
                start, end))
        catalog.normalize_hrefs(str(outdir))
        catalog.save(catalog_type="SELF_CONTAINED")  # type: ignore[attr-defined]
        return outdir / "catalog.json"
    dest = outdir / "catalog.json"
    dest.write_text(json.dumps({
        "stac_version": "1.0.0",
        "id": collection_id,
        "description": description,
        "seed": seed,
        "items": items,
        "links": [],
    }, indent=2) + "\n", encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a static STAC catalog from an export out dir "
        "(manifest.json + COGs; offline)."
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Export dir to scan; catalog.json is written here.")
    ap.add_argument("--config", default=str(DEFAULT_AOI_CFG))
    ap.add_argument("--data", default=str(DEFAULT_DATA_CFG))
    ap.add_argument("--collection-id", default="hyderabad-seasonal")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    from geoeco.utils.config import load_yaml_config, require_keys

    outdir = Path(args.out)
    if not outdir.is_dir():
        print(f"ingest stac: missing input dir: {outdir}", file=sys.stderr)
        return 2
    try:
        aoi = load_yaml_config(args.config)
        require_keys(aoi, ["bounds_wgs84"], name="aoi config")
        data = load_yaml_config(args.data)
    except FileNotFoundError as exc:
        print(f"ingest stac: missing input: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, TypeError) as exc:
        print(f"ingest stac: invalid config: {exc}", file=sys.stderr)
        return 2
    seed = resolve_seed(data if isinstance(data, dict) else {}, args.seed)
    manifest_path = outdir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        items = collect_items(outdir, aoi, manifest)
    except (KeyError, ValueError) as exc:
        print(f"ingest stac: cannot build items: {exc}", file=sys.stderr)
        return 2
    if not items:
        print(f"ingest stac: no COGs or manifest items found in {outdir}; "
              "run the export CLI first (nothing invented)", file=sys.stderr)
        return 2
    dest = write_catalog_json(outdir, args.collection_id,
                              f"Static catalog over {outdir.name}", items, seed)
    print(json.dumps({"catalog": str(dest), "items": len(items)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
