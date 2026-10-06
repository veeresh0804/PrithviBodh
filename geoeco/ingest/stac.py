"""Static STAC catalog creation (pystac).

Produces a self-contained static JSON catalog over exported COGs so the
data layer is standard EO metadata even on free hosting (no STAC API
server needed). ``pystac`` is imported lazily; :func:`item_dict` works
without it for offline tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
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
