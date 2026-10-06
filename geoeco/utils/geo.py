"""CRS / grid helpers. Hyderabad grid is EPSG:32644 @ 10 m.

Display CRS is EPSG:4326. Two-state mosaics keep per-UTM processing
(43N/44N/45N) and mosaic in EPSG:4326 for display only.

Functions here are dependency-light (no hard rasterio/geopandas
requirement) so unit tests run anywhere; reprojection helpers
import geopandas/pyproj lazily.
"""

from __future__ import annotations

TARGET_CRS: str = "EPSG:32644"
DISPLAY_CRS: str = "EPSG:4326"
TARGET_RES_M: float = 10.0

Bounds = tuple[float, float, float, float]  # (minx, miny, maxx, maxy) in TARGET_CRS


def target_epsg() -> int:
    """Return the target EPSG code as int (32644)."""
    return 32644


def ensure_epsg_string(crs: str | int) -> str:
    """Normalise a CRS to ``"EPSG:<code>"`` form.

    Args:
        crs: Either an int code or an ``"EPSG:xxxx"`` string.

    Returns:
        Normalised CRS string.
    """
    if isinstance(crs, int):
        return f"EPSG:{crs}"
    text = crs.strip().upper().replace("EPGS:", "EPSG:")
    if text.startswith("EPSG:"):
        return text
    raise ValueError(f"Unrecognised CRS: {crs!r}")


def grid_shape(bounds: Bounds, resolution_m: float = TARGET_RES_M) -> tuple[int, int]:
    """Compute (rows, cols) for bounds at a given resolution.

    Args:
        bounds: (minx, miny, maxx, maxy) in metres (target CRS).
        resolution_m: Pixel size in metres.

    Returns:
        (rows, cols) integer counts (ceil to cover bounds).
    """
    import math

    minx, miny, maxx, maxy = bounds
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"Invalid bounds: {bounds}")
    cols = math.ceil((maxx - minx) / resolution_m)
    rows = math.ceil((maxy - miny) / resolution_m)
    return rows, cols


def bounds_to_affine(bounds: Bounds, resolution_m: float = TARGET_RES_M) -> tuple[float, ...]:
    """Return a GDAL-style geotransform tuple for the bounds.

    Order: ``(origin_x, pixel_w, 0, origin_y, 0, -pixel_h)`` with
    origin at the upper-left (maxy). Dependency-free; compatible
    with rasterio.Affine constructor arguments.

    Args:
        bounds: (minx, miny, maxx, maxy).
        resolution_m: Pixel size.

    Returns:
        6-element geotransform tuple.
    """
    minx, _miny, _maxx, maxy = bounds
    return (minx, resolution_m, 0.0, maxy, 0.0, -resolution_m)


def reproject_bounds(
    bounds: Bounds, src_crs: str = TARGET_CRS, dst_crs: str = DISPLAY_CRS
) -> Bounds:
    """Reproject rectangular bounds from src to dst CRS.

    Lazily imports geopandas/pyproj; raises a clear error if absent.

    Args:
        bounds: (minx, miny, maxx, maxy) in src CRS.
        src_crs: Source CRS string.
        dst_crs: Destination CRS string.

    Returns:
        Reprojected bounds tuple.
    """
    try:
        from pyproj import Transformer
        from shapely.geometry import box
        from shapely.ops import transform as shp_transform
    except ImportError as exc:
        raise ImportError(
            "reproject_bounds needs shapely+pyproj: pip install shapely pyproj"
        ) from exc
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    geom = shp_transform(transformer.transform, box(*bounds))
    return (geom.bounds[0], geom.bounds[1], geom.bounds[2], geom.bounds[3])
