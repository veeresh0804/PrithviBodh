"""COG export helpers: int16 scaled output, 256 tiling, STAC hook.

Default export: Cloud-Optimised GeoTIFF, int16 with scale/offset,
256x256 internal tiling, DEFLATE. STAC registration itself lives in
:mod:`geoeco.ingest.stac`; :func:`stac_register_stub` records the
intent/metadata so pipelines can call export without a STAC backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

COG_BLOCKSIZE: int = 256
COG_COMPRESS: str = "DEFLATE"
COG_DTYPE: str = "int16"


def scale_to_int16(
    arr: np.ndarray, scale: float = 0.0001, nodata: int = -9999
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale a float reflectance-like array to int16 for COG export.

    Uses ``stored = round(value / scale)``; NaN maps to ``nodata``.

    Args:
        arr: Float array (e.g. reflectance 0-1, indices -1..1).
        scale: Scale factor recorded in metadata.
        nodata: Nodata sentinel (must fit int16).

    Returns:
        (int16_array, meta) with scale/nodata description.
    """
    data = np.asarray(arr, dtype=np.float64)
    out = np.full(data.shape, nodata, dtype=np.int16)
    valid = ~np.isnan(data)
    out[valid] = np.clip(np.rint(data[valid] / scale), -32768, 32767).astype(np.int16)
    return out, {"dtype": "int16", "scale": scale, "nodata": nodata}


def cog_profile(
    width: int, height: int, count: int, crs: str = "EPSG:32644", transform: Any = None
) -> dict[str, Any]:
    """Build a rasterio-style COG profile dict (no I/O).

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        count: Band count.
        crs: Target CRS string.
        transform: Affine transform (passed through).

    Returns:
        Profile dict ready for ``rasterio.open(..., **profile)``.
    """
    return {
        "driver": "COG",
        "dtype": COG_DTYPE,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "blocksize": COG_BLOCKSIZE,
        "compress": COG_COMPRESS,
        "BIGTIFF": "IF_SAFER",
    }


def write_cog(
    path: str, arr: np.ndarray, crs: str = "EPSG:32644", transform: Any = None
) -> dict[str, Any]:
    """Write an int16 array as COG (lazy rasterio import).

    Args:
        path: Destination ``.tif`` path.
        arr: Array shaped (bands, rows, cols) or (rows, cols).
        crs: CRS string.
        transform: Affine transform.

    Returns:
        The profile used for writing.

    Raises:
        ImportError: If ``rasterio`` is not installed.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("write_cog needs rasterio: pip install rasterio") from exc
    data = np.asarray(arr)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    _, rows, cols = data.shape
    profile = cog_profile(cols, rows, data.shape[0], crs, transform)
    # rasterio has no "COG" driver string for open(); use GTiff with
    # COMPRESS + TILED + COPY_SRC_OVERVIEWS workflow instead.
    profile["driver"] = "GTiff"
    profile.update({"tiled": True, "compress": COG_COMPRESS.lower()})
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return profile


def stac_register_stub(cog_path: str, item_id: str) -> dict[str, Any]:
    """Record STAC-registration intent (STUB).

    Real registration is :func:`geoeco.ingest.stac.create_item`; this
    stub returns the minimal record the export stage logs to DVC/MLflow
    so offline runs stay reproducible without a catalog backend.

    Args:
        cog_path: Path/URL of the exported COG.
        item_id: Intended STAC item id.

    Returns:
        Stub record dict.
    """
    return {"item_id": item_id, "asset_href": cog_path, "status": "pending-stac-write"}
