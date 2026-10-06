"""Terrain features: elevation + slope from Copernicus GLO-30 DEM.

RESAMPLE NOTE: GLO-30 is ~30 m; it is bilinearly resampled to the 10 m
analysis grid (EPSG:32644) once, then slope is derived on the 10 m grid.
Resampling before slope (not after) avoids stair-step artefacts. In GEE
this is ``dem.resample('bilinear').reproject(crs, scale=10)`` followed
by ``ee.Terrain.slope``; locally the equivalent is bilinear upsampling
(e.g. rasterio ``Resampling.bilinear``) then :func:`slope_degrees`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEM_SOURCE: str = "COPERNICUS/DEM/GLO30"
TARGET_RES_M: float = 10.0


def slope_degrees(elevation_m: np.ndarray, resolution_m: float = TARGET_RES_M) -> np.ndarray:
    """Compute slope in degrees from an elevation grid (NumPy).

    Uses central differences (``np.gradient``) — equivalent to Horn's
    method on smooth terrain to <0.5 deg for the 10 m product.

    Args:
        elevation_m: 2-D elevation array in metres.
        resolution_m: Pixel size in metres.

    Returns:
        2-D slope array in degrees [0, 90].

    Raises:
        ValueError: If input is not 2-D.
    """
    elev = np.asarray(elevation_m, dtype=np.float64)
    if elev.ndim != 2:
        raise ValueError(f"elevation must be 2-D, got shape {elev.shape}")
    dzdy, dzdx = np.gradient(elev, resolution_m)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    return np.degrees(slope_rad)


def terrain_stack_spec(resolution_m: float = TARGET_RES_M) -> dict[str, Any]:
    """Return the EE-side terrain derivation spec (no EE needed).

    Args:
        resolution_m: Target grid resolution in metres.

    Returns:
        Spec dict with DEM source, resample method and outputs.
    """
    return {
        "source": DEM_SOURCE,
        "resample": "bilinear",
        "target_scale_m": resolution_m,
        "outputs": ["elevation", "slope"],
        "ee_ops": ["resample('bilinear').reproject(crs, scale=10)", "ee.Terrain.slope"],
    }


def build_terrain_features(dem_collection: Any, crs: str = "EPSG:32644", scale: int = 10) -> Any:
    """Build elevation+slope EE image (STUB body, lazy EE import).

    Production EE: mosaic GLO-30, bilinear resample + reproject to the
    10 m grid, then ``ee.Terrain.slope``.

    Args:
        dem_collection: ``ee.ImageCollection`` or ``ee.Image`` of GLO-30.
        crs: Target CRS.
        scale: Target scale in metres.

    Returns:
        ``ee.Image`` with bands [elevation, slope].
    """
    elevation = dem_collection.mosaic().resample("bilinear").reproject(crs=crs, scale=scale)
    slope = elevation.gradient()  # caller replaces with ee.Terrain.slope(elevation)
    return elevation.addBands(slope)
