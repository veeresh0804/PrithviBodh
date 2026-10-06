"""Pydantic schemas for the serving layer.

Covers GeoJSON polygon input (POST /analyze) and product query params.
Polygon area cap: 100 km2 (NFR-05, api_limits.polygon_km2).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CLASSES_6: list[str] = [
    "water",
    "tree_cover",
    "cropland",
    "built_up",
    "bare_rocky",
    "grass_shrub",
]

POLYGON_AREA_CAP_KM2 = 100.0

Position = list[float]  # [lon, lat]
Ring = list[Position]


class PolygonGeometry(BaseModel):
    """GeoJSON Polygon geometry (EPSG:4326 lon/lat)."""

    type: Literal["Polygon"]
    coordinates: list[Ring]

    @field_validator("coordinates")
    @classmethod
    def _check_rings(cls, rings: list[Ring]) -> list[Ring]:
        if not rings:
            raise ValueError("Polygon must have at least one ring")
        for ring in rings:
            if len(ring) < 4:
                raise ValueError("Each ring needs >= 4 positions (closed)")
            if ring[0] != ring[-1]:
                raise ValueError("Ring is not closed (first != last position)")
            for pos in ring:
                if len(pos) != 2:
                    raise ValueError("Positions must be [lon, lat]")
                lon, lat = pos
                if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                    raise ValueError(f"Position out of range: {pos}")
        return rings


def polygon_area_km2(geom: PolygonGeometry) -> float:
    """Approximate planar area (km2) via equirectangular projection.

    Adequate for the 100 km2 admin cap gate; NOT for billing-grade stats
    (server-side PostGIS ST_Area is authoritative in prod).
    """
    ring = geom.coordinates[0]
    mean_lat = sum(p[1] for p in ring) / len(ring)
    kx = 111.320 * math.cos(math.radians(mean_lat))
    ky = 110.574
    pts = [(p[0] * kx, p[1] * ky) for p in ring]
    area = 0.0
    for i in range(len(pts) - 1):
        area += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
    return abs(area) / 2.0


class AnalyzeRequest(BaseModel):
    """POST /analyze body: polygon + product(s) to summarise."""

    polygon: PolygonGeometry
    from_product_id: int = Field(gt=0)
    to_product_id: int | None = Field(default=None, gt=0)


class ProductQuery(BaseModel):
    """GET /products filters (mirrors query params, for docs/tests)."""

    region: str | None = None
    product_type: str | None = None
    year: int | None = None
    season: str | None = None
