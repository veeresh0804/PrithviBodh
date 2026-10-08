"""Stats / analysis routes (public, read-only).

- GET  /stats/{product_id}?admin_unit=
- POST /analyze                     (100 km2 polygon cap -> 413)
- GET  /change?from=&to=&admin_unit=
- GET  /compare?product_id=&baseline=dynamic_world
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.db import STUB_AREA_HA, STUB_PRODUCTS
from api.schemas import (
    CLASSES_6,
    POLYGON_AREA_CAP_KM2,
    AnalyzeRequest,
    polygon_area_km2,
)

router = APIRouter(tags=["stats"])

BASELINES = ("dynamic_world", "worldcover", "esri")


def _product_exists(product_id: int) -> None:
    if not any(p["id"] == product_id for p in STUB_PRODUCTS):
        raise HTTPException(status_code=404,
                            detail=f"Product {product_id} not found")


@router.get("/stats/{product_id}")
def product_stats(
    product_id: int,
    admin_unit: str | None = Query(default=None),
) -> dict:
    """Per-class area (ha) + error-adjusted CIs stub for one product."""
    _product_exists(product_id)
    areas = [
        {"class": c, "area_ha": STUB_AREA_HA[c],
         "ci_low_ha": round(STUB_AREA_HA[c] * 0.92, 1),
         "ci_high_ha": round(STUB_AREA_HA[c] * 1.08, 1)}
        for c in CLASSES_6
    ]
    return {"product_id": product_id, "admin_unit": admin_unit,
            "unit": "ha", "areas": areas,
            "note": "stub proportions; prod reads PostGIS AREA_STAT (Olofsson CIs)"}


@router.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    """Summarise a user polygon: class areas + change + indicator stubs.

    Polygons larger than 100 km2 are rejected (413) per NFR-05 rate limit.
    """
    area_km2 = polygon_area_km2(req.polygon)
    if area_km2 > POLYGON_AREA_CAP_KM2:
        raise HTTPException(
            status_code=413,
            detail=f"Polygon {area_km2:.1f} km2 exceeds "
                   f"{POLYGON_AREA_CAP_KM2:.0f} km2 cap",
        )
    _product_exists(req.from_product_id)
    if req.to_product_id is not None:
        _product_exists(req.to_product_id)
    total_ha = area_km2 * 100.0
    weights = [STUB_AREA_HA[c] for c in CLASSES_6]
    s = sum(weights)
    areas = [{"class": c, "area_ha": round(total_ha * w / s, 2)}
             for c, w in zip(CLASSES_6, weights, strict=True)]
    return {
        "polygon_area_km2": round(area_km2, 3),
        "from_product_id": req.from_product_id,
        "to_product_id": req.to_product_id,
        "areas": areas,
        "indicators": {
            "built_up_growth_ha": 12.5 if req.to_product_id else None,
            "water_extent_ha": round(total_ha * weights[0] / s, 2),
            "degradation_index_pct": 3.1,
        },
        "caveat": "demo proportions scaled to polygon area; "
                  "see model card for accuracy limits",
    }


@router.get("/change")
def change(
    from_id: int = Query(alias="from"),
    to_id: int = Query(alias="to"),
    admin_unit: str | None = Query(default=None),
) -> dict:
    """From-to transition matrix stub (2019 -> 2025)."""
    _product_exists(from_id)
    _product_exists(to_id)
    matrix = [
        {"from_class": "cropland", "to_class": "built_up", "area_ha": 5400.0},
        {"from_class": "grass_shrub", "to_class": "built_up", "area_ha": 2100.0},
        {"from_class": "tree_cover", "to_class": "bare_rocky", "area_ha": 640.0},
        {"from_class": "water", "to_class": "bare_rocky", "area_ha": 180.0},
    ]
    return {"from_product_id": from_id, "to_product_id": to_id,
            "admin_unit": admin_unit, "unit": "ha", "transitions": matrix}


@router.get("/compare")
def compare(
    product_id: int,
    baseline: str = Query(default="dynamic_world"),
) -> dict:
    """Ours vs a global baseline on the same validation points (stub)."""
    _product_exists(product_id)
    if baseline not in BASELINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown baseline {baseline!r}; choose from {list(BASELINES)}",
        )
    return {
        "product_id": product_id,
        "baseline": baseline,
        "metric": "macro_f1",
        "ours": 0.78,
        "theirs": 0.66,
        "delta": 0.12,
        "ci95": [0.04, 0.20],
        "note": "stub; prod computes on independent Hyderabad test points",
    }
