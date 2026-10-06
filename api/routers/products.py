"""Product catalog + tile proxy + download routes (public, read-only)."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from api.db import STUB_PRODUCTS

router = APIRouter(tags=["products"])

COLORMAPS: dict[str, str] = {
    "lc": "lc6",
    "confidence": "viridis",
    "ndvi": "rdylgn",
    "water": "blues",
    "change": "spectral",
}


def _find(product_id: int) -> dict:
    for p in STUB_PRODUCTS:
        if p["id"] == product_id:
            return p
    raise HTTPException(status_code=404, detail=f"Product {product_id} not found")


@router.get("/products")
def list_products(
    region: str | None = Query(default=None),
    product_type: str | None = Query(default=None, alias="type"),
    year: int | None = Query(default=None),
    season: str | None = Query(default=None),
) -> list[dict]:
    """GET /products?region=&type=&year=&season= — filter the map catalog."""
    out = STUB_PRODUCTS
    if region:
        out = [p for p in out if p["region"] == region]
    if product_type:
        out = [p for p in out if p["product_type"] == product_type]
    if year is not None:
        out = [p for p in out if p["year"] == year]
    if season:
        out = [p for p in out if p["season"] == season]
    return out


@router.get("/products/{product_id}")
def get_product(product_id: int) -> dict:
    return _find(product_id)


@router.get("/tiles/{product_id}/{z}/{x}/{y}.png")
def tile(
    product_id: int,
    z: int,
    x: int,
    y: int,
) -> RedirectResponse:
    """Tile proxy stub: 307-redirect to TiTiler for the product COG.

    Prod wires TiTiler with the project colormap; the redirect keeps the
    public URL stable (/tiles/...) while TiTiler does the rendering.
    """
    product = _find(product_id)
    base = os.environ.get("TITILER_URL", "http://titiler:8001").rstrip("/")
    cmap = COLORMAPS.get(product["product_type"], "viridis")
    url = (
        f"{base}/cog/tiles/{z}/{x}/{y}.png"
        f"?url={product['cog_url']}&colormap_name={cmap}"
    )
    return RedirectResponse(url=url, status_code=307)


@router.get("/download/{product_id}")
def download(product_id: int) -> dict:
    """Download stub: returns the COG location + suggested filename.

    (Large-file streaming / presigned R2 URLs are wired in prod; the JSON
    contract — product_id, filename, cog_url — is stable.)
    """
    product = _find(product_id)
    fname = (
        f"{product['region']}_{product['year']}_{product['season']}_"
        f"{product['product_type']}.tif"
    )
    return {"product_id": product_id, "filename": fname,
            "cog_url": product["cog_url"]}
