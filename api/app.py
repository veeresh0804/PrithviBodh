"""FastAPI entrypoint: /health, CORS, routers, keyed admin job trigger."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth import require_admin_key
from api.routers import models_card, products, stats

app = FastAPI(title="GeoEco Land-Cover API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo UI; tighten in prod
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(products.router)
app.include_router(stats.router)
app.include_router(models_card.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/regions")
def regions() -> list[dict]:
    return [{"id": 1, "name": "hyderabad",
             "note": "60x60km demo AOI, EPSG:32644"}]


@app.post("/admin/jobs/inference", dependencies=[Depends(require_admin_key)])
def trigger_inference(product_type: str = "lc", year: int = 2025,
                      season: str = "post") -> dict:
    """Keyed admin hook: enqueue a tiled-inference job (stub)."""
    return {"queued": True, "product_type": product_type,
            "year": year, "season": season}
