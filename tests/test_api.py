"""API contract tests: routes, polygon cap, error handling."""

from __future__ import annotations

import os

os.environ.setdefault("ADMIN_API_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402

client = TestClient(app)

SMALL_POLY = {
    "type": "Polygon",
    "coordinates": [[[78.40, 17.35], [78.42, 17.35],
                      [78.42, 17.37], [78.40, 17.37],
                      [78.40, 17.35]]],
}

# ~2x2 degrees >> 100 km2 cap
BIG_POLY = {
    "type": "Polygon",
    "coordinates": [[[77.0, 16.0], [79.0, 16.0],
                      [79.0, 18.0], [77.0, 18.0],
                      [77.0, 16.0]]],
}


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_products_filter() -> None:
    r = client.get("/products", params={"year": 2025})
    assert r.status_code == 200
    assert all(p["year"] == 2025 for p in r.json())


def test_product_404() -> None:
    assert client.get("/products/999").status_code == 404


def test_tile_redirect() -> None:
    r = client.get("/tiles/2/10/5/5.png", follow_redirects=False)
    assert r.status_code == 307
    assert "colormap_name=" in r.headers["location"]


def test_stats_contract() -> None:
    body = client.get("/stats/2").json()
    assert {a["class"] for a in body["areas"]} == {
        "water", "tree_cover", "cropland",
        "built_up", "bare_rocky", "grass_shrub",
    }


def test_analyze_ok() -> None:
    r = client.post("/analyze", json={
        "polygon": SMALL_POLY, "from_product_id": 1, "to_product_id": 2})
    assert r.status_code == 200
    assert r.json()["polygon_area_km2"] < 100


def test_analyze_cap() -> None:
    r = client.post("/analyze", json={
        "polygon": BIG_POLY, "from_product_id": 1, "to_product_id": 2})
    assert r.status_code == 413


def test_analyze_bad_polygon() -> None:
    bad = {"type": "Polygon",
           "coordinates": [[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]}  # unclosed/short
    assert client.post("/analyze", json={
        "polygon": bad, "from_product_id": 1}).status_code == 422


def test_change_and_compare() -> None:
    assert client.get("/change", params={"from": 1, "to": 2}
                      ).json()["transitions"]
    r = client.get("/compare",
                   params={"product_id": 2, "baseline": "dynamic_world"})
    assert r.json()["delta"] > 0
    assert client.get("/compare", params={
        "product_id": 2, "baseline": "nope"}).status_code == 400


def test_model_card_and_download() -> None:
    assert "test_macro_f1" in client.get("/models/1/card").json()
    assert client.get("/models/999/card").status_code == 404
    assert client.get("/download/2").json()["filename"].endswith(".tif")


def test_admin_keyed() -> None:
    assert client.post("/admin/jobs/inference").status_code == 401
    r = client.post("/admin/jobs/inference",
                    headers={"X-API-Key": "test-key"})
    assert r.json()["queued"] is True
