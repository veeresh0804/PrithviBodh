"""Shared geographic statistics (haversine + audit-A3 proxy table).

Single source of truth for distance-to-centre math, committed to git so CI
(fresh clones without the gitignored `data/` tree) can use it. Both
`geoeco.labels.seed_check` and `data/labels/audit/collect.py` import from
here. No network, no benchmark reads, no label writes — coordinates only.
"""
from __future__ import annotations

import math
from collections import Counter

R_KM = 6371.0
CENTER = (78.48, 17.38)  # hyderabad.yaml center


def hav(lon: float, lat: float) -> float:
    """Great-circle distance (km) from (lon, lat) to CENTER."""
    lonr, latr = math.radians(lon), math.radians(lat)
    lon0r, lat0r = math.radians(CENTER[0]), math.radians(CENTER[1])
    h = (math.sin((latr - lat0r) / 2) ** 2 + math.cos(lat0r)
         * math.cos(latr) * math.sin((lonr - lon0r) / 2) ** 2)
    return 2 * R_KM * math.asin(math.sqrt(h))


def summarize(features: list[dict]) -> tuple[Counter, Counter, dict]:
    """Audit-A3 proxy table: per-split mean distance-to-centre + mean lon/lat.

    Args:
        features: GeoJSON point features with
            `properties.split` ("test"/"train"), `properties.block`,
            optional `properties.fine_block`, and Point `geometry.coordinates`.

    Returns:
        (by_split_block, by_fine, dist) where dist[split] holds n,
        mean_dist_km (2 dp), mean_lon and mean_lat (3 dp).
    """
    by_split_block = Counter(
        (f["properties"]["split"], f["properties"]["block"]) for f in features)
    by_fine = Counter(
        (f["properties"]["split"], f["properties"].get("fine_block")) for f in features)
    dist = {}
    for split in ("test", "train"):
        pts = [f["geometry"]["coordinates"] for f in features if f["properties"]["split"] == split]
        ds = [hav(x, y) for x, y in pts]
        dist[split] = {"n": len(pts), "mean_dist_km": round(sum(ds) / len(ds), 2),
                       "mean_lon": round(sum(x for x, _ in pts) / len(pts), 3),
                       "mean_lat": round(sum(y for _, y in pts) / len(pts), 3)}
    return by_split_block, by_fine, dist
