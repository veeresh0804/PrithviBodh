"""Sentinel-2 SR ingest helpers (GEE).

Source: ``COPERNICUS/S2_SR_HARMONIZED`` masked with
``GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`` (default threshold 0.6,
config ``s2.cloud_thr``). Bands B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
(20 m bands resampled to 10 m). Outputs per season: median composite
+ valid-observation count (for cloud analysis / cloud-stress test).

``earthengine-api`` is imported lazily so the package imports cleanly
without EE credentials.
"""

from __future__ import annotations

from typing import Any

S2_SOURCE: str = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SOURCE: str = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
DEFAULT_CLOUD_THR: float = 0.6
S2_BANDS: tuple[str, ...] = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
CLOUD_BAND: str = "cs"


def build_s2_collection_spec(
    start_date: str, end_date: str, cloud_thr: float = DEFAULT_CLOUD_THR
) -> dict[str, Any]:
    """Return a JSON-serialisable spec of the S2 query (no EE needed).

    Args:
        start_date: ISO start date (inclusive).
        end_date: ISO end date (exclusive).
        cloud_thr: Keep pixels with Cloud Score+ ``cs >= thr`` clear.

    Returns:
        Spec dict for logging/DVC/MLflow.
    """
    if not 0.0 < cloud_thr < 1.0:
        raise ValueError(f"cloud_thr must be in (0,1), got {cloud_thr}")
    return {
        "collection": S2_SOURCE,
        "cloud_collection": CLOUD_SOURCE,
        "bands": list(S2_BANDS),
        "cloud_band": CLOUD_BAND,
        "cloud_thr": cloud_thr,
        "date_range": [start_date, end_date],
        "outputs": ["median", "valid_count"],
    }


def mask_s2_clouds(image: Any, cloud_thr: float = DEFAULT_CLOUD_THR) -> Any:
    """Mask clouds on an S2 SR image joined with Cloud Score+.

    Expects the image to already carry the ``cs`` band (via a
    ``linkCollection``/join on ``system:index`` in the caller).
    Keeps pixels with ``cs >= cloud_thr``.

    Args:
        image: ``ee.Image`` with S2 SR bands + ``cs`` band.
        cloud_thr: Clear-sky threshold (default 0.6).

    Returns:
        Cloud-masked ``ee.Image``.
    """
    return image.updateMask(image.select(CLOUD_BAND).gte(cloud_thr))


def s2_seasonal_composite(collection: Any) -> Any:
    """Median composite + valid-observation count for a season.

    Args:
        collection: Season-filtered, cloud-masked S2 ``ee.ImageCollection``.

    Returns:
        ``ee.Image`` with median bands plus ``valid_count`` band.
    """
    median = collection.select(list(S2_BANDS)).median()
    count = collection.select(S2_BANDS[0]).count().rename("valid_count")
    return median.addBands(count)


def build_s2_collection(
    aoi: Any, start_date: str, end_date: str, cloud_thr: float = DEFAULT_CLOUD_THR
) -> Any:
    """Build a cloud-masked S2 SR ``ee.ImageCollection``.

    Joins ``S2_SR_HARMONIZED`` with Cloud Score+ on ``system:index``,
    applies :func:`mask_s2_clouds`, and filters to ``S2_BANDS``.

    Args:
        aoi: ``ee.Geometry`` of the area of interest.
        start_date: ISO start date.
        end_date: ISO end date.
        cloud_thr: Clear-sky threshold.

    Returns:
        Cloud-masked ``ee.ImageCollection``.

    Raises:
        ImportError: If ``earthengine-api`` is not installed.
    """
    import ee  # lazy

    s2 = (
        ee.ImageCollection(S2_SOURCE)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    )
    cs = (
        ee.ImageCollection(CLOUD_SOURCE).filterBounds(aoi).filterDate(start_date, end_date)
    )
    linked = ee.Join.saveFirst("cloud").apply(s2, cs, ee.Filter.equals("system:index", "system:index"))
    def _attach(img: Any) -> Any:
        cloud_img = ee.Image(img.get("cloud")).select(CLOUD_BAND)
        return ee.Image(img).addBands(cloud_img)

    return ee.ImageCollection(linked).map(_attach).map(lambda img: mask_s2_clouds(img, cloud_thr))
