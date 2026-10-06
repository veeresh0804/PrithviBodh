"""Sentinel-1 GRD ingest helpers (GEE).

Source: ``COPERNICUS/S1_GRD``, IW mode, VV+VH. Orbit (ASCENDING /
DESCENDING) is a parameter chosen after the week-2 availability check
(config ``s1.orbit``); primary years 2019 (S1A+B) and 2025 (S1C).

``earthengine-api`` is imported lazily inside builder functions so the
package imports without EE credentials. Pure-Python helpers
(``*_spec``) return JSON-serialisable dicts for logging/DVC/MLflow.
"""

from __future__ import annotations

from typing import Any, Literal

S1_SOURCE: str = "COPERNICUS/S1_GRD"
S1_MODE: str = "IW"
S1_POLARISATIONS: tuple[str, str] = ("VV", "VH")

Orbit = Literal["ASCENDING", "DESCENDING"]


def build_s1_collection_spec(
    start_date: str,
    end_date: str,
    orbit: Orbit = "ASCENDING",
    instrument_mode: str = S1_MODE,
) -> dict[str, Any]:
    """Return a JSON-serialisable spec of the S1 query (no EE needed).

    Args:
        start_date: ISO start date (inclusive).
        end_date: ISO end date (exclusive).
        orbit: Orbit pass; set from availability report config.
        instrument_mode: Normally ``"IW"``.

    Returns:
        Spec dict mirroring the EE filter chain.
    """
    return {
        "collection": S1_SOURCE,
        "filters": {
            "instrumentMode": instrument_mode,
            "polarisation": list(S1_POLARISATIONS),
            "orbitPass": orbit,
            "date_range": [start_date, end_date],
        },
        "features": ["VV_dB", "VH_dB", "VV_minus_VH_ratio", "VH_std_seasonal"],
    }


def build_s1_collection(aoi: Any, start_date: str, end_date: str, orbit: Orbit = "ASCENDING") -> Any:
    """Build a filtered S1 GRD ``ee.ImageCollection``.

    Args:
        aoi: ``ee.Geometry`` of the area of interest.
        start_date: ISO start date.
        end_date: ISO end date.
        orbit: Orbit pass to keep (single orbit for consistency).

    Returns:
        Filtered ``ee.ImageCollection`` with VV+VH bands in dB.

    Raises:
        ImportError: If ``earthengine-api`` is not installed.
    """
    import ee  # lazy: needs `earthengine authenticate` beforehand

    spec = build_s1_collection_spec(start_date, end_date, orbit)
    flt = spec["filters"]
    col = (
        ee.ImageCollection(spec["collection"])
        .filterBounds(aoi)
        .filterDate(flt["date_range"][0], flt["date_range"][1])
        .filter(ee.Filter.eq("instrumentMode", flt["instrumentMode"]))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", flt["orbitPass"]))
    )
    return col


def mask_s1_edge_noise(image: Any, buffer_m: float = 500.0) -> Any:
    """Mask S1 scene edge noise (STUB with documented approach).

    Production approach: mask pixels where ``angle`` band is outside the
    valid IW swath range or within ``buffer_m`` of the scene border
    (``image.geometry().buffer(-buffer).difference`` pattern). Kept as a
    pass-through stub here so unit tests run without EE; the EE masking
    expression is applied by the caller notebook/pipeline step.

    Args:
        image: ``ee.Image`` (S1 GRD scene).
        buffer_m: Border buffer in metres.

    Returns:
        The input image (stub); production code chains ``updateMask``.
    """
    # STUB: production one-liner (EE-side):
    #   edge = image.select('angle').gt(31).and(image.select('angle').lt(46))
    #   return image.updateMask(edge)
    _ = buffer_m
    return image


def apply_refined_lee_fallback(image: Any, kernel_m: float = 70.0) -> Any:
    """Speckle filter: Refined Lee with focal-median fallback.

    Native GEE has no Refined Lee operator. Preferred path is an
    external SNAP/PolSAR-corrected export; the in-EE fallback used here
    is a focal-median on linear power (``10^(dB/10)`` -> ``focalMedian``
    -> back to dB), which approximates Lee smoothing while staying
    server-side and quota-safe.

    Args:
        image: ``ee.Image`` with VV/VH in dB.
        kernel_m: Square kernel size in metres (~7 px at 10 m).

    Returns:
        The input image (stub); wire ``focalMedian`` in production.
    """
    # Production EE sketch (per polarisation band, linear domain):
    #   lin = ee.Image(10).pow(img.divide(10))
    #   sm = lin.focalMedian(kernel_m, 'square', 'meters')
    #   return ee.Image(10).multiply(sm.log10()).copyProperties(img)
    _ = kernel_m
    return image


def s1_seasonal_features(collection: Any) -> Any:
    """Derive seasonal S1 features: median VV/VH (dB), ratio, std VH.

    Outputs per season: ``VV`` (median dB), ``VH`` (median dB),
    ``VVVH`` = VV − VH (dB ratio), ``VH_std`` (temporal std of VH,
    captures crop/water dynamics).

    Args:
        collection: Season-filtered, edge-masked S1 ``ee.ImageCollection``.

    Returns:
        Single ``ee.Image`` with bands [VV, VH, VVVH, VH_std].

    Raises:
        ImportError: If ``earthengine-api`` is not installed.
    """
    import ee  # lazy

    vv = collection.select("VV").median().rename("VV")
    vh = collection.select("VH").median().rename("VH")
    ratio = vv.subtract(vh).rename("VVVH")
    vh_std = collection.select("VH").reduce(ee.Reducer.stdDev()).rename("VH_std")
    return vv.addBands([vh, ratio, vh_std])
