"""EE-side spectral index expressions (mirror of :mod:`geoeco.features.indices`).

The scripts under ``scripts/ee/`` build index bands server-side with
``ee.Image.expression(INDEX_EXPRESSIONS[name], inputs)``. This module owns
those expression strings plus:

  * the Sentinel-2 band each expression variable reads (``S2_INPUT_BANDS``),
  * :func:`supported_indices` — config ``sentinel2.indices`` validation,
  * :func:`evaluate_expression` — an arithmetic-only evaluator so unit tests
    can verify the EXACT expression strings against hand-computed values and
    the NumPy reference implementations **without Earth Engine credentials**
    (test inputs are SYNTHETIC, no network, no data files).

No ``ee`` import at module level: :func:`add_index_bands` imports it lazily.

Optical index bands (NDVI/EVI/MNDWI/NDBI) are clamped to [-1, 1] inside
:func:`add_index_bands`, matching the NumPy reference in
:mod:`geoeco.features.indices` (whose docstring says optical indices are
clipped); ``VV_MINUS_VH`` is deliberately NOT clamped (SAR, no such
guarantee in the reference). The test-only evaluator :func:`evaluate_expression`
returns the RAW expression (no clamp) — tests mirror the production clamp
with ``np.clip(..., -1, 1)`` for optical indices.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Variable names are the expression language; keep them in sync with
# geoeco/features/indices.py docstrings (NDVI/EVI/MNDWI/NDBI + VV-VH dB).
INDEX_EXPRESSIONS: dict[str, str] = {
    "NDVI": "(nir - red) / (nir + red)",
    "EVI": "2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)",
    "MNDWI": "(green - swir1) / (green + swir1)",
    "NDBI": "(swir1 - nir) / (swir1 + nir)",
    "VV_MINUS_VH": "vv - vh",
}

# expression variable -> Sentinel-2 L2A band name (config sentinel2.bands).
S2_INPUT_BANDS: dict[str, dict[str, str]] = {
    "NDVI": {"nir": "B8", "red": "B4"},
    "EVI": {"nir": "B8", "red": "B4", "blue": "B2"},
    "MNDWI": {"green": "B3", "swir1": "B11"},
    "NDBI": {"swir1": "B11", "nir": "B8"},
}

# Whitelist for the test-only evaluator: digits, names, arithmetic, dots.
_SAFE_EXPR = re.compile(r"^[0-9A-Za-z_+\-*/(). ]+$")

# Optical indices clamped to [-1, 1] to match geoeco.features.indices;
# VV_MINUS_VH is a dB difference with no clip in the reference.
CLAMPED_INDICES: frozenset[str] = frozenset(("NDVI", "EVI", "MNDWI", "NDBI"))


def supported_indices(configured: list[str]) -> list[str]:
    """Validate configured index names against :data:`INDEX_EXPRESSIONS`.

    Args:
        configured: e.g. ``data_cfg["sentinel2"]["indices"]``.

    Returns:
        The list unchanged (order preserved).

    Raises:
        ValueError: If any name has no expression (fail loudly, never
            silently drop a configured feature).
    """
    names = [str(n) for n in configured]
    unknown = [n for n in names if n not in INDEX_EXPRESSIONS]
    if unknown:
        raise ValueError(
            f"unknown indices {unknown}; supported: {sorted(INDEX_EXPRESSIONS)}"
        )
    return names


def evaluate_expression(index_or_expr: str, values: dict[str, Any]) -> Any:
    """Evaluate an index expression against SYNTHETIC inputs (tests/offline).

    Arithmetic only: the expression must pass :data:`_SAFE_EXPR` and the
    eval namespace has no builtins. Division-by-zero follows NumPy semantics
    for arrays (inf/nan) — unlike :mod:`geoeco.features.indices`, which
    returns 0.0; tests therefore use non-degenerate inputs when comparing.

    Args:
        index_or_expr: Key of :data:`INDEX_EXPRESSIONS` or a raw expression.
        values: Mapping of variable name -> float or ndarray.

    Returns:
        Scalar float or ndarray, depending on the inputs.

    Raises:
        ValueError: If the expression contains disallowed characters.
    """
    expr = INDEX_EXPRESSIONS.get(index_or_expr, index_or_expr)
    if not _SAFE_EXPR.match(expr):
        raise ValueError(
            f"expression contains disallowed characters: {expr!r}"
        )
    with np.errstate(all="ignore"):
        return eval(expr, {"__builtins__": {}}, dict(values))


def add_index_bands(image: Any, indices: list[str]) -> Any:
    """Append index bands to an ``ee.Image`` (reflectance inputs 0..1).

    Args:
        image: ``ee.Image`` carrying the input bands named in
            :data:`S2_INPUT_BANDS` (B2/B3/B4/B8/B11 ...).
        indices: Configured index names (validate with
            :func:`supported_indices` first).

    Returns:
        ``ee.Image`` with the original bands plus one band per index,
        named after the index (e.g. ``NDVI``).

    Raises:
        KeyError: If an index has no entry in :data:`INDEX_EXPRESSIONS`
            or :data:`S2_INPUT_BANDS`.
    """
    out = image
    for name in indices:
        expr = INDEX_EXPRESSIONS[name]  # KeyError: fail loudly
        inputs = {
            var: image.select(band)
            for var, band in S2_INPUT_BANDS[name].items()
        }
        band = image.expression(expr, inputs).rename(name)
        if name in CLAMPED_INDICES:
            band = band.clamp(-1.0, 1.0)  # matches features.indices clipping
        out = out.addBands(band)
    return out
