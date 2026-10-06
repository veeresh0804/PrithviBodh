"""Labelling protocol: 6 classes + Hyderabad disambiguation rules.

Classes (default 6, config ``classes_default6``):
  water, tree_cover, cropland, built_up, bare_rocky, grass_shrub

Hyderabad rules encoded here (photo-interpretation uses HR basemap + S2):
  R1 granite vs rooftop — spectrally similar bright surfaces; use texture
     + context: high focal-std + rectangular geometry + road adjacency
     -> built_up; smooth sheet / dome + no roads -> bare_rocky.
  R2 seasonal tanks — label by the majority water state across the three
     seasonal composites of the map year (monsoon fill vs summer dry).
  R3 fallow vs bare — multi-season NDVI: max seasonal NDVI > 0.35 with a
     field pattern -> cropland (fallow); persistently low -> bare_rocky.
"""

from __future__ import annotations

from typing import Literal

CLASSES: tuple[str, ...] = (
    "water",
    "tree_cover",
    "cropland",
    "built_up",
    "bare_rocky",
    "grass_shrub",
)

CLASS_IDS: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

FALLOW_NDVI_THR: float = 0.35

Label = Literal["water", "tree_cover", "cropland", "built_up", "bare_rocky", "grass_shrub"]


def validate_label(label: str) -> Label:
    """Validate a label string against the 6-class scheme.

    Args:
        label: Candidate class name.

    Returns:
        The label, typed as :data:`Label`.

    Raises:
        ValueError: If the label is not one of the 6 classes.
    """
    if label not in CLASS_IDS:
        raise ValueError(f"Unknown label {label!r}; expected one of {CLASSES}")
    return label  # type: ignore[return-value]


def resolve_granite_vs_rooftop(focal_std: float, rectangular: bool, near_road: bool) -> Label:
    """Apply Hyd rule R1: granite sheet vs concrete rooftop.

    Args:
        focal_std: Focal texture value (higher = rougher built fabric).
        rectangular: Whether the patch shows rectangular building geometry.
        near_road: Whether adjacent to a road network.

    Returns:
        ``"built_up"`` when texture/structure/context indicate buildings,
        else ``"bare_rocky"``.
    """
    if rectangular and (near_road or focal_std > 8.0):
        return "built_up"
    return "bare_rocky"


def resolve_tank_season(seasonal_water: list[bool]) -> Label:
    """Apply Hyd rule R2: seasonal tanks labelled by majority state.

    Args:
        seasonal_water: Per-season water flags [pre, monsoon, post].

    Returns:
        ``"water"`` if water in the majority of seasons, else the
        caller-typical dry state ``"bare_rocky"`` (field team confirms
        cropland/grass on the bunded bed where appropriate).

    Raises:
        ValueError: If the list is empty.
    """
    if not seasonal_water:
        raise ValueError("seasonal_water must be non-empty")
    return "water" if sum(1 for w in seasonal_water if w) * 2 >= len(seasonal_water) else "bare_rocky"


def resolve_fallow_vs_bare(max_seasonal_ndvi: float, field_pattern: bool) -> Label:
    """Apply Hyd rule R3: fallow cropland vs bare/rocky.

    Args:
        max_seasonal_ndvi: Maximum NDVI across the three seasons.
        field_pattern: Whether bund/plot geometry is visible.

    Returns:
        ``"cropland"`` if vegetated in any season (or field pattern with
        marginal NDVI), else ``"bare_rocky"``.
    """
    if max_seasonal_ndvi > FALLOW_NDVI_THR:
        return "cropland"
    if field_pattern and max_seasonal_ndvi > 0.2:
        return "cropland"
    return "bare_rocky"


def class_id(label: str) -> int:
    """Map a class name to its integer id.

    Args:
        label: One of the 6 class names.

    Returns:
        Integer id (position in :data:`CLASSES`).
    """
    return CLASS_IDS[validate_label(label)]
