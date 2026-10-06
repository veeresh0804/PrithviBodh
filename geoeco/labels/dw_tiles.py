"""Dynamic World training tiles: PANGAEA download + 9 -> 6 remap (STUB I/O).

Dense labels for pretraining segmentation come from the DW training
dataset (PANGAEA DOI 10.1594/PANGAEA.933475, CC BY-4.0): ~24k tiles of
510x510 px @10 m with image IDs only (no imagery) — matching S2 (+
nearest S1) is re-pulled from EE. This module holds the crosswalk and
remap; network I/O is an explicit stub so imports stay offline-safe.

DW 9 classes -> project 6:
  water -> water; trees -> tree_cover; crops -> cropland;
  built -> built_up; bare -> bare_rocky; snow_and_ice -> bare_rocky
  (rare/absent in Hyderabad; kept to keep the map total);
  grass + flooded_vegetation + shrub_and_scrub -> grass_shrub.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DW_CLASSES_9: tuple[str, ...] = (
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
)

DW_IDS: dict[str, int] = {c: i for i, c in enumerate(DW_CLASSES_9)}

# DW id (0-8) -> project class name (one of the 6).
DW_TO_SIX: dict[int, str] = {
    0: "water",
    1: "tree_cover",
    2: "grass_shrub",
    3: "grass_shrub",
    4: "cropland",
    5: "grass_shrub",
    6: "built_up",
    7: "bare_rocky",
    8: "bare_rocky",
}

SIX_IDS: dict[str, int] = {
    "water": 0,
    "tree_cover": 1,
    "cropland": 2,
    "built_up": 3,
    "bare_rocky": 4,
    "grass_shrub": 5,
}


def remap_dw_to_six(dw_label: np.ndarray) -> np.ndarray:
    """Remap a DW 9-class id array to project 6-class ids.

    Args:
        dw_label: Integer array with values 0-8 (DW ids).

    Returns:
        Integer array with values 0-5 (project ids).

    Raises:
        ValueError: If any input id is outside 0-8.
    """
    arr = np.asarray(dw_label, dtype=int)
    if arr.size and (arr.min() < 0 or arr.max() > 8):
        raise ValueError(f"DW ids must be in [0,8], got range [{arr.min()},{arr.max()}]")
    lut = np.array([SIX_IDS[DW_TO_SIX[i]] for i in range(9)], dtype=np.int64)
    return lut[arr]


def download_pangaea_tile_stub(tile_id: str, out_dir: str | Path) -> Path:
    """Locate where a PANGAEA tile would be cached (STUB, no network).

    Real step (notebook/pipeline): query PANGAEA DOI
    10.1594/PANGAEA.933475 for ``tile_id``, download the 510x510 label
    GeoTIFF, attribute CC BY-4.0.

    Args:
        tile_id: DW tile identifier.
        out_dir: Local cache directory.

    Returns:
        Expected local path (file is NOT downloaded by this stub).
    """
    return Path(out_dir) / f"{tile_id}_label.tif"
