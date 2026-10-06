"""256x256 chip generation for deep / foundation-model training.

Chips are cut from co-registered (bands, H, W) stacks + matching label
masks. Local fine-tune set: ~100 hand-drawn 256x256 patches (QGIS
polygons rasterised); pretraining streams thousands of 510x510 DW tiles
re-chipped to 256.
"""

from __future__ import annotations

import numpy as np

CHIP_SIZE: int = 256


def generate_chips(
    stack: np.ndarray,
    label: np.ndarray | None = None,
    chip_size: int = CHIP_SIZE,
    stride: int = CHIP_SIZE,
    nodata_frac_thr: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray | None, tuple[int, int]]]:
    """Slice a stack into chips.

    Args:
        stack: Array shaped (bands, H, W) or (H, W).
        label: Optional label mask shaped (H, W).
        chip_size: Chip edge in pixels.
        stride: Stride in pixels (``< chip_size`` gives overlap).
        nodata_frac_thr: Drop chips whose NaN fraction exceeds this
            (checked on the stack; labels ignored for the mask).

    Returns:
        List of (chip, label_chip_or_None, (row_off, col_off)) tuples.
        Partial edge windows are skipped (no padding) so every chip is
        exactly ``chip_size`` square.

    Raises:
        ValueError: On shape mismatch or non-positive sizes.
    """
    if chip_size <= 0 or stride <= 0:
        raise ValueError(f"chip_size/stride must be positive, got {chip_size}/{stride}")
    img = np.asarray(stack)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    if img.ndim != 3:
        raise ValueError(f"stack must be 2-D or 3-D, got {img.shape}")
    _, height, width = img.shape
    mask: np.ndarray | None = None
    if label is not None:
        mask = np.asarray(label)
        if mask.shape != (height, width):
            raise ValueError(f"label shape {mask.shape} != image HW {(height, width)}")
    chips: list[tuple[np.ndarray, np.ndarray | None, tuple[int, int]]] = []
    for roff in range(0, height - chip_size + 1, stride):
        for coff in range(0, width - chip_size + 1, stride):
            chip = img[:, roff : roff + chip_size, coff : coff + chip_size]
            if np.isnan(chip).mean() > nodata_frac_thr:
                continue
            lab_chip = mask[roff : roff + chip_size, coff : coff + chip_size] if mask is not None else None
            chips.append((chip, lab_chip, (roff, coff)))
    return chips
