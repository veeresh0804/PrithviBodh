"""Global seed fixing for reproducibility.

Target: reruns from (config + data version + seed) reproduce metrics
within +/-0.5pp (NFR-03). Covers stdlib ``random``, NumPy and
PyTorch (CPU + CUDA) when installed; never fails when torch is absent.
"""

from __future__ import annotations

import os
import random


def fix_seeds(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Fix all relevant RNG seeds.

    Args:
        seed: Integer seed value.
        deterministic_torch: If True, request deterministic cuDNN
            behaviour (may be slower).

    Returns:
        The seed that was set (echo, for logging/MLflow).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Older torch builds; best-effort only.
                pass
    except ImportError:
        pass
    return seed
