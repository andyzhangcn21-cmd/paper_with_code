# utils/seed.py

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch
    so that each run with the same seed is reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
