"""Reproducibility seed helpers."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Seed everything for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

if __name__ == "__main__":
    seed = 42
    sd = seed_everything(seed)
    print(f"Seeded everything with seed: {sd}")
