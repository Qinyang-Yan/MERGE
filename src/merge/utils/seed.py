"""Reproducibility helpers for Python, NumPy, PyTorch, and DataLoader workers."""

import random


def set_global_seed(seed: int) -> None:
    """Seed supported random-number generators and deterministic cuDNN selection."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed worker-local Python and NumPy generators from PyTorch's worker seed."""
    import numpy as np
    import torch

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
