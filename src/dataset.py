"""Load the official NumPy regression arrays."""

from pathlib import Path

import numpy as np
import torch


def load_split(directory: str | Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    root = Path(directory)
    x = torch.as_tensor(np.load(root / f"X_{split}.npy"), dtype=torch.float32)
    y = torch.as_tensor(np.load(root / f"y_{split}.npy"), dtype=torch.float32)
    return x, y
