"""Compatibility helpers for global magnitude pruning experiments."""

import numpy as np


def global_magnitude_mask(parameters: list[np.ndarray], sparsity: float) -> list[np.ndarray]:
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    sizes = [parameter.size for parameter in parameters]
    magnitudes = np.concatenate([np.abs(parameter).ravel() for parameter in parameters])
    keep = np.ones(len(magnitudes), dtype=bool)
    keep[np.argsort(magnitudes, kind="stable")[: int(len(magnitudes) * sparsity)]] = False
    return [mask.reshape(parameter.shape) for mask, parameter in zip(np.split(keep, np.cumsum(sizes)[:-1]), parameters, strict=True)]


def apply_masks(parameters, masks):
    return [parameter * mask for parameter, mask in zip(parameters, masks, strict=True)]
