import numpy as np

def global_magnitude_mask(parameters: list[np.ndarray], sparsity: float) -> list[np.ndarray]:
    """Keep the largest-magnitude parameter entries globally."""
    if not 0 <= sparsity < 1: raise ValueError('sparsity must be in [0, 1)')
    sizes = [p.size for p in parameters]
    magnitudes = np.concatenate([np.abs(p).ravel() for p in parameters])
    prune_count = int(np.floor(len(magnitudes) * sparsity))
    # Ranking indices, rather than thresholding, makes the requested sparsity exact even with tied magnitudes.
    keep = np.ones(len(magnitudes), dtype=bool)
    keep[np.argsort(magnitudes, kind='stable')[:prune_count]] = False
    boundaries = np.cumsum(sizes)[:-1]
    return [mask.reshape(param.shape) for mask, param in zip(np.split(keep, boundaries), parameters, strict=True)]

def apply_masks(parameters: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    return [p * mask for p, mask in zip(parameters, masks, strict=True)]
