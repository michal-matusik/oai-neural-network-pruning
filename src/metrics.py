"""Official loss, sparsity, and score conversion."""

import torch


@torch.no_grad()
def mean_squared_error(model, x, target) -> float:
    return float(torch.mean((model(x) - target) ** 2))


def sparsity(model) -> float:
    parameters = list(model.parameters())
    return sum((parameter == 0).sum().item() for parameter in parameters) / sum(parameter.numel() for parameter in parameters)


def task_metric(mse: float, model_sparsity: float) -> float:
    return (1.0 - min(mse, 1000.0) / 1000.0) ** 1.5 * model_sparsity ** 1.5


def task_points(metric: float) -> float:
    return (min(max(metric, 0.085), 0.95) - 0.085) / (0.95 - 0.085) * 100.0
