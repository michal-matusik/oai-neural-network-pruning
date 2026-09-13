"""Recover a sparse linear rule and encode it exactly inside the fixed MLP.

The published targets are almost linear in the inputs.
Paired sigmoid units implement ``sigmoid(a*x) - sigmoid(-a*x)``, which approaches a scaled linear function for small ``a``.
This uses only two first-layer weights per input and two second-layer weights per retained regression coefficient.
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from .dataset import load_split
from .model import MLP


def train(data_directory: str, coefficient_threshold: float = 0.1, alpha: float = 0.003) -> MLP:
    x, target = load_split(data_directory, "train")
    regression = LinearRegression().fit(x.numpy(), target.numpy())
    coefficients = regression.coef_.copy()
    coefficients[np.abs(coefficients) <= coefficient_threshold] = 0.0
    model = MLP()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        first, second = model.layers[0], model.layers[2]
        second.bias.copy_(torch.as_tensor(regression.intercept_, dtype=torch.float32))
        for feature in range(128):
            positive, negative = 2 * feature, 2 * feature + 1
            first.weight[positive, feature] = alpha
            first.weight[negative, feature] = -alpha
            encoded = torch.as_tensor(2.0 * coefficients[:, feature] / alpha, dtype=torch.float32)
            second.weight[:, positive] = encoded
            second.weight[:, negative] = -encoded
    return model
