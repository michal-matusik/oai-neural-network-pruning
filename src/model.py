"""Exact architecture fixed by the official evaluator."""

from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(nn.Linear(128, 1024), nn.Sigmoid(), nn.Linear(1024, 10))

    def forward(self, x):
        return self.layers(self.flatten(x))
