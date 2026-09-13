"""Evaluate on the official validation arrays."""

import argparse

from .dataset import load_split
from .metrics import mean_squared_error, sparsity, task_metric, task_points
from .train import train


def evaluate(data_directory: str):
    model = train(data_directory)
    x, target = load_split(data_directory, "valid")
    mse = mean_squared_error(model, x, target)
    sparse = sparsity(model)
    metric = task_metric(mse, sparse)
    return {"mse": mse, "sparsity": sparse, "task_metric": metric, "estimated_percent": task_points(metric), "samples": len(x)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory")
    print(evaluate(parser.parse_args().data_directory))
