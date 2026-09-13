"""Model serialization compatible with the official evaluator."""

import pickle


def save_parameters(model, path: str = "model_parameters.pkl") -> None:
    with open(path, "wb") as output:
        pickle.dump({name: parameter.detach().cpu() for name, parameter in model.named_parameters()}, output)
