"""Trainable graph model and evaluation tools for AE Arch Engine."""

__all__ = ["GraphSAGEClassifier", "evaluate", "load_checkpoint", "train"]


def __getattr__(name):
    if name in __all__:
        from .ae_graph_train import GraphSAGEClassifier, evaluate, load_checkpoint, train
        return locals()[name]
    raise AttributeError(name)
