"""Dataset-specific pretrained CNN + dynamic GNN classifiers for leukemia microscopy."""

__all__ = [
    "HybridCNNGNN",
    "create_model",
]


def __getattr__(name: str):
    if name in __all__:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
