"""From-scratch ResNet18-OSL, CNN-Transformer, and CNN-GNN classifiers for ALL-IDB1."""

__all__ = [
    "HybridCNNGNN",
    "HybridCNNTransformer",
    "OrthogonalSoftmaxLayer",
    "ResNet18OSL",
    "SpatialGraphConvolution",
    "create_model",
]


def __getattr__(name: str):
    if name in __all__:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
