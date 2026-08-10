from __future__ import annotations

import sys

import torch.nn as nn


class _Autoencoder(nn.Module):
    """PyTorch autoencoder for anomaly detection (mirrors train_anomaly.py).

    The trained artifact was dumped from `train_anomaly.py`, which pickles the
    class reference under `__main__`. This module re-declares the identical
    class so joblib can unpickle it at serve time.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 7):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def register_artifact_classes() -> None:
    """Register artifact classes under `__main__` for joblib unpickling."""
    module = sys.modules.get("__main__")
    if module is not None and not hasattr(module, "_Autoencoder"):
        setattr(module, "_Autoencoder", _Autoencoder)
