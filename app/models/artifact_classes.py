from __future__ import annotations

import sys

import numpy as np
import torch
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


class IQRDetector:
    """Statistical anomaly detector using IQR on each feature.

    Mirrors train_anomaly.py so joblib can unpickle the saved artifact.
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.bounds = {}

    def fit(self, X: np.ndarray):
        for j in range(X.shape[1]):
            col = X[:, j]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            self.bounds[j] = (
                q1 - self.iqr_multiplier * iqr,
                q3 + self.iqr_multiplier * iqr,
            )

    def score(self, X: np.ndarray) -> np.ndarray:
        anomaly_flags = np.zeros(X.shape[0], dtype=float)
        for j in range(X.shape[1]):
            lo, hi = self.bounds[j]
            outlier_mask = (X[:, j] < lo) | (X[:, j] > hi)
            anomaly_flags += outlier_mask.astype(float)
        return anomaly_flags / X.shape[1]


class HybridEnsemble:
    """Ensemble of Tier 1-2 detectors using score averaging.

    Mirrors train_anomaly.py so joblib can unpickle the saved artifact.
    """

    def __init__(self, detectors: list, weights: list = None):
        self.detectors = detectors
        self.weights = weights or [1.0 / len(detectors)] * len(detectors)

    def score(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(X.shape[0], dtype=float)
        for detector, weight in zip(self.detectors, self.weights):
            if hasattr(detector, "score"):
                s = detector.score(X)
            elif hasattr(detector, "decision_function"):
                s = -detector.decision_function(X)
                s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            elif isinstance(detector, nn.Module):
                detector.eval()
                with torch.no_grad():
                    X_t = torch.tensor(X, dtype=torch.float32)
                    recon = detector(X_t).numpy()
                s = np.mean((X - recon) ** 2, axis=1)
                s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            else:
                continue
            scores += weight * s
        return scores


class _SequenceForecaster(nn.Module):
    """PyTorch sequence forecaster (LSTM/GRU/BiLSTM).

    Mirrors train_forecaster.py so torch.load can reconstruct the model.
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 model_type: str = "gru", dropout: float = 0.2):
        super().__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size

        if model_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif model_type == "bilstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True,
                               bidirectional=True)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.dropout = nn.Dropout(dropout)
        rnn_out_size = hidden_size * 2 if model_type == "bilstm" else hidden_size
        self.head = nn.Sequential(
            nn.Linear(rnn_out_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last = rnn_out[:, -1, :]
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


def register_artifact_classes() -> None:
    """Register artifact classes under `__main__` for joblib unpickling."""
    module = sys.modules.get("__main__")
    if module is None:
        return
    for name, cls in [
        ("_Autoencoder", _Autoencoder),
        ("IQRDetector", IQRDetector),
        ("HybridEnsemble", HybridEnsemble),
    ]:
        if not hasattr(module, name):
            setattr(module, name, cls)
