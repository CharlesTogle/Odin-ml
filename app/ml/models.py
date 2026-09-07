"""Canonical model artifact classes shared by training and serving.

Historically each artifact class was defined twice: once in the corresponding
`training/scripts/train_*.py` (where it was trained) and again in
`app/models/artifact_classes.py` (where the server needed it to `joblib`/`torch`
unpickle the saved artifact). That duplication required a fragile
`register_artifact_classes()` hack that stamped the classes onto `__main__`.

This module is the single source of truth. The training scripts build and dump
instances using these classes, and the serving layer imports the identical
classes to load them back. Keep class definitions byte-for-byte identical to
what artifacts were serialized as; changing a class here can break loading of
already-trained artifacts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RuleBasedClassifier:
    """Deterministic rule-based PFP classifier with ROC-calibrated thresholds.

    Calibrates CV, obligation ratio, and runway thresholds on training data using
    Youden's J statistic (maximizes TPR - FPR) for each dimension independently.
    """

    def __init__(self):
        self.cv_threshold = 0.50
        self.obl_threshold = 0.60
        self.runway_threshold = 3.0
        self._fitted = False

    def _calibrate_threshold(
        self,
        scores: np.ndarray,
        positive_mask: np.ndarray,
        candidate_range: tuple[float, float] = (0.05, 0.80),
        step: float = 0.01,
    ) -> float:
        thresholds = np.arange(candidate_range[0], candidate_range[1], step)
        best_j = -1.0
        best_threshold = float(np.median(thresholds))

        for t in thresholds:
            predicted_positive = scores >= t
            tp = np.sum(predicted_positive & positive_mask)
            fn = np.sum(~predicted_positive & positive_mask)
            fp = np.sum(predicted_positive & ~positive_mask)
            tn = np.sum(~predicted_positive & ~positive_mask)

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            j = tpr - fpr

            if j > best_j:
                best_j = j
                best_threshold = t

        return float(best_threshold)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        cv_idx = feature_names.index("income_stability_cv")
        obl_idx = feature_names.index("obligation_ratio")

        cv_scores = X[:, cv_idx]
        obl_scores = X[:, obl_idx]

        stable_mask = np.array([label.startswith("Stable") for label in y])
        self.cv_threshold = self._calibrate_threshold(
            cv_scores, stable_mask, candidate_range=(0.05, 0.80), step=0.01
        )

        obligated_mask = np.array(["Obligated" in label for label in y])
        self.obl_threshold = self._calibrate_threshold(
            obl_scores, obligated_mask, candidate_range=(0.20, 0.90), step=0.01
        )

        if "runway_months" in feature_names:
            runway_idx = feature_names.index("runway_months")
            runway_scores = X[:, runway_idx]
            tolerant_mask = np.array(["Tolerant" in label for label in y])
            self.runway_threshold = self._calibrate_threshold(
                runway_scores, tolerant_mask, candidate_range=(1.0, 8.0), step=0.5
            )

        self._fitted = True
        return self

    def predict(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        cv_idx = feature_names.index("income_stability_cv")
        obl_idx = feature_names.index("obligation_ratio")

        cv_scores = X[:, cv_idx]
        obl_scores = X[:, obl_idx]

        if "runway_months" in feature_names:
            runway_idx = feature_names.index("runway_months")
            runway_scores = X[:, runway_idx]
        else:
            runway_scores = np.full(len(cv_scores), 3.0)

        predictions = []
        for cv, obl, runway in zip(cv_scores, obl_scores, runway_scores, strict=True):
            stability = "Stable" if cv < self.cv_threshold else "Variable"
            obligation = "Obligated" if obl > self.obl_threshold else "Flexible"
            tolerance = "Tolerant" if runway >= self.runway_threshold else "At-Risk"
            predictions.append(f"{stability}/{obligation}/{tolerance}")

        return np.array(predictions)

    def get_params(self) -> dict:
        return {
            "cv_threshold": self.cv_threshold,
            "obl_threshold": self.obl_threshold,
            "runway_threshold": self.runway_threshold,
        }


class IQRDetector:
    """Statistical anomaly detector using IQR on each feature."""

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.bounds: dict[int, tuple[float, float]] = {}

    def fit(self, X: np.ndarray) -> IQRDetector:
        for j in range(X.shape[1]):
            col = X[:, j]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            self.bounds[j] = (q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        anomaly_flags = np.zeros(X.shape[0], dtype=float)
        for j in range(X.shape[1]):
            lo, hi = self.bounds[j]
            outlier_mask = (X[:, j] < lo) | (X[:, j] > hi)
            anomaly_flags += outlier_mask.astype(float)
        return anomaly_flags / X.shape[1]


class _Autoencoder(nn.Module):
    """PyTorch autoencoder for anomaly detection (reconstruction error)."""

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


class HybridEnsemble:
    """Ensemble of detectors using score averaging."""

    def __init__(self, detectors: list, weights: list | None = None):
        self.detectors = detectors
        self.weights = weights or [1.0 / len(detectors)] * len(detectors)

    def score(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(X.shape[0], dtype=float)
        for detector, weight in zip(self.detectors, self.weights, strict=True):
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
    """PyTorch sequence forecaster (LSTM/GRU/BiLSTM)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        model_type: str = "gru",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.rnn: nn.Module

        if model_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif model_type == "bilstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
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
