from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from app.models.artifact_classes import _Autoencoder
from app.models.registry import ModuleModel
from app.schemas.anomaly import AnomalyRequest, AnomalyResult
from app.services.features import anomaly_feature_vectors

ANOMALY_FEATURE_COLS = [
    "mean_income_rolling", "std_income_rolling",
    "mean_expenses_rolling", "std_expenses_rolling",
    "category_dist", "txn_frequency_rolling", "avg_txn_size_rolling",
    "category_entropy", "volatility_index", "spending_concentration",
    "amount_deviation", "category_deviation", "frequency_deviation",
    "income_deviation", "expense_deviation", "is_novel_category",
    "amount_vs_category_mean", "amount_vs_category_std",
    "category_frequency_change", "amount_percentile_in_category",
    "days_since_last_txn", "is_weekend", "amount_zscore_overall", "amount_zscore_category",
]

DEFAULT_THRESHOLD = 0.0117


def _score(module: ModuleModel, X: np.ndarray) -> np.ndarray:
    model = module.model
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            recon = model(torch.tensor(X, dtype=torch.float32)).numpy()
        return np.mean((X - recon) ** 2, axis=1)
    if hasattr(model, "score"):
        return np.asarray(model.score(X), dtype=float)
    if hasattr(model, "decision_function"):
        scores = -model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores
    raise TypeError(f"unsupported anomaly detector type: {type(model)}")


def _normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-8:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def _explain(transaction: dict, score: float, threshold: float) -> list[str]:
    reasons = []
    amount = float(transaction["amount"])
    if score >= threshold:
        reasons.append("Reconstruction error above the detection threshold")
    category = str(transaction.get("category", "unknown"))
    if category not in {"food", "housing", "transport", "health", "education", "other"}:
        reasons.append("Unusual merchant category")
    if amount > 2000.0:
        reasons.append("Transaction amount exceeds typical single-spend size")
    return reasons or ["No anomaly signals detected"]


def detect(module: ModuleModel, request: AnomalyRequest) -> list[AnomalyResult]:
    """Score each transaction; target = the most recent transaction.

    The trained detector emits anomaly scores per feature vector; the serving
    endpoint returns one result per submitted transaction, marking the newest
    one as the detection target (consistent with the on-transaction trigger).
    """
    transactions = [t.model_dump() for t in request.transactions]
    _, feature_frame = anomaly_feature_vectors(transactions)
    if feature_frame.empty:
        raise ValueError("anomaly detection requires baseline transaction history")

    X = feature_frame[ANOMALY_FEATURE_COLS].values.astype(np.float32)
    scores = _normalize(_score(module, X))

    results = []
    for i, txn in enumerate(transactions):
        score = float(scores[i]) if i < len(scores) else 0.0
        threshold = DEFAULT_THRESHOLD
        results.append(AnomalyResult(
            transaction_id=str(txn.get("transaction_id") or i),
            is_anomalous=score >= threshold,
            score=round(score, 4),
            threshold=threshold,
            explanation=_explain(txn, score, threshold),
        ))
    return results
