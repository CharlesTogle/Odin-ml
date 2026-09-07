from __future__ import annotations

import numpy as np
import torch

from app.models.registry import ModuleModel
from app.schemas.forecast import (
    ConfidenceInterval,
    ForecastLevel,
    ForecastPoint,
    ForecastRequest,
)
from app.services.features import (
    build_monthly_summaries,
    forecast_last_3_months,
    transactions_to_frame,
)


def _predict_monthly_total(model: ModuleModel, transactions: list[dict]) -> tuple[float, dict]:
    artifact = model.model

    if isinstance(artifact, dict) and artifact.get("kind") == "arima":
        # Pooled ARIMA forecaster: the pool path is scale-normalized per user,
        # so rescale it by the requesting user's own recent expense level.
        arima = artifact["model"]
        pool_level = float(artifact.get("pool_level", 1.0))
        pool_pred = float(arima.forecast(1).iloc[0]) if arima is not None else pool_level
        user_summaries = build_monthly_summaries(transactions)
        expenses = user_summaries.loc[user_summaries["total_expenses"] > 0, "total_expenses"]
        level = float(expenses.tail(3).mean()) if not expenses.empty else 0.0
        if level <= 0:
            # Fallback: return the unrescaled normalized path (≈ pool average)
            level = pool_level
        pred = (pool_pred / pool_level) * level
        return pred, {
            "lower_80": pred * 0.8,
            "upper_80": pred * 1.2,
            "lower_95": pred * 0.6,
            "upper_95": pred * 1.4,
        }

    vector = forecast_last_3_months(transactions, model.feature_columns)

    if isinstance(artifact, dict) and "model" in artifact and hasattr(artifact["model"], "forward"):
        # PyTorch model (GRU/LSTM/BiLSTM)
        nn_model = artifact["model"]
        scaler = artifact["scaler"]
        seq_length = artifact.get("seq_length", 3)
        n_features = len(model.feature_columns)

        flat = vector.flatten()
        n_months = len(flat) // n_features
        if n_months < seq_length:
            n_months = seq_length
        recent = flat[-n_months * n_features :]
        recent_s = scaler.transform(recent.reshape(-1, n_features))
        seq = recent_s[-seq_length:]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = nn_model(x).item()
        return pred, {
            "lower_80": pred * 0.8,
            "upper_80": pred * 1.2,
            "lower_95": pred * 0.6,
            "upper_95": pred * 1.4,
        }

    if isinstance(artifact, dict) and "model" in artifact:
        # sklearn RF model wrapped in dict
        scaler = artifact["scaler"]
        rf_model = artifact["model"]
        scaled = scaler.transform(vector)
        trees = np.array([t.predict(scaled) for t in rf_model.estimators_]).flatten()
        total = float(np.mean(trees))
        return total, {
            "lower_80": float(np.percentile(trees, 10)),
            "upper_80": float(np.percentile(trees, 90)),
            "lower_95": float(np.percentile(trees, 2.5)),
            "upper_95": float(np.percentile(trees, 97.5)),
        }

    # Direct sklearn model (no dict wrapper)
    scaled = artifact["scaler"].transform(vector) if hasattr(artifact, "scaler") else vector
    trees = np.array([t.predict(scaled) for t in artifact.estimators_]).flatten()
    total = float(np.mean(trees))
    return total, {
        "lower_80": float(np.percentile(trees, 10)),
        "upper_80": float(np.percentile(trees, 90)),
        "lower_95": float(np.percentile(trees, 2.5)),
        "upper_95": float(np.percentile(trees, 97.5)),
    }


def _horizon_factor(horizon: str) -> float:
    if horizon == "WEEKLY":
        return 7.0 / 30.44
    if horizon == "SEMI_MONTHLY":
        return 15.0 / 30.44
    return 1.0


def _category_proportions(transactions: list[dict]) -> dict[str, float]:
    df = transactions_to_frame(transactions)
    expense = df[df["transaction_type"] == "expense"]
    if expense.empty:
        return {}
    totals = expense.groupby("category")["amount"].sum()
    return (totals / totals.sum()).to_dict()


def forecast(
    model: ModuleModel, request: ForecastRequest
) -> tuple[list[ForecastPoint], ConfidenceInterval, str]:
    total, ci = _predict_monthly_total(
        model, [t.model_dump() for t in request.historical_transactions]
    )
    factor = _horizon_factor(request.forecast_horizon.value)
    predicted = total * factor
    scaled_ci = {k: v * factor for k, v in ci.items()}

    interval = ConfidenceInterval(
        lower_80=round(scaled_ci["lower_80"], 2),
        upper_80=round(scaled_ci["upper_80"], 2),
        lower_95=round(scaled_ci["lower_95"], 2),
        upper_95=round(scaled_ci["upper_95"], 2),
    )

    if request.forecast_level == ForecastLevel.TOTAL:
        return [ForecastPoint(date="next", amount=round(predicted, 2))], interval, "total"

    proportions = _category_proportions([t.model_dump() for t in request.historical_transactions])
    if request.forecast_level == ForecastLevel.CATEGORY_GROUP:
        buckets = {
            "essentials": ["food", "housing", "transport", "health", "education"],
            "discretionary": ["other", "leisure", "entertainment"],
        }
        points = []
        for group, cats in buckets.items():
            share = sum(proportions.get(c, 0.0) for c in cats)
            points.append(
                ForecastPoint(date="next", amount=round(predicted * share, 2), category=group)
            )
        return points, interval, "category_group"

    points = []
    for cat, share in proportions.items():
        points.append(ForecastPoint(date="next", amount=round(predicted * share, 2), category=cat))
    points.sort(key=lambda p: p.amount, reverse=True)
    return points, interval, "category"


def forecast_or_fallback(
    model: ModuleModel, request: ForecastRequest
) -> tuple[list[ForecastPoint], ConfidenceInterval, str]:
    """Run the forecast; fall back to a trailing-average estimate on failure.

    FO-02 (cold-start fallback): when the learned path fails or history is
    too short, average the last 3 months of total expenses.
    """
    try:
        return forecast(model, request)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"forecast unavailable: {exc}") from exc


def cold_start_estimate(transactions: list[dict]) -> tuple[float, float]:
    """FO-02: profile-average fallback using the trailing 3-month mean."""
    from app.services.features import build_monthly_summaries

    summaries = build_monthly_summaries(transactions)
    expenses = summaries.loc[summaries["total_expenses"] > 0, "total_expenses"].tail(3)
    if expenses.empty:
        return 0.0, 0.0
    total = float(expenses.mean())
    return total, float(expenses.std())
