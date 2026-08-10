from __future__ import annotations

import numpy as np

from app.models.registry import ModuleModel
from app.schemas.forecast import (
    ConfidenceInterval,
    ForecastLevel,
    ForecastPoint,
    ForecastRequest,
)
from app.services.features import forecast_last_3_months, transactions_to_frame


def _predict_monthly_total(model: ModuleModel, transactions: list[dict]) -> tuple[float, dict]:
    vector = forecast_last_3_months(transactions, model.feature_columns)
    artifact = model.model
    scaled = artifact["scaler"].transform(vector)
    trees = np.array([t.predict(scaled) for t in artifact["model"].estimators_]).flatten()

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


def forecast(model: ModuleModel, request: ForecastRequest) -> tuple[list[ForecastPoint], ConfidenceInterval, str]:
    total, ci = _predict_monthly_total(model, [t.model_dump() for t in request.historical_transactions])
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
            points.append(ForecastPoint(date="next", amount=round(predicted * share, 2), category=group))
        return points, interval, "category_group"

    points = []
    for cat, share in proportions.items():
        points.append(ForecastPoint(date="next", amount=round(predicted * share, 2), category=cat))
    points.sort(key=lambda p: p.amount, reverse=True)
    return points, interval, "category"


def forecast_or_fallback(model: ModuleModel, request: ForecastRequest) -> tuple[list[ForecastPoint], ConfidenceInterval, str]:
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
