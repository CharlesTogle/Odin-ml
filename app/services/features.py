from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import ensure_training_scripts_on_path

ensure_training_scripts_on_path()

import feature_engineering as fe  # noqa: E402
import feature_engineering_forecaster as fef  # noqa: E402

EXPENSE_BUCKETS = {
    "food": "food_expense",
    "housing": "housing_expense",
    "transport": "transport_expense",
    "health": "health_expense",
    "education": "education_expense",
}

PFP_FEATURE_COLS = fe.FEATURE_COLUMNS


def _bucket(category: str) -> str:
    key = str(category).strip().lower()
    return EXPENSE_BUCKETS.get(key, "other_expense")


def transactions_to_frame(transactions: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["transaction_type"] = df["transaction_type"].str.lower()
    return df


CATEGORY_COLUMNS = tuple(EXPENSE_BUCKETS.values()) + ("other_expense",)


def build_monthly_summaries(transactions: list[dict]) -> pd.DataFrame:
    df = transactions_to_frame(transactions)
    df["month"] = df["date"].dt.month

    income_mask = df["transaction_type"] == "income"
    expense_mask = df["transaction_type"] == "expense"

    income = df.loc[income_mask].groupby("month")["amount"].sum()
    expense = df.loc[expense_mask].groupby("month")["amount"].sum()

    summaries = pd.DataFrame({
        "month": range(1, 13),
        "persona_id": "serving_user",
        "total_income": income.reindex(range(1, 13)).fillna(0.0).to_numpy(),
        "total_expenses": expense.reindex(range(1, 13)).fillna(0.0).to_numpy(),
    })

    cat_pivot = df.loc[expense_mask].copy()
    cat_pivot["bucket"] = cat_pivot["category"].map(_bucket)
    for col in CATEGORY_COLUMNS:
        bucket_sum = cat_pivot.loc[cat_pivot["bucket"] == col].groupby("month")["amount"].sum()
        summaries[col] = bucket_sum.reindex(range(1, 13)).fillna(0.0).to_numpy()

    summaries["savings"] = (summaries["total_income"] - summaries["total_expenses"]).clip(lower=0.0)
    summaries["debt_payment"] = summaries["total_expenses"] * 0.1
    summaries["transaction_count"] = (
        df.loc[expense_mask].groupby("month").size().reindex(range(1, 13)).fillna(0.0).to_numpy()
    )
    return summaries


def pfp_feature_vector(transactions: list[dict], feature_cols: list[str]) -> np.ndarray:
    """Build the 19-column PFP feature vector for the latest available month."""
    summaries = build_monthly_summaries(transactions)
    active = summaries[summaries["total_expenses"] > 0]
    if active.empty:
        raise ValueError("no expense history available for PFP classification")

    rows = fe.compute_derived_features("serving_user", {}, summaries, None)
    if not rows:
        raise ValueError("no derived features could be computed")

    derived = pd.DataFrame(rows).sort_values("month")
    derived = fe.encode_cyclical(derived, "month", 12)
    derived = fe.compute_interaction_features(derived)

    latest = derived.iloc[-1]
    vector = np.array([latest[c] for c in feature_cols], dtype=float).reshape(1, -1)
    return vector


def forecast_monthly_features(transactions: list[dict], feature_cols: list[str]) -> pd.DataFrame:
    """Build the per-month aggregated feature table for a user (columns = feature_cols)."""
    summaries = build_monthly_summaries(transactions)
    df = transactions_to_frame(transactions)
    df["persona_id"] = "serving_user"
    df["date"] = df["date"].apply(lambda d: d.replace(year=2023))

    grid = fef.process_persona("serving_user", df, summaries, train_imputation=None, is_train=False)
    if grid.empty:
        raise ValueError("no daily grid could be built from transaction history")

    agg_dict = {col: "mean" for col in feature_cols if col in grid.columns}
    monthly = grid.groupby(["persona_id", "month"]).agg(agg_dict).reset_index()
    return monthly.sort_values("month").reset_index(drop=True)


def forecast_last_3_months(transactions: list[dict], feature_cols: list[str]) -> np.ndarray:
    """Flatten the latest 3 months of monthly features into a 69-column vector."""
    monthly = forecast_monthly_features(transactions, feature_cols)
    if len(monthly) < 3:
        raise ValueError("forecast requires at least 3 months of expense history")
    recent = monthly.tail(3)
    return np.array(recent[feature_cols].values.flatten(), dtype=np.float32).reshape(1, -1)


def anomaly_feature_vectors(transactions: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    """Build per-transaction anomaly feature vectors for a single user."""
    import feature_engineering_anomaly as fea

    df = transactions_to_frame(transactions)
    df["persona_id"] = "serving_user"
    df["month"] = df["date"].dt.month
    if "transaction_id" not in df.columns:
        df["transaction_id"] = df.index.astype(str)

    history = fea.build_category_history(df)
    out_rows = []
    for _, row in df.iterrows():
        feats = fea.compute_features_for_persona("serving_user", row, history)
        if feats:
            out_rows.append({**feats, "user_id": "serving_user", "month": int(row["month"])})

    feature_frame = pd.DataFrame(out_rows)
    return out_rows, feature_frame
