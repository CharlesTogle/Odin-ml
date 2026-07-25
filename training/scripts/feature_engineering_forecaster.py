"""
Feature Engineering Pipeline for LSTM Spending Forecaster

Reads raw transactions and monthly summaries from synth/ and produces
daily-level feature matrices for time-series forecasting.

Features (23 total per persona-day):
  - Temporal encoding: day-of-week sin/cos, day-of_month
  - Lag features: 1d, 7d, 14d, 15d, 30d, 60d expense lags
  - Rolling statistics: 7d, 14d, 30d mean and std of expenses
  - Calendar features: is_payday, days_to_payday
  - RFM features: recency, frequency_30d, monetary_30d
  - STL decomposition: trend, seasonal, residual (monthly level)

Usage:
    python scripts/feature_engineering_forecaster.py \
        --transactions synth/transactions.parquet \
        --summaries synth/monthly_summaries.parquet \
        --splits datasets/processed/split_metadata.json \
        --output datasets/forecaster/

Design principles:
  - Features computed incrementally (no future data leakage)
  - Imputation fit on train only, applied to val/test
  - Persona-level splitting preserved from preprocessor.py
"""

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import periodogram


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECASTER_FEATURES = [
    # Temporal encoding (3)
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_month",
    # Lag features (6)
    "lag_1d",
    "lag_7d",
    "lag_14d",
    "lag_15d",
    "lag_30d",
    "lag_60d",
    # Rolling statistics (6)
    "rolling_mean_7d",
    "rolling_std_7d",
    "rolling_mean_14d",
    "rolling_std_14d",
    "rolling_mean_30d",
    "rolling_std_30d",
    # Calendar features (2)
    "is_payday",
    "days_to_payday",
    # RFM features (3)
    "recency",
    "frequency_30d",
    "monetary_30d",
    # STL decomposition (3)
    "stl_trend",
    "stl_seasonal",
    "stl_residual",
]

META_COLUMNS = [
    "user_id",
    "date",
    "month",
    "year",
    "target_expenses",
    "has_transaction",
]

RAW_EXPENSE_COLUMNS = [
    "food_expense",
    "housing_expense",
    "transport_expense",
    "health_expense",
    "education_expense",
    "other_expense",
]


@dataclass
class ForecasterEngineeringConfig:
    input_transactions: str = "synth/transactions.parquet"
    input_summaries: str = "synth/monthly_summaries.parquet"
    input_splits: str = "datasets/processed/split_metadata.json"
    output_dir: str = "datasets/forecaster/"
    seed: int = 42


@dataclass
class ForecasterEngineeringReport:
    timestamp: str = ""
    n_personas: int = 0
    n_features: int = 0
    n_rows: dict = field(default_factory=dict)
    feature_stats: dict = field(default_factory=dict)
    imputation_values: dict = field(default_factory=dict)
    split_sizes: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_splits(splits_path: str) -> dict:
    with open(splits_path) as f:
        meta = json.load(f)
    return meta.get("personas", {})


def load_transactions(transactions_path: str) -> pd.DataFrame:
    df = pd.read_parquet(transactions_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_summaries(summaries_path: str) -> pd.DataFrame:
    return pd.read_parquet(summaries_path)


# ---------------------------------------------------------------------------
# Daily Grid Construction
# ---------------------------------------------------------------------------

def build_daily_grid(persona_id: str, year: int = 2023) -> pd.DataFrame:
    """Create a full daily grid for one persona across all 12 months."""
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    dates = pd.date_range(start, end, freq="D")
    grid = pd.DataFrame({
        "persona_id": persona_id,
        "date": dates,
        "month": dates.month,
        "year": dates.year,
        "day_of_week": dates.dayofweek,
        "day_of_month": dates.day,
    })
    return grid


def aggregate_daily_expenses(transactions: pd.DataFrame, persona_id: str) -> pd.DataFrame:
    """Aggregate expense transactions to daily level for one persona."""
    mask = (transactions["persona_id"] == persona_id) & (transactions["transaction_type"] == "expense")
    persona_txns = transactions[mask].copy()
    if persona_txns.empty:
        return pd.DataFrame(columns=["date", "daily_expense", "txn_count"])

    daily = persona_txns.groupby("date").agg(
        daily_expense=("amount", "sum"),
        txn_count=("transaction_id", "count"),
    ).reset_index()
    return daily


# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------

def compute_temporal_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week sin/cos encoding and normalized day-of-month."""
    dow = grid["day_of_week"].values
    grid["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    grid["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)
    grid["day_of_month"] = grid["day_of_month"].values / 31.0
    return grid


def compute_lag_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Lag features: expense amount at various lookback periods."""
    exp = grid["daily_expense"].values
    n = len(exp)

    for lag_name, lag_days in [("lag_1d", 1), ("lag_7d", 7), ("lag_14d", 14),
                                ("lag_15d", 15), ("lag_30d", 30), ("lag_60d", 60)]:
        lagged = np.full(n, np.nan)
        if n > lag_days:
            lagged[lag_days:] = exp[:n - lag_days]
        grid[lag_name] = lagged
    return grid


def compute_rolling_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and std over 7, 14, 30 day windows."""
    exp = grid["daily_expense"]

    for window in [7, 14, 30]:
        roll_mean = exp.rolling(window=window, min_periods=1).mean()
        roll_std = exp.rolling(window=window, min_periods=2).std()
        grid[f"rolling_mean_{window}d"] = roll_mean
        grid[f"rolling_std_{window}d"] = roll_std.fillna(0.0)
    return grid


def compute_calendar_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Payday indicators and days-to-payday."""
    dom = grid["day_of_month"].values * 31.0  # un-normalize
    grid["is_payday"] = ((dom >= 15) & (dom <= 16)) | ((dom >= 29) & (dom <= 31))
    grid["is_payday"] = grid["is_payday"].astype(float)

    # Days to next payday (15th or last day of month)
    days_to_payday = np.zeros(len(dom))
    for i, d in enumerate(dom):
        if d <= 15:
            days_to_payday[i] = 15 - d
        else:
            # Days to end of month (approximate next payday)
            days_in_month = 30 if grid["month"].values[i] in [4, 6, 9, 11] else 31
            if grid["month"].values[i] == 2:
                days_in_month = 28
            days_to_payday[i] = max(0, days_in_month - d)
    grid["days_to_payday"] = days_to_payday / 30.0  # normalize
    return grid


def compute_rfm_features(grid: pd.DataFrame, txn_dates: pd.DatetimeIndex,
                          txn_amounts: np.ndarray) -> pd.DataFrame:
    """RFM: recency, frequency_30d, monetary_30d."""
    n = len(grid)
    dates = grid["date"].values

    recency = np.full(n, np.nan)
    freq_30d = np.zeros(n)
    mon_30d = np.zeros(n)

    for i in range(n):
        current_date = pd.Timestamp(dates[i])
        # Recency: days since last transaction
        past_txns = txn_dates[txn_dates <= current_date]
        if len(past_txns) > 0:
            recency[i] = (current_date - past_txns[-1]).days
        else:
            recency[i] = 365  # no prior transactions

        # Frequency and monetary: count and mean in last 30 days
        window_start = current_date - pd.Timedelta(days=30)
        mask = (txn_dates >= window_start) & (txn_dates <= current_date)
        freq_30d[i] = mask.sum()
        if mask.sum() > 0:
            mon_30d[i] = txn_amounts[mask].mean()
        else:
            mon_30d[i] = 0.0

    grid["recency"] = recency
    grid["frequency_30d"] = freq_30d
    grid["monetary_30d"] = mon_30d
    return grid


def compute_stl_decomposition(monthly_expenses: np.ndarray) -> dict:
    """STL-like decomposition at monthly level using moving average."""
    n = len(monthly_expenses)
    if n < 4:
        return {"trend": np.zeros(n), "seasonal": np.zeros(n), "residual": np.zeros(n)}

    # Trend: centered moving average with window 3
    trend = np.zeros(n)
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n, i + 2)
        trend[i] = np.mean(monthly_expenses[lo:hi])

    # Detrended
    detrended = monthly_expenses - trend

    # Seasonal: average of same month position (since we have 12 months = 1 cycle)
    seasonal = np.zeros(n)
    for i in range(n):
        seasonal[i] = detrended[i]  # with 12 months, seasonal = detrended component

    # Residual
    residual = monthly_expenses - trend - seasonal

    return {"trend": trend, "seasonal": seasonal, "residual": residual}


def compute_monthly_target(summaries: pd.DataFrame, persona_id: str) -> pd.Series:
    """Get monthly total expenses as target variable."""
    mask = summaries["persona_id"] == persona_id
    persona_summaries = summaries[mask].sort_values("month")
    target = persona_summaries.set_index("month")["total_expenses"]
    return target


# ---------------------------------------------------------------------------
# Per-Persona Pipeline
# ---------------------------------------------------------------------------

def process_persona(persona_id: str, transactions: pd.DataFrame,
                     summaries: pd.DataFrame, train_imputation: Optional[dict] = None,
                     is_train: bool = True) -> pd.DataFrame:
    """Process one persona: build daily grid, compute all features, return DataFrame."""
    # Get this persona's transactions
    persona_txns = transactions[transactions["persona_id"] == persona_id].copy()
    if persona_txns.empty:
        return pd.DataFrame()

    # Build daily grid
    grid = build_daily_grid(persona_id)

    # Aggregate daily expenses
    daily_exp = aggregate_daily_expenses(transactions, persona_id)

    # Merge daily expenses into grid
    grid = grid.merge(daily_exp, on="date", how="left")
    grid["daily_expense"] = grid["daily_expense"].fillna(0.0)
    grid["txn_count"] = grid["txn_count"].fillna(0).astype(int)
    grid["has_transaction"] = (grid["txn_count"] > 0).astype(float)

    # Transaction-only data for RFM
    expense_mask = persona_txns["transaction_type"] == "expense"
    txn_dates = persona_txns.loc[expense_mask, "date"].sort_values().values
    txn_amounts = persona_txns.loc[expense_mask, "amount"].values

    # Compute features
    grid = compute_temporal_features(grid)
    grid = compute_lag_features(grid)
    grid = compute_rolling_features(grid)
    grid = compute_calendar_features(grid)

    if len(txn_dates) > 0:
        grid = compute_rfm_features(grid, pd.DatetimeIndex(txn_dates), txn_amounts)
    else:
        grid["recency"] = 365.0
        grid["frequency_30d"] = 0.0
        grid["monetary_30d"] = 0.0

    # STL decomposition from monthly summaries
    persona_summ = summaries[summaries["persona_id"] == persona_id].sort_values("month")
    monthly_expenses = persona_summ["total_expenses"].values if len(persona_summ) > 0 else np.zeros(12)

    # Ensure we have exactly 12 months
    if len(monthly_expenses) < 12:
        monthly_expenses = np.pad(monthly_expenses, (0, 12 - len(monthly_expenses)))
    elif len(monthly_expenses) > 12:
        monthly_expenses = monthly_expenses[:12]

    stl = compute_stl_decomposition(monthly_expenses)

    # Map STL to daily level (broadcast monthly value to each day)
    grid["stl_trend"] = grid["month"].map(
        {m + 1: stl["trend"][m] for m in range(12)}
    ).fillna(0.0)
    grid["stl_seasonal"] = grid["month"].map(
        {m + 1: stl["seasonal"][m] for m in range(12)}
    ).fillna(0.0)
    grid["stl_residual"] = grid["month"].map(
        {m + 1: stl["residual"][m] for m in range(12)}
    ).fillna(0.0)

    # Target: next month total expenses
    target_map = {m: total for m, total in zip(
        persona_summ["month"].values, persona_summ["total_expenses"].values
    )}
    grid["target_expenses"] = grid["month"].map(
        {m: target_map.get(m + 1, 0.0) for m in range(1, 13)}
    )
    # For month 12, target is 0 (no month 13)
    grid.loc[grid["month"] == 12, "target_expenses"] = 0.0

    # Add metadata
    grid["user_id"] = persona_id

    return grid


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def compute_train_imputation(train_dfs: list[pd.DataFrame]) -> dict:
    """Compute imputation values from training data only."""
    all_train = pd.concat(train_dfs, ignore_index=True)
    imputation = {}
    for col in FORECASTER_FEATURES:
        if col in all_train.columns:
            imputation[col] = float(all_train[col].median())
    return imputation


def apply_imputation(df: pd.DataFrame, imputation: dict) -> pd.DataFrame:
    """Apply imputation values to a DataFrame."""
    for col, value in imputation.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: ForecasterEngineeringConfig) -> ForecasterEngineeringReport:
    start_time = time.time()
    report = ForecasterEngineeringReport(timestamp=datetime.now().isoformat())

    print("[1/6] Loading data...")
    transactions = load_transactions(config.input_transactions)
    summaries = load_summaries(config.input_summaries)
    persona_splits = load_splits(config.input_splits)

    train_ids = persona_splits.get("train", [])
    val_ids = persona_splits.get("val", [])
    test_ids = persona_splits.get("test", [])

    report.n_personas = len(train_ids) + len(val_ids) + len(test_ids)
    report.split_sizes = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}

    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Process each split
    all_splits = {}
    for split_name, persona_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        print(f"\n[2/6] Processing {split_name} split ({len(persona_ids)} personas)...")
        dfs = []
        for i, pid in enumerate(persona_ids):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Processing persona {i+1}/{len(persona_ids)}: {pid}")
            df = process_persona(pid, transactions, summaries)
            if not df.empty:
                dfs.append(df)
        all_splits[split_name] = dfs

    # Compute imputation from train only
    print("\n[3/6] Computing imputation values from train split...")
    train_imputation = compute_train_imputation(all_splits["train"])
    report.imputation_values = {k: round(v, 6) for k, v in train_imputation.items()}

    # Apply imputation and export
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        dfs = all_splits[split_name]
        if not dfs:
            print(f"  WARNING: No data for {split_name} split")
            continue

        print(f"\n[4/6] Exporting {split_name} split...")
        combined = pd.concat(dfs, ignore_index=True)

        # Apply imputation
        combined = apply_imputation(combined, train_imputation)

        # Drop rows where target is 0 (month 12 has no target)
        # Keep them for feature computation but mark them
        combined["has_target"] = (combined["target_expenses"] > 0).astype(float)

        # Select final columns
        output_cols = META_COLUMNS + FORECASTER_FEATURES + ["has_target"]
        combined = combined[[c for c in output_cols if c in combined.columns]]

        # Export
        out_path = output_dir / f"{split_name}.parquet"
        combined.to_parquet(out_path, index=False)
        report.n_rows[split_name] = len(combined)
        print(f"  Exported {len(combined)} rows to {out_path}")

    # Feature statistics
    print("\n[5/6] Computing feature statistics...")
    train_combined = pd.concat(all_splits["train"], ignore_index=True)
    train_combined = apply_imputation(train_combined, train_imputation)
    for feat in FORECASTER_FEATURES:
        if feat in train_combined.columns:
            report.feature_stats[feat] = {
                "mean": round(float(train_combined[feat].mean()), 4),
                "std": round(float(train_combined[feat].std()), 4),
                "min": round(float(train_combined[feat].min()), 4),
                "max": round(float(train_combined[feat].max()), 4),
            }
    report.n_features = len(FORECASTER_FEATURES)

    # Export metadata
    print("\n[6/6] Exporting metadata...")
    metadata = {
        "timestamp": report.timestamp,
        "n_personas": report.n_personas,
        "n_features": report.n_features,
        "feature_columns": FORECASTER_FEATURES,
        "meta_columns": META_COLUMNS,
        "raw_expense_columns": RAW_EXPENSE_COLUMNS,
        "split_sizes": report.split_sizes,
        "n_rows": report.n_rows,
        "feature_stats": report.feature_stats,
        "imputation_values": report.imputation_values,
    }
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(metadata, f, indent=2)

    pipeline_report = {
        "timestamp": report.timestamp,
        "script": "feature_engineering_forecaster.py",
        "n_personas": report.n_personas,
        "n_features": report.n_features,
        "split_sizes": report.split_sizes,
        "n_rows": report.n_rows,
        "duration_seconds": round(time.time() - start_time, 2),
    }
    with open(output_dir / "pipeline_report.json", "w") as f:
        json.dump(pipeline_report, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone. {report.n_features} features, {sum(report.n_rows.values())} total rows, {elapsed:.1f}s")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Engineering for LSTM Spending Forecaster"
    )
    parser.add_argument("--transactions", default="synth/transactions.parquet",
                        help="Path to transactions.parquet")
    parser.add_argument("--summaries", default="synth/monthly_summaries.parquet",
                        help="Path to monthly_summaries.parquet")
    parser.add_argument("--splits", default="datasets/processed/split_metadata.json",
                        help="Path to split_metadata.json")
    parser.add_argument("--output", default="datasets/forecaster/",
                        help="Output directory for forecaster features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = ForecasterEngineeringConfig(
        input_transactions=args.transactions,
        input_summaries=args.summaries,
        input_splits=args.splits,
        output_dir=args.output,
        seed=args.seed,
    )
    run_pipeline(config)
