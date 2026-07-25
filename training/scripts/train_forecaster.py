"""
Training Pipeline for LSTM Spending Forecaster

Trains and compares Tier 2-3 regressors for monthly expense forecasting
using temporal walk-forward evaluation.

Tiers:
  Tier 2: Random Forest Regressor (monthly aggregated features)
  Tier 3a: LSTM (daily feature sequences)
  Tier 3b: GRU (daily feature sequences)
  Tier 3c: BiLSTM (daily feature sequences)

Evaluation:
  - 5-fold expanding window (temporal_folds.json)
  - Primary metric: MAPE (Mean Absolute Percentage Error)
  - Secondary: RMSE, MAE, R-squared
  - Decision rule: best model must beat naive baseline by 20% MAPE reduction

Usage:
    python scripts/train_forecaster.py --input datasets/forecaster/ --output models/forecaster/
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import multiprocessing, importlib

def _try_import_tf(result_queue):
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        result_queue.put(("ok", tf, keras, layers))
    except Exception as e:
        result_queue.put(("err", str(e), None, None))

HAS_TENSORFLOW = False
tf = keras = layers = None
try:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_try_import_tf, args=(q,))
    p.start()
    p.join(timeout=45)
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        warnings.warn("tensorflow import timed out — Tier 3 (LSTM/GRU/BiLSTM) will be skipped")
    elif not q.empty():
        status, a, b, c = q.get_nowait()
        if status == "ok":
            tf, keras, layers = a, b, c
            HAS_TENSORFLOW = True
        else:
            warnings.warn(f"tensorflow import failed ({a}) — Tier 3 (LSTM/GRU/BiLSTM) will be skipped")
except Exception:
    warnings.warn("tensorflow unavailable — Tier 3 (LSTM/GRU/BiLSTM) will be skipped")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECASTER_FEATURES = [
    "day_of_week_sin", "day_of_week_cos", "day_of_month",
    "lag_1d", "lag_7d", "lag_14d", "lag_15d", "lag_30d", "lag_60d",
    "rolling_mean_7d", "rolling_std_7d",
    "rolling_mean_14d", "rolling_std_14d",
    "rolling_mean_30d", "rolling_std_30d",
    "is_payday", "days_to_payday",
    "recency", "frequency_30d", "monetary_30d",
    "stl_trend", "stl_seasonal", "stl_residual",
]

META_COLUMNS = ["user_id", "date", "month", "year", "target_expenses",
                "has_transaction", "has_target"]

PRE_REGISTERED_MAPE_REDUCTION = 0.20  # 20% MAPE reduction vs naive

SEQ_LENGTH = 30  # days lookback for LSTM


@dataclass
class FoldResult:
    fold: int
    train_months: list
    test_months: list
    n_train_samples: int
    n_test_samples: int
    tier_results: dict = field(default_factory=dict)


@dataclass
class TrainingReport:
    timestamp: str = ""
    n_folds: int = 0
    fold_results: list = field(default_factory=list)
    aggregate_metrics: dict = field(default_factory=dict)
    naive_mape: float = 0.0
    winner: str = ""
    winner_reason: str = ""


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_forecaster_data(input_dir: str) -> dict:
    input_path = Path(input_dir)
    splits = {}
    for name in ["train", "val", "test"]:
        path = input_path / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Forecaster data not found: {path}")
        splits[name] = pd.read_parquet(path)
    return splits


def load_temporal_folds(folds_path: str) -> list:
    with open(folds_path) as f:
        return json.load(f)


def load_feature_columns(input_dir: str) -> dict:
    path = Path(input_dir) / "feature_columns.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Feature Aggregation
# ---------------------------------------------------------------------------

def aggregate_to_monthly(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Aggregate daily features to monthly level."""
    agg_dict = {}
    for col in feature_cols:
        if col in df.columns:
            agg_dict[col] = "mean"
    agg_dict["target_expenses"] = "first"
    agg_dict["has_target"] = "first"
    agg_dict["has_transaction"] = "mean"

    monthly = df.groupby(["user_id", "month"]).agg(agg_dict).reset_index()
    return monthly


def prepare_monthly_sequences(monthly_df: pd.DataFrame, feature_cols: list,
                               lookback: int = 3,
                               filter_months: Optional[list] = None) -> tuple:
    """Create sequences for LSTM: past N months of features -> next month target.
    
    If filter_months is provided, only produce samples where the target month
    is in filter_months (but lookback can use any prior months).
    """
    X_seq, y_seq, meta = [], [], []
    for uid in monthly_df["user_id"].unique():
        user_data = monthly_df[monthly_df["user_id"] == uid].sort_values("month")
        features = user_data[feature_cols].values
        targets = user_data["target_expenses"].values
        months = user_data["month"].values

        for i in range(lookback, len(user_data)):
            if targets[i] > 0:
                if filter_months is None or months[i] in filter_months:
                    X_seq.append(features[i - lookback:i])
                    y_seq.append(targets[i])
                    meta.append({"user_id": uid, "month": int(months[i])})

    if not X_seq:
        return np.array([]), np.array([]), []
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), meta


def prepare_flat_features(monthly_df: pd.DataFrame, feature_cols: list,
                          filter_months: Optional[list] = None) -> tuple:
    """Create flat feature matrix for RF: past N months flattened.
    
    If filter_months is provided, only produce samples where the target month
    is in filter_months (but lookback can use any prior months).
    """
    lookback = 3
    X_flat, y_flat, meta = [], [], []
    for uid in monthly_df["user_id"].unique():
        user_data = monthly_df[monthly_df["user_id"] == uid].sort_values("month")
        features = user_data[feature_cols].values
        targets = user_data["target_expenses"].values
        months = user_data["month"].values

        for i in range(lookback, len(user_data)):
            if targets[i] > 0:
                if filter_months is None or months[i] in filter_months:
                    x = features[i - lookback:i].flatten()
                    X_flat.append(x)
                    y_flat.append(targets[i])
                    meta.append({"user_id": uid, "month": int(months[i])})

    if not X_flat:
        return np.array([]), np.array([]), []
    return np.array(X_flat, dtype=np.float32), np.array(y_flat, dtype=np.float32), meta


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, tier_name: str) -> dict:
    """Compute regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE: exclude zero/near-zero targets to avoid division-by-zero explosion
    # Threshold: 1% of mean target value
    mean_target = np.mean(y_true)
    nonzero_mask = y_true > (mean_target * 0.01)
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                                     / y_true[nonzero_mask])) * 100)
    else:
        mape = float("inf")

    return {
        "mape": round(mape, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "tier": tier_name,
    }


# ---------------------------------------------------------------------------
# Model Training Functions
# ---------------------------------------------------------------------------

def train_rf(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Train Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def build_lstm_model(input_shape: tuple, model_type: str = "lstm") -> Any:
    """Build LSTM/GRU/BiLSTM model."""
    if not HAS_TENSORFLOW:
        return None

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    if model_type == "lstm":
        model.add(layers.LSTM(32, dropout=0.2))
    elif model_type == "gru":
        model.add(layers.GRU(32, dropout=0.2))
    elif model_type == "bilstm":
        model.add(layers.Bidirectional(layers.LSTM(32, dropout=0.2)))

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model


def train_lstm_variant(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       model_type: str = "lstm") -> tuple:
    """Train LSTM/GRU/BiLSTM model."""
    if not HAS_TENSORFLOW:
        return None, np.zeros(len(y_test))

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), model_type)
    if model is None:
        return None, np.zeros(len(y_test))

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=5, batch_size=64,
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    return model, y_pred


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def run_wfv(splits: dict, folds: list, feature_cols: list,
            run_name: str = "") -> dict:
    """Run walk-forward validation across all folds.

    Key design: aggregate ALL data to monthly once, then for each fold
    create train/test by filtering on months. Test samples use ALL prior
    months as lookback context (not just test-month data).
    """
    print(f"\n{'='*60}")
    print(f"Walk-Forward Validation: {run_name}")
    print(f"{'='*60}")

    # Aggregate ALL data to monthly once
    all_data = pd.concat(splits.values(), ignore_index=True)
    all_monthly = aggregate_to_monthly(all_data, feature_cols)
    print(f"Total monthly samples: {len(all_monthly)}")
    print(f"Unique personas: {all_monthly['user_id'].nunique()}")

    all_fold_results = []

    for fold_info in folds:
        fold_num = fold_info["fold"]
        train_months = fold_info["train_months"]
        test_months = fold_info["test_months"]

        print(f"\n--- Fold {fold_num}: Train {train_months} -> Test {test_months} ---")

        # For RF and LSTM: use ALL months up to test month for lookback context
        # Train on train_months only, predict on test_months
        # But the lookback window can use any months <= max(train_months)

        # Training set: months in train_months, with lookback from earlier months
        train_monthly = all_monthly[all_monthly["month"].isin(train_months)].copy()

        # Test set: months in test_months, with lookback from all prior months
        context_months = train_months + test_months  # all months up to test
        test_context = all_monthly[all_monthly["month"].isin(context_months)].copy()

        if train_monthly.empty:
            print(f"  WARNING: Empty train for fold {fold_num}, skipping")
            continue

        print(f"  Train monthly: {len(train_monthly)} samples "
              f"({train_monthly['user_id'].nunique()} personas)")
        print(f"  Test context: {len(test_context)} samples "
              f"({test_context['user_id'].nunique()} personas)")

        fold_results = {"fold": fold_num, "train_months": train_months,
                        "test_months": test_months,
                        "n_train": len(train_monthly),
                        "n_test": 0,
                        "tier_results": {}}

        # --- Naive Baseline ---
        test_actual = test_context[test_context["month"].isin(test_months)]
        naive_pred = np.full(len(test_actual),
                              train_monthly["target_expenses"].mean())
        naive_metrics = compute_metrics(
            test_actual["target_expenses"].values, naive_pred, "naive_baseline"
        )
        fold_results["tier_results"]["naive_baseline"] = naive_metrics
        fold_results["n_test"] = len(test_actual)
        print(f"  Naive: MAPE={naive_metrics['mape']:.2f}%")

        # --- Tier 2: Random Forest ---
        # Train on train months, test on test months (with lookback context)
        X_train_rf, y_train_rf, _ = prepare_flat_features(train_monthly, feature_cols)
        X_test_rf, y_test_rf, meta_rf = prepare_flat_features(
            test_context, feature_cols, filter_months=test_months
        )

        if len(X_train_rf) > 0 and len(X_test_rf) > 0:
            scaler_rf = StandardScaler()
            X_train_rf_s = scaler_rf.fit_transform(X_train_rf)
            X_test_rf_s = scaler_rf.transform(X_test_rf)

            rf_model, rf_pred = train_rf(X_train_rf_s, y_train_rf,
                                          X_test_rf_s, y_test_rf)
            rf_metrics = compute_metrics(y_test_rf, rf_pred, "tier2_random_forest")
            fold_results["tier_results"]["tier2_random_forest"] = rf_metrics
            print(f"  RF: MAPE={rf_metrics['mape']:.2f}%, R²={rf_metrics['r2']:.4f}")
        else:
            print(f"  RF: Skipped (train={len(X_train_rf)}, test={len(X_test_rf)})")

        # --- Tier 3: LSTM / GRU / BiLSTM ---
        if HAS_TENSORFLOW:
            X_train_seq, y_train_seq, _ = prepare_monthly_sequences(
                train_monthly, feature_cols, lookback=3
            )
            X_test_seq, y_test_seq, meta_seq = prepare_monthly_sequences(
                test_context, feature_cols, lookback=3, filter_months=test_months
            )

            if len(X_train_seq) > 0 and len(X_test_seq) > 0:
                scaler_seq = StandardScaler()
                n_train, seq_len, n_feat = X_train_seq.shape
                X_train_flat = X_train_seq.reshape(-1, n_feat)
                X_train_flat_s = scaler_seq.fit_transform(X_train_flat)
                X_train_seq_s = X_train_flat_s.reshape(n_train, seq_len, n_feat)

                n_test = X_test_seq.shape[0]
                X_test_flat = X_test_seq.reshape(-1, n_feat)
                X_test_flat_s = scaler_seq.transform(X_test_flat)
                X_test_seq_s = X_test_flat_s.reshape(n_test, seq_len, n_feat)

                for variant in ["lstm", "gru", "bilstm"]:
                    tier_name = f"tier3_{variant}"
                    model, pred = train_lstm_variant(
                        X_train_seq_s, y_train_seq,
                        X_test_seq_s, y_test_seq,
                        model_type=variant,
                    )
                    metrics = compute_metrics(y_test_seq, pred, tier_name)
                    fold_results["tier_results"][tier_name] = metrics
                    print(f"  {variant.upper()}: MAPE={metrics['mape']:.2f}%, "
                          f"R²={metrics['r2']:.4f}")
            else:
                print(f"  LSTM: Skipped (train={len(X_train_seq)}, "
                      f"test={len(X_test_seq)})")
        else:
            print("  LSTM/GRU/BiLSTM: Skipped (tensorflow not installed)")

        all_fold_results.append(fold_results)

    return {"folds": all_fold_results}


# ---------------------------------------------------------------------------
# Aggregate & Decision Rule
# ---------------------------------------------------------------------------

def aggregate_metrics(fold_results: list) -> dict:
    """Aggregate metrics across folds."""
    tier_names = set()
    for fr in fold_results:
        tier_names.update(fr["tier_results"].keys())

    aggregate = {}
    for tier in tier_names:
        mapes = [fr["tier_results"][tier]["mape"]
                 for fr in fold_results if tier in fr["tier_results"]]
        rmses = [fr["tier_results"][tier]["rmse"]
                 for fr in fold_results if tier in fr["tier_results"]]
        r2s = [fr["tier_results"][tier]["r2"]
               for fr in fold_results if tier in fr["tier_results"]]

        if mapes:
            aggregate[tier] = {
                "mape_mean": round(float(np.mean(mapes)), 4),
                "mape_std": round(float(np.std(mapes)), 4),
                "rmse_mean": round(float(np.mean(rmses)), 4),
                "rmse_std": round(float(np.std(rmses)), 4),
                "r2_mean": round(float(np.mean(r2s)), 4),
                "r2_std": round(float(np.std(r2s)), 4),
                "n_folds": len(mapes),
            }
    return aggregate


def apply_decision_rule(aggregate: dict, naive_mape: float) -> tuple:
    """Apply pre-registered decision rule."""
    best_tier = None
    best_mape = float("inf")

    for tier, metrics in aggregate.items():
        if tier == "naive_baseline":
            continue
        if metrics["mape_mean"] < best_mape:
            best_mape = metrics["mape_mean"]
            best_tier = tier

    if best_tier is None or naive_mape <= 0:
        return "naive_baseline", "No learned model available"

    mape_reduction = 1.0 - (best_mape / naive_mape)
    if mape_reduction >= PRE_REGISTERED_MAPE_REDUCTION:
        reason = (f"{best_tier} reduces MAPE by {mape_reduction*100:.1f}% "
                  f"(>{PRE_REGISTERED_MAPE_REDUCTION*100:.0f}% threshold)")
        return best_tier, reason
    else:
        reason = (f"Best model {best_tier} reduces MAPE by only "
                  f"{mape_reduction*100:.1f}% (<{PRE_REGISTERED_MAPE_REDUCTION*100:.0f}% "
                  f"threshold). Naive baseline preferred for simplicity.")
        return "naive_baseline", reason


# ---------------------------------------------------------------------------
# Model Saving
# ---------------------------------------------------------------------------

def save_models(splits: dict, feature_cols: list, output_dir: Path,
                winner: str):
    """Re-train winner on full data and save artifacts."""
    print(f"\nRe-training {winner} on full data for final model...")

    all_data = pd.concat(splits.values(), ignore_index=True)
    monthly = aggregate_to_monthly(all_data, feature_cols)

    scaler = StandardScaler()

    if winner == "tier2_random_forest":
        X, y, _ = prepare_flat_features(monthly, feature_cols)
        X_s = scaler.fit_transform(X)
        model = RandomForestRegressor(
            n_estimators=200, max_depth=10, n_jobs=-1, random_state=42
        )
        model.fit(X_s, y)
        joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols},
                    output_dir / "tier2_random_forest.joblib")
        print("  Saved tier2_random_forest.joblib")

    elif winner.startswith("tier3_") and HAS_TENSORFLOW:
        variant = winner.replace("tier3_", "")
        X, y, _ = prepare_monthly_sequences(monthly, feature_cols, lookback=3)
        if len(X) > 0:
            n, seq_len, n_feat = X.shape
            X_flat = X.reshape(-1, n_feat)
            X_flat_s = scaler.fit_transform(X_flat)
            X_s = X_flat_s.reshape(n, seq_len, n_feat)

            model = build_lstm_model((seq_len, n_feat), variant)
            model.fit(X_s, y, epochs=50, batch_size=32, verbose=0)
            model.save(output_dir / f"{winner}.keras")
            joblib.dump({"scaler": scaler, "feature_cols": feature_cols,
                         "seq_length": seq_len},
                        output_dir / f"{winner}_meta.joblib")
            print(f"  Saved {winner}.keras + meta.joblib")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def write_evaluation_report(report: TrainingReport, output_dir: Path):
    """Write human-readable evaluation report."""
    lines = [
        "# Forecaster Training Evaluation Report",
        f"\n**Timestamp:** {report.timestamp}",
        f"**Folds:** {report.n_folds}",
        f"**MAPE Reduction Threshold:** {PRE_REGISTERED_MAPE_REDUCTION*100:.0f}%",
        f"\n## Winner: {report.winner}",
        f"**Reason:** {report.winner_reason}",
        "\n## Aggregate Results",
        "\n| Tier | MAPE (mean±std) | RMSE (mean±std) | R² (mean±std) | Folds |",
        "|------|----------------|----------------|--------------|-------|",
    ]

    for tier, metrics in sorted(report.aggregate_metrics.items()):
        lines.append(
            f"| {tier} | "
            f"{metrics['mape_mean']:.2f}±{metrics['mape_std']:.2f}% | "
            f"{metrics['rmse_mean']:.2f}±{metrics['rmse_std']:.2f} | "
            f"{metrics['r2_mean']:.4f}±{metrics['r2_std']:.4f} | "
            f"{metrics['n_folds']} |"
        )

    lines.append("\n## Per-Fold Results")
    for fr in report.fold_results:
        lines.append(f"\n### Fold {fr['fold']}")
        lines.append(f"- Train months: {fr['train_months']}")
        lines.append(f"- Test months: {fr['test_months']}")
        lines.append(f"- Train samples: {fr['n_train']}")
        lines.append(f"- Test samples: {fr['n_test']}")
        lines.append("\n| Tier | MAPE | RMSE | MAE | R² |")
        lines.append("|------|------|------|-----|-----|")
        for tier, m in sorted(fr["tier_results"].items()):
            lines.append(
                f"| {tier} | {m['mape']:.2f}% | {m['rmse']:.2f} | "
                f"{m['mae']:.2f} | {m['r2']:.4f} |"
            )

    with open(output_dir / "evaluation_report.md", "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_training(config: dict) -> TrainingReport:
    start_time = time.time()
    report = TrainingReport(timestamp=datetime.now().isoformat())

    input_dir = config["input"]
    output_dir = Path(config["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading forecaster data...")
    splits = load_forecaster_data(input_dir)
    feature_meta = load_feature_columns(input_dir)
    feature_cols = feature_meta.get("feature_columns", FORECASTER_FEATURES)

    folds_path = Path(config.get("folds", "datasets/processed/temporal_folds.json"))
    folds = load_temporal_folds(str(folds_path))
    report.n_folds = len(folds)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Folds: {len(folds)}")

    # Run WFV
    print("\n[2/5] Running walk-forward validation...")
    wfv_results = run_wfv(splits, folds, feature_cols)
    report.fold_results = wfv_results["folds"]

    # Aggregate
    print("\n[3/5] Aggregating metrics...")
    report.aggregate_metrics = aggregate_metrics(report.fold_results)

    naive_mape = report.aggregate_metrics.get("naive_baseline", {}).get("mape_mean", 0)
    report.naive_mape = naive_mape

    # Decision rule
    print("\n[4/5] Applying decision rule...")
    winner, reason = apply_decision_rule(report.aggregate_metrics, naive_mape)
    report.winner = winner
    report.winner_reason = reason
    print(f"  Winner: {winner}")
    print(f"  Reason: {reason}")

    # Save models
    print("\n[5/5] Saving models and reports...")
    save_models(splits, feature_cols, output_dir, winner)

    # Save evaluation JSON
    eval_json = {
        "timestamp": report.timestamp,
        "n_folds": report.n_folds,
        "pre_registered_mape_reduction": PRE_REGISTERED_MAPE_REDUCTION,
        "naive_mape": naive_mape,
        "aggregate_metrics": report.aggregate_metrics,
        "winner": report.winner,
        "winner_reason": report.winner_reason,
        "fold_details": report.fold_results,
    }
    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(eval_json, f, indent=2)

    # Write report
    write_evaluation_report(report, output_dir)

    elapsed = time.time() - start_time
    print(f"\nDone. Winner: {winner}, {elapsed:.1f}s")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Pipeline for LSTM Spending Forecaster"
    )
    parser.add_argument("--input", default="datasets/forecaster/",
                        help="Input directory with forecaster features")
    parser.add_argument("--output", default="models/forecaster/",
                        help="Output directory for models")
    parser.add_argument("--folds", default="datasets/processed/temporal_folds.json",
                        help="Path to temporal_folds.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {
        "input": args.input,
        "output": args.output,
        "folds": args.folds,
    }
    run_training(config)
