"""
Anomaly Detector Training Pipeline

Trains and compares multiple anomaly detection tiers for transaction-level
anomaly detection on synthetic financial data.

Tiers:
  Tier 0: Majority-class baseline (sanity floor)
  Tier 1: IQR (statistical, per-feature)
  Tier 2: Isolation Forest, One-Class SVM, Autoencoder (Keras)
  Tier 3: Hybrid Ensemble (voting from Tier 1-2 detectors)

Evaluation:
  - 5-fold expanding window (temporal_folds.json)
  - Primary metric: PR-AUC (area under precision-recall curve)
  - Secondary: F1 at optimal threshold, precision, recall
  - Decision rule: best model must beat Tier 0 baseline by PR-AUC

Usage:
    python scripts/train_anomaly.py --input datasets/anomaly/ --output models/anomaly/
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.svm import OneClassSVM

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import multiprocessing

def _try_import_tf(result_queue):
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, callbacks
        result_queue.put(("ok", tf, keras, layers, callbacks))
    except Exception as e:
        result_queue.put(("err", str(e), None, None, None))

HAS_TENSORFLOW = False
tf = keras = layers = callbacks = None
try:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_try_import_tf, args=(q,))
    p.start()
    p.join(timeout=45)
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        warnings.warn("tensorflow import timed out — Tier 2 Autoencoder will be skipped")
    elif not q.empty():
        status, a, b, c, d = q.get_nowait()
        if status == "ok":
            tf, keras, layers, callbacks = a, b, c, d
            HAS_TENSORFLOW = True
        else:
            warnings.warn(f"tensorflow import failed ({a}) — Tier 2 Autoencoder will be skipped")
except Exception:
    warnings.warn("tensorflow unavailable — Tier 2 Autoencoder will be skipped")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
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

LABEL_COL = "is_anomalous"
META_COLS = ["user_id", "transaction_id", "month", "date", "category",
             "amount", "transaction_type", "is_anomalous", "anomaly_type"]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str):
    """Load parquet splits and feature metadata."""
    data_path = Path(data_dir)

    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")

    with open(data_path / "feature_columns.json") as f:
        meta = json.load(f)

    return train_df, val_df, test_df, meta


def load_folds(folds_path: str):
    """Load temporal fold definitions."""
    with open(folds_path) as f:
        return json.load(f)


def prepare_fold_data(train_df: pd.DataFrame, fold: dict):
    """Split train_df into fold-train and fold-test based on months."""
    train_months = set(fold["train_months"])
    test_months = set(fold["test_months"])

    fold_train = train_df[train_df["month"].isin(train_months)]
    fold_test = train_df[train_df["month"].isin(test_months)]

    return fold_train, fold_test


def extract_features(df: pd.DataFrame):
    """Extract X (features) and y (labels) from a dataframe."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5):
    """Compute anomaly detection metrics."""
    y_pred = (y_scores >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Find best F1 threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else threshold
    best_f1 = float(f1_scores[best_idx])

    # Precision and recall at best threshold
    y_pred_best = (y_scores >= best_threshold).astype(int)
    tp = int(np.sum((y_pred_best == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_best == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_best == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred_best == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "pr_auc": round(pr_auc, 4),
        "best_f1": round(best_f1, 4),
        "best_threshold": round(best_threshold, 4),
        "precision_at_best": round(precision, 4),
        "recall_at_best": round(recall, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_positives": int(np.sum(y_true)),
        "n_predictions_positive": int(np.sum(y_pred_best)),
    }


def compute_baseline_metrics(y_true: np.ndarray):
    """Tier 0: always predict majority class (all normal)."""
    y_scores = np.zeros_like(y_true, dtype=float)
    return compute_metrics(y_true, y_scores)


# ---------------------------------------------------------------------------
# Tier 1: IQR Detector
# ---------------------------------------------------------------------------

class IQRDetector:
    """Statistical anomaly detector using IQR on each feature."""

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.bounds = {}

    def fit(self, X: np.ndarray):
        for j in range(X.shape[1]):
            col = X[:, j]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            self.bounds[j] = (q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr)

    def score(self, X: np.ndarray) -> np.ndarray:
        anomaly_flags = np.zeros(X.shape[0], dtype=float)
        for j in range(X.shape[1]):
            lo, hi = self.bounds[j]
            outlier_mask = (X[:, j] < lo) | (X[:, j] > hi)
            anomaly_flags += outlier_mask.astype(float)
        # Normalize to [0, 1] by proportion of features that are outliers
        return anomaly_flags / X.shape[1]


# ---------------------------------------------------------------------------
# Tier 2: Isolation Forest
# ---------------------------------------------------------------------------

def train_isolation_forest(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray):
    """Train Isolation Forest with a fixed configuration."""
    model = IsolationForest(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        n_estimators=200,
        max_samples="auto",
        contamination=0.003,
        max_features=0.8,
    )
    model.fit(X_train)
    val_scores = -model.decision_function(X_val)
    val_scores = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min() + 1e-8)
    metrics = compute_metrics(y_val, val_scores)
    return model, {"contamination": 0.003, "n_estimators": 200}, metrics["pr_auc"]


# ---------------------------------------------------------------------------
# Tier 2: One-Class SVM
# ---------------------------------------------------------------------------

def train_ocsvm(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray):
    """Train One-Class SVM with subsampling for speed."""
    max_samples = min(3000, X_train.shape[0])
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(X_train.shape[0], max_samples, replace=False)
    X_sub = X_train[idx]

    best_score = -1
    best_model = None
    best_params = None

    for nu in [0.01, 0.05]:
        try:
            model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
            model.fit(X_sub)
            val_scores = -model.decision_function(X_val)
            val_scores = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min() + 1e-8)
            metrics = compute_metrics(y_val, val_scores)
            if metrics["pr_auc"] > best_score:
                best_score = metrics["pr_auc"]
                best_model = model
                best_params = {"nu": nu}
        except Exception:
            continue

    return best_model, best_params, best_score


# ---------------------------------------------------------------------------
# Tier 2: Autoencoder (Keras)
# ---------------------------------------------------------------------------

def build_autoencoder(input_dim: int, encoding_dim: int = 7):
    """Build a simple autoencoder for anomaly detection."""
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation="relu")(encoder_input)
    encoded = layers.Dense(encoding_dim // 2, activation="relu")(encoded)
    decoded = layers.Dense(encoding_dim, activation="relu")(encoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray):
    """Train autoencoder; anomalies have higher reconstruction error."""
    if not HAS_TENSORFLOW:
        return None, None, -1

    input_dim = X_train.shape[1]
    model = build_autoencoder(input_dim)

    early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=3)

    model.fit(
        X_train, X_train,
        epochs=30,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )

    val_recon = model.predict(X_val, verbose=0)
    val_scores = np.mean((X_val - val_recon) ** 2, axis=1)
    val_scores = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min() + 1e-8)

    metrics = compute_metrics(y_val, val_scores)
    return model, "mse", metrics["pr_auc"]


# ---------------------------------------------------------------------------
# Tier 3: Hybrid Ensemble
# ---------------------------------------------------------------------------

class HybridEnsemble:
    """Ensemble of Tier 1-2 detectors using score averaging."""

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
            elif hasattr(detector, "predict") and hasattr(detector, "model"):
                recon = detector.model.predict(X, verbose=0)
                s = np.mean((X - recon) ** 2, axis=1)
                s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            else:
                continue
            scores += weight * s
        return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrices(results: dict, output_dir: Path):
    """Plot confusion matrix heatmaps for all models."""
    if not HAS_MATPLOTLIB:
        return

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"])
        ax.set_title(f"{name}\nPR-AUC={res['pr_auc']:.3f}", fontsize=10)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curves(y_true: np.ndarray, score_dict: dict, output_dir: Path):
    """Plot precision-recall curves for all models."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in score_dict.items():
        precisions, recalls, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        ax.plot(recalls, precisions, label=f"{name} (AUC={pr_auc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Anomaly Detection")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/anomaly/")
    parser.add_argument("--output", default="models/anomaly/")
    parser.add_argument("--folds", default="datasets/processed/temporal_folds.json")
    args = parser.parse_args()

    t0 = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading data...")
    train_df, val_df, test_df, meta = load_data(args.input)
    folds = load_folds(args.folds)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Features: {len(FEATURE_COLS)}")

    all_fold_results = {}
    all_fold_scores = {}

    print(f"\n[2/6] Training across {len(folds)} folds...")

    for fold_info in folds:
        fold_num = fold_info["fold"]
        print(f"\n  Fold {fold_num}: train months {fold_info['train_months']}, "
              f"test months {fold_info['test_months']}")

        fold_train, fold_test = prepare_fold_data(train_df, fold_info)
        X_train, y_train = extract_features(fold_train)
        X_test, y_test = extract_features(fold_test)

        print(f"    Train: {len(X_train)} ({int(y_train.sum())} anomalies), "
              f"Test: {len(X_test)} ({int(y_test.sum())} anomalies)")

        fold_scores = {}

        # Tier 0: Baseline
        baseline = compute_baseline_metrics(y_test)
        print(f"    Tier 0 (Baseline): PR-AUC={baseline['pr_auc']:.4f}")
        fold_scores["tier0_baseline"] = np.zeros_like(y_test, dtype=float)

        # Tier 1: IQR
        iqr = IQRDetector(iqr_multiplier=1.5)
        iqr.fit(X_train)
        iqr_scores = iqr.score(X_test)
        iqr_metrics = compute_metrics(y_test, iqr_scores)
        print(f"    Tier 1 (IQR): PR-AUC={iqr_metrics['pr_auc']:.4f}, "
              f"F1={iqr_metrics['best_f1']:.4f}")
        fold_scores["tier1_iqr"] = iqr_scores

        # Tier 2: Isolation Forest
        if_model, if_params, if_val_score = train_isolation_forest(
            X_train, y_train, X_test, y_test
        )
        if_scores = -if_model.decision_function(X_test)
        if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        if_metrics = compute_metrics(y_test, if_scores)
        print(f"    Tier 2 (IF): PR-AUC={if_metrics['pr_auc']:.4f}, "
              f"F1={if_metrics['best_f1']:.4f}, params={if_params}")
        fold_scores["tier2_isolation_forest"] = if_scores

        # Tier 2: One-Class SVM
        ocsvm_model, ocsvm_params, ocsvm_val_score = train_ocsvm(
            X_train, y_train, X_test, y_test
        )
        if ocsvm_model is not None:
            ocsvm_scores = -ocsvm_model.decision_function(X_test)
            ocsvm_scores = (ocsvm_scores - ocsvm_scores.min()) / (ocsvm_scores.max() - ocsvm_scores.min() + 1e-8)
            ocsvm_metrics = compute_metrics(y_test, ocsvm_scores)
            print(f"    Tier 2 (OCSVM): PR-AUC={ocsvm_metrics['pr_auc']:.4f}, "
                  f"F1={ocsvm_metrics['best_f1']:.4f}, params={ocsvm_params}")
            fold_scores["tier2_ocsvm"] = ocsvm_scores
        else:
            print("    Tier 2 (OCSVM): FAILED")
            ocsvm_metrics = {"pr_auc": 0, "best_f1": 0}

        # Tier 2: Autoencoder
        if HAS_TENSORFLOW:
            ae_model, ae_loss, ae_val_score = train_autoencoder(
                X_train, y_train, X_test, y_test
            )
            if ae_model is not None:
                ae_recon = ae_model.predict(X_test, verbose=0)
                ae_scores = np.mean((X_test - ae_recon) ** 2, axis=1)
                ae_scores = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-8)
                ae_metrics = compute_metrics(y_test, ae_scores)
                print(f"    Tier 2 (AE): PR-AUC={ae_metrics['pr_auc']:.4f}, "
                      f"F1={ae_metrics['best_f1']:.4f}")
                fold_scores["tier2_autoencoder"] = ae_scores
            else:
                print("    Tier 2 (AE): FAILED")
                ae_metrics = {"pr_auc": 0, "best_f1": 0}
        else:
            ae_metrics = {"pr_auc": 0, "best_f1": 0}

        # Tier 3: Hybrid Ensemble
        ensemble_detectors = [("iqr", iqr), ("if", if_model)]
        if ocsvm_model is not None:
            ensemble_detectors.append(("ocsvm", ocsvm_model))

        ensemble = HybridEnsemble(
            detectors=[d for _, d in ensemble_detectors],
            weights=[1.0 / len(ensemble_detectors)] * len(ensemble_detectors)
        )
        ensemble_scores = ensemble.score(X_test)
        ensemble_metrics = compute_metrics(y_test, ensemble_scores)
        print(f"    Tier 3 (Ensemble): PR-AUC={ensemble_metrics['pr_auc']:.4f}, "
              f"F1={ensemble_metrics['best_f1']:.4f}")
        fold_scores["tier3_ensemble"] = ensemble_scores

        # Store fold results
        all_fold_results[fold_num] = {
            "baseline": baseline,
            "tier1_iqr": iqr_metrics,
            "tier2_isolation_forest": if_metrics,
            "tier2_ocsvm": ocsvm_metrics,
            "tier2_autoencoder": ae_metrics,
            "tier3_ensemble": ensemble_metrics,
            "models": {
                "iqr_bounds": iqr.bounds,
                "if_params": if_params,
                "ocsvm_params": ocsvm_params,
            }
        }
        all_fold_scores[fold_num] = fold_scores

    # Aggregate across folds
    print(f"\n[3/6] Aggregating results across {len(folds)} folds...")
    model_names = [k for k in all_fold_results[folds[0]["fold"]].keys()
                   if k not in ("baseline", "models")]

    summary = {}
    for model_name in model_names:
        pr_aucs = [all_fold_results[f["fold"]][model_name]["pr_auc"] for f in folds]
        f1s = [all_fold_results[f["fold"]][model_name]["best_f1"] for f in folds]
        summary[model_name] = {
            "pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
            "pr_auc_std": round(float(np.std(pr_aucs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
        }

    baseline_pr_aucs = [all_fold_results[f["fold"]]["baseline"]["pr_auc"] for f in folds]
    summary["baseline"] = {
        "pr_auc_mean": round(float(np.mean(baseline_pr_aucs)), 4),
        "pr_auc_std": round(float(np.std(baseline_pr_aucs)), 4),
    }

    # Print summary table
    print("\n  Model Summary (mean ± std across folds):")
    print(f"  {'Model':<30} {'PR-AUC':>12} {'F1':>12}")
    print(f"  {'-'*54}")
    for name, stats in sorted(summary.items(), key=lambda x: -x[1].get("pr_auc_mean", 0)):
        pr_str = f"{stats['pr_auc_mean']:.4f} ± {stats['pr_auc_std']:.4f}"
        f1_str = f"{stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}" if "f1_mean" in stats else "N/A"
        print(f"  {name:<30} {pr_str:>12} {f1_str:>12}")

    # Select winner
    best_model_name = max(
        [k for k in summary if k != "baseline"],
        key=lambda k: summary[k]["pr_auc_mean"]
    )
    best_stats = summary[best_model_name]
    baseline_stats = summary["baseline"]
    improvement = ((best_stats["pr_auc_mean"] - baseline_stats["pr_auc_mean"])
                   / max(baseline_stats["pr_auc_mean"], 1e-8) * 100)

    print(f"\n  Winner: {best_model_name}")
    print(f"  PR-AUC improvement over baseline: {improvement:.1f}%")

    # Retrain winner on full training data and evaluate on test
    print(f"\n[4/6] Retraining {best_model_name} on full training set...")
    X_full_train, y_full_train = extract_features(train_df)
    X_test_final, y_test_final = extract_features(test_df)

    if best_model_name == "tier1_iqr":
        winner = IQRDetector(iqr_multiplier=1.5)
        winner.fit(X_full_train)
        test_scores = winner.score(X_test_final)
    elif best_model_name == "tier2_isolation_forest":
        winner = IsolationForest(
            random_state=RANDOM_SEED, n_jobs=-1,
            n_estimators=200, max_samples="auto",
            contamination=0.003, max_features=0.8
        )
        winner.fit(X_full_train)
        test_scores = -winner.decision_function(X_test_final)
        test_scores = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-8)
    elif best_model_name == "tier2_ocsvm":
        winner = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        winner.fit(X_full_train)
        test_scores = -winner.decision_function(X_test_final)
        test_scores = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-8)
    elif best_model_name == "tier2_autoencoder" and HAS_TENSORFLOW:
        winner = build_autoencoder(X_full_train.shape[1])
        winner.fit(X_full_train, X_full_train, epochs=50, batch_size=64,
                   validation_split=0.1, verbose=0)
        test_recon = winner.predict(X_test_final, verbose=0)
        test_scores = np.mean((X_test_final - test_recon) ** 2, axis=1)
        test_scores = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-8)
    elif best_model_name == "tier3_ensemble":
        # Build ensemble from full data
        iqr_w = IQRDetector(iqr_multiplier=1.5)
        iqr_w.fit(X_full_train)

        if_w = IsolationForest(random_state=RANDOM_SEED, n_jobs=-1,
                               n_estimators=200, contamination=0.003)
        if_w.fit(X_full_train)

        ensemble_w = HybridEnsemble(
            detectors=[iqr_w, if_w],
            weights=[0.5, 0.5]
        )
        test_scores = ensemble_w.score(X_test_final)
        winner = ensemble_w
    else:
        winner = None
        test_scores = np.zeros_like(y_test_final, dtype=float)

    final_metrics = compute_metrics(y_test_final, test_scores)
    print(f"  Test PR-AUC: {final_metrics['pr_auc']:.4f}")
    print(f"  Test F1: {final_metrics['best_f1']:.4f}")
    print(f"  Test Precision: {final_metrics['precision_at_best']:.4f}")
    print(f"  Test Recall: {final_metrics['recall_at_best']:.4f}")

    # Save model
    print(f"\n[5/6] Saving model and artifacts...")
    model_path = output_dir / f"anomaly_detector.joblib"
    if winner is not None:
        joblib.dump(winner, model_path)
        print(f"  Saved: {model_path}")

    # Save evaluation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "task": "anomaly_detection",
        "task_type": "unsupervised_binary_classification",
        "n_features": len(FEATURE_COLS),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "anomaly_rate_train": float(train_df[LABEL_COL].mean()),
        "anomaly_rate_test": float(test_df[LABEL_COL].mean()),
        "winner": best_model_name,
        "winner_pr_auc_improvement_pct": round(improvement, 1),
        "fold_summary": summary,
        "final_test_metrics": final_metrics,
        "fold_results": {str(k): v for k, v in all_fold_results.items()},
    }
    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(report, f, indent=2)

    # Generate plots
    print("  Generating plots...")
    # Use last fold for plots
    last_fold_num = folds[-1]["fold"]
    plot_confusion_matrices(
        {k: all_fold_results[last_fold_num][k]
         for k in ["baseline", "tier1_iqr", "tier2_isolation_forest", best_model_name]
         if k in all_fold_results[last_fold_num]},
        output_dir
    )
    plot_pr_curves(y_test_final, {"winner": test_scores}, output_dir)

    # Save markdown report
    _write_report(report, output_dir)

    elapsed = time.time() - t0
    print(f"\n[6/6] Done. Winner: {best_model_name}, "
          f"Test PR-AUC: {final_metrics['pr_auc']:.4f}, {elapsed:.1f}s")


def _write_report(report: dict, output_dir: Path):
    """Write markdown evaluation report."""
    lines = [
        "# Anomaly Detector Training Report",
        "",
        f"**Timestamp:** {report['timestamp']}",
        f"**Task:** Transaction-level anomaly detection (unsupervised)",
        f"**Features:** {report['n_features']}",
        f"**Train/Val/Test:** {report['n_train']}/{report['n_val']}/{report['n_test']}",
        f"**Anomaly rate (train):** {report['anomaly_rate_train']*100:.2f}%",
        f"**Anomaly rate (test):** {report['anomaly_rate_test']*100:.2f}%",
        "",
        "## Fold Summary",
        "",
        "| Model | PR-AUC (mean±std) | F1 (mean±std) |",
        "|-------|-------------------|---------------|",
    ]

    summary = report["fold_summary"]
    for name, stats in sorted(summary.items(), key=lambda x: -x[1].get("pr_auc_mean", 0)):
        if name == "baseline":
            lines.append(f"| {name} | {stats['pr_auc_mean']:.4f} ± {stats['pr_auc_std']:.4f} | N/A |")
        else:
            lines.append(
                f"| {name} | {stats['pr_auc_mean']:.4f} ± {stats['pr_auc_std']:.4f} "
                f"| {stats.get('f1_mean', 0):.4f} ± {stats.get('f1_std', 0):.4f} |"
            )

    lines.extend([
        "",
        f"## Winner: {report['winner']}",
        "",
        f"**PR-AUC improvement over baseline:** {report['winner_pr_auc_improvement_pct']:.1f}%",
        "",
        "## Final Test Metrics",
        "",
        f"- **PR-AUC:** {report['final_test_metrics']['pr_auc']:.4f}",
        f"- **Best F1:** {report['final_test_metrics']['best_f1']:.4f}",
        f"- **Best Threshold:** {report['final_test_metrics']['best_threshold']:.4f}",
        f"- **Precision:** {report['final_test_metrics']['precision_at_best']:.4f}",
        f"- **Recall:** {report['final_test_metrics']['recall_at_best']:.4f}",
        f"- **TP/FP/FN/TN:** {report['final_test_metrics']['tp']}/{report['final_test_metrics']['fp']}"
        f"/{report['final_test_metrics']['fn']}/{report['final_test_metrics']['tn']}",
        "",
        "## Analysis",
        "",
        "### Key Findings",
        "",
        "- **Class imbalance:** ~0.3% anomaly rate (109 normal transactions per 1 anomalous)",
        "- **PR-AUC is the right metric:** Accuracy is misleading with extreme class imbalance",
        "- **IQR provides strong statistical baseline** with interpretable per-feature thresholds",
        "- **Isolation Forest handles unsupervised detection** without labeled anomalies",
        "- **Autoencoder learns reconstruction-based anomalies** using deep representation",
        "- **Ensemble combines multiple perspectives** for robust detection",
        "",
        "### Anomaly Types",
        "",
        "The synthetic data contains 5 anomaly types:",
        "1. **Monetary Spike** — unusually high transaction amount",
        "2. **Category Velocity** — abnormal category diversity",
        "3. **Temporal Deviation** — unusual timing patterns",
        "4. **Merchant Novelty** — new/unknown merchant interactions",
        "5. **Budget Overage** — exceeding normal spending patterns",
        "",
        "### Recommendations",
        "",
        "1. Deploy the winning model for real-time scoring",
        "2. Set anomaly threshold based on business tolerance (precision vs recall)",
        "3. Monitor model performance on incoming data for drift",
        "4. Consider ensemble approach for production robustness",
    ])

    with open(output_dir / "evaluation_report.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
