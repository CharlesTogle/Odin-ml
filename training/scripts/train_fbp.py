"""
PFP Classifier Training Pipeline for Odin ML

Trains and compares Tier 0-4 classifiers for Personal Financial Profile (PFP)
classification using temporal fold evaluation.

Tiers:
  Tier 0: Majority-class baseline (sanity floor)
  Tier 1: Rule-based classifier (ROC-calibrated thresholds)
  Tier 2: Logistic Regression (L2, multi-class)
  Tier 3: Random Forest, SVM (RBF kernel)
  Tier 4: XGBoost Classifier

Evaluation:
  - 5-fold expanding window (temporal_folds.json)
  - Primary metric: Macro-F1 Score
  - Pre-registered margin: 2 points Macro-F1 (Tier 1 vs best learned)

Usage:
    python scripts/train_fbp.py --input datasets/engineered/ --output models/fbp/
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("xgboost not installed — Tier 4 (XGBoost) will be skipped")

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

PFP_CLASSES = [
    "Stable/Obligated/Tolerant",
    "Stable/Obligated/Tight",
    "Stable/Flexible/Tolerant",
    "Stable/Flexible/Tight",
    "Variable/Obligated/Tolerant",
    "Variable/Obligated/Tight",
    "Variable/Flexible/Tolerant",
    "Variable/Flexible/Tight",
]

PFP_PALETTE = {
    "Stable/Obligated/Tolerant": "#3498db",
    "Stable/Obligated/Tight":    "#2980b9",
    "Stable/Flexible/Tolerant":  "#2ecc71",
    "Stable/Flexible/Tight":     "#27ae60",
    "Variable/Obligated/Tolerant": "#e67e22",
    "Variable/Obligated/Tight":    "#d35400",
    "Variable/Flexible/Tolerant":  "#f39c12",
    "Variable/Flexible/Tight":     "#e74c3c",
}

PRE_REGISTERED_MARGIN = 0.02  # 2 points of Macro-F1

META_COLUMNS = ["user_id", "month", "fbp_label", "runway_months", "financial_tolerance",
                "is_anomalous", "anomaly_type"]

RAW_COLUMNS = [
    "total_income", "total_expenses", "food_expense", "housing_expense",
    "transport_expense", "health_expense", "education_expense", "other_expense",
    "savings", "debt_payment", "transaction_count",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    train_months: list[int]
    test_months: list[int]
    n_train_personas: int
    n_test_personas: int
    tier_results: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class TrainingReport:
    timestamp: str
    n_folds: int
    pre_registered_margin: float
    fold_results: list[FoldResult] = field(default_factory=list)
    aggregate_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    winner: str = ""
    winner_reason: str = ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_engineered_data(input_dir: str) -> dict[str, pd.DataFrame]:
    """Load engineered train/val/test parquets."""
    input_path = Path(input_dir)
    splits = {}
    for name in ["train", "val", "test"]:
        path = input_path / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Engineered data not found: {path}")
        splits[name] = pd.read_parquet(path)
    return splits


def load_temporal_folds(folds_path: str) -> list[dict]:
    """Load temporal fold definitions."""
    with open(folds_path) as f:
        return json.load(f)


def load_feature_columns(input_dir: str) -> dict:
    """Load feature column definitions."""
    path = Path(input_dir) / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"Feature columns not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_split_metadata(input_dir: str) -> dict:
    """Load persona-level split metadata."""
    processed_dir = Path(input_dir).parent / "processed"
    path = processed_dir / "split_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Persona-level aggregation
# ---------------------------------------------------------------------------

def aggregate_to_personas(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Aggregate monthly rows to one row per persona.

    Uses vectorized pandas groupby for performance.
    Returns:
        personas_df: DataFrame with one row per persona (aggregated features + label)
        label_map: mapping from persona_id to fbp_label (majority vote)
    """
    agg_cols = [c for c in feature_cols if c in df.columns]

    # Vectorized mean aggregation
    personas_df = df.groupby("user_id")[agg_cols].mean().reset_index()

    # Majority vote for label
    label_counts = df.groupby(["user_id", "fbp_label"]).size().reset_index(name="count")
    idx = label_counts.groupby("user_id")["count"].idxmax()
    labels = label_counts.loc[idx, ["user_id", "fbp_label"]]
    personas_df = personas_df.merge(labels, on="user_id", how="left")

    label_map = dict(zip(personas_df["user_id"], personas_df["fbp_label"]))
    return personas_df, label_map


def aggregate_to_personas_month_filter(
    df: pd.DataFrame, feature_cols: list[str], months: list[int]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Aggregate monthly rows to one row per persona, filtered to specific months."""
    filtered = df[df["month"].isin(months)]
    if len(filtered) == 0:
        return pd.DataFrame(), {}
    return aggregate_to_personas(filtered, feature_cols)


def prepare_feature_matrix(
    personas_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix X and label vector y from persona DataFrame.

    Returns:
        X: feature matrix (n_personas, n_features)
        y: label array (n_personas,)
        feature_names: list of feature column names used
    """
    available = [c for c in feature_cols if c in personas_df.columns]
    X = personas_df[available].values
    y = personas_df["fbp_label"].values
    return X, y, available


# ---------------------------------------------------------------------------
# Tier 1: Rule-based classifier
# ---------------------------------------------------------------------------

class RuleBasedClassifier:
    """Deterministic rule-based PFP classifier with ROC-calibrated thresholds.

    Calibrates CV, obligation ratio, and runway thresholds on training data using
    Youden's J statistic (maximizes TPR - FPR) for each dimension independently.
    """

    def __init__(self):
        self.cv_threshold = 0.50   # default (stable vs volatile)
        self.obl_threshold = 0.60  # default (obligated vs flexible)
        self.runway_threshold = 3.0  # default (tight vs tolerant, months)
        self._fitted = False

    def _calibrate_threshold(
        self, scores: np.ndarray, positive_mask: np.ndarray,
        candidate_range: tuple[float, float] = (0.05, 0.80), step: float = 0.01
    ) -> float:
        """Find optimal threshold via Youden's J statistic."""
        thresholds = np.arange(candidate_range[0], candidate_range[1], step)
        best_j = -1
        best_threshold = np.median(thresholds)

        for t in thresholds:
            predicted_positive = scores >= t
            tp = np.sum(predicted_positive & positive_mask)
            fn = np.sum(~predicted_positive & positive_mask)
            fp = np.sum(predicted_positive & ~positive_mask)
            tn = np.sum(~predicted_positive & ~positive_mask)

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            j = tpr - fpr

            if j > best_j:
                best_j = j
                best_threshold = t

        return float(best_threshold)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        """Calibrate thresholds on training data."""
        cv_idx = feature_names.index("income_stability_cv")
        obl_idx = feature_names.index("obligation_ratio")

        cv_scores = X[:, cv_idx]
        obl_scores = X[:, obl_idx]

        # Income stability: Stable = CV < threshold (lower CV = more stable)
        stable_mask = np.array([label.startswith("Stable") for label in y])
        self.cv_threshold = self._calibrate_threshold(
            cv_scores, stable_mask,
            candidate_range=(0.05, 0.80), step=0.01
        )

        # Obligation: Obligated = ratio > threshold
        obligated_mask = np.array(["Obligated" in label for label in y])
        self.obl_threshold = self._calibrate_threshold(
            obl_scores, obligated_mask,
            candidate_range=(0.20, 0.90), step=0.01
        )

        # Runway tolerance: Tolerant = runway >= threshold (months)
        # Higher runway = more tolerant; positive_mask = Tolerant
        if "runway_months" in feature_names:
            runway_idx = feature_names.index("runway_months")
            runway_scores = X[:, runway_idx]
            tolerant_mask = np.array(["Tolerant" in label for label in y])
            self.runway_threshold = self._calibrate_threshold(
                runway_scores, tolerant_mask,
                candidate_range=(1.0, 8.0), step=0.5
            )

        self._fitted = True
        return self

    def predict(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """Predict PFP class using calibrated thresholds."""
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
        for cv, obl, runway in zip(cv_scores, obl_scores, runway_scores):
            stability = "Stable" if cv < self.cv_threshold else "Variable"
            obligation = "Obligated" if obl > self.obl_threshold else "Flexible"
            tolerance = "Tolerant" if runway >= self.runway_threshold else "Tight"
            predictions.append(f"{stability}/{obligation}/{tolerance}")

        return np.array(predictions)

    def get_params(self) -> dict:
        return {
            "cv_threshold": self.cv_threshold,
            "obl_threshold": self.obl_threshold,
            "runway_threshold": self.runway_threshold,
        }


# ---------------------------------------------------------------------------
# Tier training and evaluation
# ---------------------------------------------------------------------------

def train_tier0(X_train: np.ndarray, y_train: np.ndarray) -> DummyClassifier:
    """Tier 0: Majority-class baseline."""
    model = DummyClassifier(strategy="most_frequent", random_state=42)
    model.fit(X_train, y_train)
    return model


def train_tier1(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> RuleBasedClassifier:
    """Tier 1: Rule-based classifier with ROC-calibrated thresholds."""
    model = RuleBasedClassifier()
    model.fit(X_train, y_train, feature_names)
    return model


def train_tier2(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Tier 2: Logistic Regression (L2, multi-class)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        max_iter=1000, random_state=42
    )
    model.fit(X_scaled, y_train)
    return {"model": model, "scaler": scaler}


def train_tier3_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Tier 3: Random Forest."""
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_tier3_svm(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Tier 3: SVM (RBF kernel) with subsampling for speed."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Subsample if dataset is large (SVM is O(n²) to O(n³))
    max_samples = 3000
    if len(X_scaled) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_scaled), max_samples, replace=False)
        X_sub = X_scaled[idx]
        y_sub = y_train[idx]
    else:
        X_sub = X_scaled
        y_sub = y_train

    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_sub, y_sub)
    return {"model": model, "scaler": scaler}


def train_tier4_xgb(X_train: np.ndarray, y_train: np.ndarray) -> Optional[Any]:
    """Tier 4: XGBoost Classifier."""
    if not HAS_XGBOOST:
        return None
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_encoded)
    return {"model": model, "label_encoder": le}


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_tier0(model: DummyClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def predict_tier1(model: RuleBasedClassifier, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    return model.predict(X, feature_names)


def predict_tier2(tier2: dict, X: np.ndarray) -> np.ndarray:
    X_scaled = tier2["scaler"].transform(X)
    return tier2["model"].predict(X_scaled)


def predict_tier3_rf(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def predict_tier3_svm(tier3_svm: dict, X: np.ndarray) -> np.ndarray:
    X_scaled = tier3_svm["scaler"].transform(X)
    return tier3_svm["model"].predict(X_scaled)


def predict_tier4_xgb(tier4: dict, X: np.ndarray) -> np.ndarray:
    if tier4 is None:
        return np.array([])
    le = tier4["label_encoder"]
    y_encoded = tier4["model"].predict(X)
    return le.inverse_transform(y_encoded)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, tier_name: str
) -> dict[str, Any]:
    """Compute metrics for a single tier on a single fold."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    per_class = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    # Per-class accuracy
    per_class_acc = {}
    for cls in PFP_CLASSES:
        mask = y_true == cls
        if mask.sum() > 0:
            per_class_acc[cls] = float(np.mean(y_pred[mask] == cls))
        else:
            per_class_acc[cls] = 0.0

    return {
        "tier": tier_name,
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "per_class_accuracy": per_class_acc,
        "classification_report": per_class,
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, tier_name: str, fold: int,
    output_dir: Path
):
    """Save confusion matrix plot."""
    if not HAS_MATPLOTLIB:
        return

    available_classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=available_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=available_classes, yticklabels=available_classes, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {tier_name} (Fold {fold})")
    plt.tight_layout()

    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_dir / f"{tier_name}_fold{fold}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training(
    input_dir: str = "datasets/engineered/",
    output_dir: str = "models/fbp/",
    temporal_folds_path: Optional[str] = None,
    seed: int = 42,
) -> TrainingReport:
    """Run the full Tier 0-4 training pipeline."""
    print("=" * 60)
    print("Odin ML — PFP Classifier Training (Tier 0-4)")
    print("=" * 60)

    # ---- Load data ----
    print("\n[1/6] Loading engineered data...")
    splits = load_engineered_data(input_dir)
    feature_meta = load_feature_columns(input_dir)
    feature_cols = feature_meta.get("feature_columns", [])

    print(f"  Train: {len(splits['train']):,} rows")
    print(f"  Val:   {len(splits['val']):,} rows")
    print(f"  Test:  {len(splits['test']):,} rows")
    print(f"  Features: {len(feature_cols)}")

    # ---- Load temporal folds ----
    if temporal_folds_path is None:
        folds_path = Path(input_dir).parent / "processed" / "temporal_folds.json"
        if not folds_path.exists():
            folds_path = Path("datasets/processed/temporal_folds.json")
    else:
        folds_path = Path(temporal_folds_path)

    folds = load_temporal_folds(str(folds_path))
    print(f"  Temporal folds: {len(folds)}")

    # ---- Load persona-level split metadata ----
    split_meta = load_split_metadata(input_dir)
    train_pids = set(split_meta.get("personas", {}).get("train", []))
    val_pids = set(split_meta.get("personas", {}).get("val", []))
    test_pids = set(split_meta.get("personas", {}).get("test", []))

    if not train_pids:
        # Fallback: derive from data
        all_pids = set(splits["train"]["user_id"].unique())
        n = len(all_pids)
        pid_list = sorted(all_pids)
        train_pids = set(pid_list[:int(n * 0.7)])
        val_pids = set(pid_list[int(n * 0.7):int(n * 0.85)])
        test_pids = set(pid_list[int(n * 0.85):])

    print(f"  Train personas: {len(train_pids):,}")
    print(f"  Val personas:   {len(val_pids):,}")
    print(f"  Test personas:  {len(test_pids):,}")

    # ---- Pre-filter data splits by persona IDs ----
    train_data = splits["train"][splits["train"]["user_id"].isin(train_pids)]
    val_data = splits["val"][splits["val"]["user_id"].isin(val_pids)]
    test_data = splits["test"][splits["test"]["user_id"].isin(test_pids)]

    # ---- Aggregate full-feature persona matrices (for final model training) ----
    print("\n[2/6] Aggregating to persona-level (full features)...")
    train_personas, _ = aggregate_to_personas(train_data, feature_cols)
    val_personas, _ = aggregate_to_personas(val_data, feature_cols)
    test_personas, _ = aggregate_to_personas(test_data, feature_cols)

    X_train_all, y_train_all, used_features = prepare_feature_matrix(train_personas, feature_cols)
    X_val_all, y_val_all, _ = prepare_feature_matrix(val_personas, feature_cols)
    X_test_all, y_test_all, _ = prepare_feature_matrix(test_personas, feature_cols)

    print(f"  Feature matrix shape: {X_train_all.shape}")

    # ---- Train and evaluate per fold ----
    print("\n[3/6] Training and evaluating per temporal fold...")
    report = TrainingReport(
        timestamp=datetime.now().isoformat(),
        n_folds=len(folds),
        pre_registered_margin=PRE_REGISTERED_MARGIN,
    )

    tier_names = ["tier0_majority", "tier1_rule_based", "tier2_logistic_regression",
                  "tier3_random_forest", "tier3_svm"]
    if HAS_XGBOOST:
        tier_names.append("tier4_xgboost")

    fold_macro_f1s = {t: [] for t in tier_names}
    fold_accuracies = {t: [] for t in tier_names}

    for fold_info in folds:
        fold_num = fold_info["fold"]
        train_months = fold_info["train_months"]
        test_months = fold_info["test_months"]

        print(f"\n  Fold {fold_num}: Train months {train_months}, Test months {test_months}")

        # Aggregate train personas using train_months only (feature computation window)
        fold_train_personas, _ = aggregate_to_personas_month_filter(
            train_data, feature_cols, train_months
        )
        # Aggregate val personas using test_months (held-out personas, different time window)
        fold_val_personas, _ = aggregate_to_personas_month_filter(
            val_data, feature_cols, test_months
        )

        if len(fold_train_personas) == 0 or len(fold_val_personas) == 0:
            print(f"    Skipping fold {fold_num} — insufficient data")
            continue

        X_ft, y_ft, _ = prepare_feature_matrix(fold_train_personas, feature_cols)
        X_fv, y_fv, _ = prepare_feature_matrix(fold_val_personas, feature_cols)

        print(f"    Train: {len(fold_train_personas)} personas (months {train_months})")
        print(f"    Val:   {len(fold_val_personas)} personas (months {test_months})")

        fold_result = FoldResult(
            fold=fold_num,
            train_months=train_months,
            test_months=test_months,
            n_train_personas=len(fold_train_personas),
            n_test_personas=len(fold_val_personas),
        )

        # Tier 0
        t0 = train_tier0(X_ft, y_ft)
        y_pred_t0 = predict_tier0(t0, X_fv)
        metrics_t0 = evaluate_model(y_fv, y_pred_t0, "tier0_majority")
        fold_result.tier_results["tier0_majority"] = metrics_t0
        fold_macro_f1s["tier0_majority"].append(metrics_t0["macro_f1"])
        fold_accuracies["tier0_majority"].append(metrics_t0["accuracy"])
        plot_confusion_matrix(y_fv, y_pred_t0, "tier0_majority", fold_num, Path(output_dir))
        print(f"    Tier 0 (Majority):       Macro-F1={metrics_t0['macro_f1']:.4f}  Acc={metrics_t0['accuracy']:.4f}")

        # Tier 1
        t1 = train_tier1(X_ft, y_ft, used_features)
        y_pred_t1 = predict_tier1(t1, X_fv, used_features)
        metrics_t1 = evaluate_model(y_fv, y_pred_t1, "tier1_rule_based")
        fold_result.tier_results["tier1_rule_based"] = metrics_t1
        fold_result.tier_results["tier1_thresholds"] = t1.get_params()
        fold_macro_f1s["tier1_rule_based"].append(metrics_t1["macro_f1"])
        fold_accuracies["tier1_rule_based"].append(metrics_t1["accuracy"])
        plot_confusion_matrix(y_fv, y_pred_t1, "tier1_rule_based", fold_num, Path(output_dir))
        print(f"    Tier 1 (Rule-Based):     Macro-F1={metrics_t1['macro_f1']:.4f}  Acc={metrics_t1['accuracy']:.4f}"
              f"  CV_th={t1.cv_threshold:.3f}  Obl_th={t1.obl_threshold:.3f}  Runway_th={t1.runway_threshold:.1f}")

        # Tier 2
        t2 = train_tier2(X_ft, y_ft)
        y_pred_t2 = predict_tier2(t2, X_fv)
        metrics_t2 = evaluate_model(y_fv, y_pred_t2, "tier2_logistic_regression")
        fold_result.tier_results["tier2_logistic_regression"] = metrics_t2
        fold_macro_f1s["tier2_logistic_regression"].append(metrics_t2["macro_f1"])
        fold_accuracies["tier2_logistic_regression"].append(metrics_t2["accuracy"])
        plot_confusion_matrix(y_fv, y_pred_t2, "tier2_logistic_regression", fold_num, Path(output_dir))
        print(f"    Tier 2 (Logistic Reg):   Macro-F1={metrics_t2['macro_f1']:.4f}  Acc={metrics_t2['accuracy']:.4f}")

        # Tier 3: Random Forest
        t3_rf = train_tier3_rf(X_ft, y_ft)
        y_pred_t3_rf = predict_tier3_rf(t3_rf, X_fv)
        metrics_t3_rf = evaluate_model(y_fv, y_pred_t3_rf, "tier3_random_forest")
        fold_result.tier_results["tier3_random_forest"] = metrics_t3_rf
        fold_macro_f1s["tier3_random_forest"].append(metrics_t3_rf["macro_f1"])
        fold_accuracies["tier3_random_forest"].append(metrics_t3_rf["accuracy"])
        plot_confusion_matrix(y_fv, y_pred_t3_rf, "tier3_random_forest", fold_num, Path(output_dir))
        print(f"    Tier 3 (Random Forest):  Macro-F1={metrics_t3_rf['macro_f1']:.4f}  Acc={metrics_t3_rf['accuracy']:.4f}")

        # Tier 3: SVM
        t3_svm = train_tier3_svm(X_ft, y_ft)
        y_pred_t3_svm = predict_tier3_svm(t3_svm, X_fv)
        metrics_t3_svm = evaluate_model(y_fv, y_pred_t3_svm, "tier3_svm")
        fold_result.tier_results["tier3_svm"] = metrics_t3_svm
        fold_macro_f1s["tier3_svm"].append(metrics_t3_svm["macro_f1"])
        fold_accuracies["tier3_svm"].append(metrics_t3_svm["accuracy"])
        plot_confusion_matrix(y_fv, y_pred_t3_svm, "tier3_svm", fold_num, Path(output_dir))
        print(f"    Tier 3 (SVM):            Macro-F1={metrics_t3_svm['macro_f1']:.4f}  Acc={metrics_t3_svm['accuracy']:.4f}")

        # Tier 4: XGBoost
        if HAS_XGBOOST:
            t4 = train_tier4_xgb(X_ft, y_ft)
            y_pred_t4 = predict_tier4_xgb(t4, X_fv)
            if len(y_pred_t4) > 0:
                metrics_t4 = evaluate_model(y_fv, y_pred_t4, "tier4_xgboost")
                fold_result.tier_results["tier4_xgboost"] = metrics_t4
                fold_macro_f1s["tier4_xgboost"].append(metrics_t4["macro_f1"])
                fold_accuracies["tier4_xgboost"].append(metrics_t4["accuracy"])
                plot_confusion_matrix(y_fv, y_pred_t4, "tier4_xgboost", fold_num, Path(output_dir))
                print(f"    Tier 4 (XGBoost):        Macro-F1={metrics_t4['macro_f1']:.4f}  Acc={metrics_t4['accuracy']:.4f}")

        report.fold_results.append(fold_result)

    # ---- Aggregate across folds ----
    print("\n[4/6] Aggregating metrics across folds...")
    for tier_name in tier_names:
        f1s = fold_macro_f1s[tier_name]
        accs = fold_accuracies[tier_name]
        report.aggregate_metrics[tier_name] = {
            "macro_f1_mean": float(np.mean(f1s)) if f1s else 0.0,
            "macro_f1_std": float(np.std(f1s)) if f1s else 0.0,
            "accuracy_mean": float(np.mean(accs)) if accs else 0.0,
            "accuracy_std": float(np.std(accs)) if accs else 0.0,
        }

    # ---- Pre-registered decision ----
    print("\n[5/6] Applying pre-registered decision rule...")
    tier1_f1 = report.aggregate_metrics["tier1_rule_based"]["macro_f1_mean"]

    learned_tiers = {k: v for k, v in report.aggregate_metrics.items()
                     if k.startswith("tier") and k != "tier0_majority" and k != "tier1_rule_based"}

    if learned_tiers:
        best_learned_tier = max(learned_tiers, key=lambda k: learned_tiers[k]["macro_f1_mean"])
        best_learned_f1 = learned_tiers[best_learned_tier]["macro_f1_mean"]
        margin = best_learned_f1 - tier1_f1

        if margin > PRE_REGISTERED_MARGIN:
            report.winner = best_learned_tier
            report.winner_reason = (
                f"Best learned model ({best_learned_tier}) beats Tier 1 by "
                f"{margin:.4f} (> {PRE_REGISTERED_MARGIN} margin)"
            )
        else:
            report.winner = "tier1_rule_based"
            report.winner_reason = (
                f"No learned model exceeds Tier 1 by >{PRE_REGISTERED_MARGIN} margin. "
                f"Best learned: {best_learned_tier} (F1={best_learned_f1:.4f}), "
                f"Tier 1: {tier1_f1:.4f}, gap={margin:.4f}"
            )
    else:
        report.winner = "tier1_rule_based"
        report.winner_reason = "No learned models evaluated"

    print(f"  Winner: {report.winner}")
    print(f"  Reason: {report.winner_reason}")

    # Print summary table
    print("\n  Aggregate Results:")
    print(f"  {'Tier':<30} {'Macro-F1':>10} {'±':>5} {'Accuracy':>10} {'±':>5}")
    print("  " + "-" * 65)
    for tier_name in tier_names:
        m = report.aggregate_metrics[tier_name]
        print(f"  {tier_name:<30} {m['macro_f1_mean']:>10.4f} {m['macro_f1_std']:>5.4f} "
              f"{m['accuracy_mean']:>10.4f} {m['accuracy_std']:>5.4f}")

    # ---- Save models ----
    print("\n[6/6] Saving models and reports...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cm_dir = output_path / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    # Re-train final models on full training data and save
    print("  Training final models on full training data...")
    t0_final = train_tier0(X_train_all, y_train_all)
    joblib.dump(t0_final, output_path / "tier0_majority.joblib")

    t1_final = train_tier1(X_train_all, y_train_all, used_features)
    joblib.dump({
        "cv_threshold": t1_final.cv_threshold,
        "obl_threshold": t1_final.obl_threshold,
        "runway_threshold": t1_final.runway_threshold,
        "feature_names": used_features,
    }, output_path / "tier1_rule_based.joblib")

    t2_final = train_tier2(X_train_all, y_train_all)
    joblib.dump(t2_final, output_path / "tier2_logistic_regression.joblib")

    t3_rf_final = train_tier3_rf(X_train_all, y_train_all)
    joblib.dump(t3_rf_final, output_path / "tier3_random_forest.joblib")

    t3_svm_final = train_tier3_svm(X_train_all, y_train_all)
    joblib.dump(t3_svm_final, output_path / "tier3_svm.joblib")

    if HAS_XGBOOST:
        t4_final = train_tier4_xgb(X_train_all, y_train_all)
        if t4_final is not None:
            joblib.dump(t4_final, output_path / "tier4_xgboost.joblib")

    # Save evaluation JSON
    eval_data = {
        "timestamp": report.timestamp,
        "n_folds": report.n_folds,
        "pre_registered_margin": report.pre_registered_margin,
        "aggregate_metrics": report.aggregate_metrics,
        "winner": report.winner,
        "winner_reason": report.winner_reason,
        "feature_columns": used_features,
        "fold_details": [
            {
                "fold": fr.fold,
                "train_months": fr.train_months,
                "test_months": fr.test_months,
                "n_train_personas": fr.n_train_personas,
                "n_test_personas": fr.n_test_personas,
                "tier_results": {
                    k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
                    for k, v in fr.tier_results.items()
                },
            }
            for fr in report.fold_results
        ],
    }
    with open(output_path / "evaluation.json", "w") as f:
        json.dump(eval_data, f, indent=2, default=str)

    # Save human-readable report
    _write_evaluation_report(report, tier_names, output_path)

    print(f"\n  Saved to {output_dir}/")
    for item in sorted(output_path.iterdir()):
        if item.is_file():
            print(f"    {item.name} ({item.stat().st_size:,} bytes)")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return report


def _write_evaluation_report(
    report: TrainingReport, tier_names: list[str], output_path: Path
):
    """Write human-readable evaluation report."""
    lines = [
        "# PFP Classifier — Evaluation Report",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Folds:** {report.n_folds}",
        f"**Pre-registered margin:** {report.pre_registered_margin}",
        "",
        "---",
        "",
        "## Winner",
        "",
        f"**{report.winner}**",
        "",
        f"> {report.winner_reason}",
        "",
        "---",
        "",
        "## Aggregate Results",
        "",
        "| Tier | Macro-F1 (mean ± std) | Accuracy (mean ± std) |",
        "|------|----------------------|----------------------|",
    ]

    for tier_name in tier_names:
        m = report.aggregate_metrics.get(tier_name, {})
        f1_mean = m.get("macro_f1_mean", 0)
        f1_std = m.get("macro_f1_std", 0)
        acc_mean = m.get("accuracy_mean", 0)
        acc_std = m.get("accuracy_std", 0)
        lines.append(
            f"| {tier_name} | {f1_mean:.4f} ± {f1_std:.4f} | {acc_mean:.4f} ± {acc_std:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Per-Fold Results",
        "",
    ])

    for fr in report.fold_results:
        lines.append(f"### Fold {fr.fold}")
        lines.append("")
        lines.append(f"- Train months: {fr.train_months}")
        lines.append(f"- Test months: {fr.test_months}")
        lines.append(f"- Train personas: {fr.n_train_personas}")
        lines.append(f"- Test personas: {fr.n_test_personas}")
        lines.append("")
        lines.append("| Tier | Macro-F1 | Accuracy |")
        lines.append("|------|----------|----------|")
        for tier_name, metrics in fr.tier_results.items():
            if isinstance(metrics, dict) and "macro_f1" in metrics:
                lines.append(
                    f"| {tier_name} | {metrics['macro_f1']:.4f} | {metrics['accuracy']:.4f} |"
                )
        lines.append("")

    # Per-class accuracy from last fold
    if report.fold_results:
        last_fold = report.fold_results[-1]
        short_classes = [c.split("/")[0][0] + c.split("/")[1][0] + c.split("/")[2][0] for c in PFP_CLASSES]
        lines.extend([
            "---",
            "",
            "## Per-Class Accuracy (Last Fold)",
            "",
            "| Tier | " + " | ".join(PFP_CLASSES) + " |",
            "|------|" + "|".join(["------"] * len(PFP_CLASSES)) + "|",
        ])
        for tier_name in tier_names:
            metrics = last_fold.tier_results.get(tier_name, {})
            if isinstance(metrics, dict) and "per_class_accuracy" in metrics:
                accs = [f"{metrics['per_class_accuracy'].get(c, 0):.4f}" for c in PFP_CLASSES]
                lines.append(f"| {tier_name} | " + " | ".join(accs) + " |")
        lines.append("")

    with open(output_path / "evaluation_report.md", "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Odin ML — PFP Classifier Training (Tier 0-4)"
    )
    parser.add_argument(
        "--input", default="datasets/engineered/",
        help="Input directory with engineered data (default: datasets/engineered/)"
    )
    parser.add_argument(
        "--output", default="models/fbp/",
        help="Output directory for trained models (default: models/fbp/)"
    )
    parser.add_argument(
        "--temporal-folds", default=None,
        help="Path to temporal_folds.json (default: auto-detect)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()
    run_training(
        input_dir=args.input,
        output_dir=args.output,
        temporal_folds_path=args.temporal_folds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
