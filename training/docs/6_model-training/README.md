# Phase 6: Model Training — PFP Classifier

**Document Version:** v2.0
**Author(s):** Guevarra
**Date:** 2026-07-24
**Status:** Complete
**Purpose:** Train and compare Tier 0-4 PFP classifiers using temporal fold evaluation

---

## 1. Overview

This phase trains the **Personal Financial Profile (PFP) Classifier** — a multi-class classifier that assigns each persona to one of 8 PFP classes based on three binary dimensions (Stability × Obligation × Tolerance):

| Class | Income Stability | Obligation Weight | Financial Tolerance |
|-------|-----------------|-------------------|---------------------|
| Stable/Flexible/Tolerant | Stable (CV < threshold) | Flexible (ratio ≤ threshold) | Tolerant (runway ≥ threshold) |
| Stable/Flexible/Tight | Stable (CV < threshold) | Flexible (ratio ≤ threshold) | Tight (runway < threshold) |
| Stable/Obligated/Tolerant | Stable (CV < threshold) | Obligated (ratio > threshold) | Tolerant (runway ≥ threshold) |
| Stable/Obligated/Tight | Stable (CV < threshold) | Obligated (ratio > threshold) | Tight (runway < threshold) |
| Variable/Flexible/Tolerant | Variable (CV ≥ threshold) | Flexible (ratio ≤ threshold) | Tolerant (runway ≥ threshold) |
| Variable/Flexible/Tight | Variable (CV ≥ threshold) | Flexible (ratio ≤ threshold) | Tight (runway < threshold) |
| Variable/Obligated/Tolerant | Variable (CV ≥ threshold) | Obligated (ratio > threshold) | Tolerant (runway ≥ threshold) |
| Variable/Obligated/Tight | Variable (CV ≥ threshold) | Obligated (ratio > threshold) | Tight (runway < threshold) |

The training follows the **pre-registered outcome framing** from MDD §7: this is a hypothesis test comparing Tier 1 (rule-based) against learned tiers (Tier 2-4), with a pre-registered margin of 2 points Macro-F1.

---

## 2. Tier Definitions

### Tier 0: Naive Baseline (Sanity Floor)
- **Model:** Majority-class predictor
- **Purpose:** The absolute floor any candidate must clear
- **Implementation:** `sklearn.dummy.DummyClassifier(strategy='most_frequent')`

### Tier 1: Deterministic Rule-Based Classifier
- **Model:** Threshold rules on income stability CV and obligation ratio
- **Thresholds:** Calibrated on training fold via ROC-based cutoff selection (Youden's J)
- **Purpose:** Establishes whether learned models add value beyond deterministic rules
- **Implementation:** Custom classifier with ROC threshold calibration

### Tier 2: Intermediate Learned Model
- **Model:** Logistic Regression (multi-class, L2 regularization)
- **Purpose:** Simplest learned candidate; tests if any learning signal exists
- **Implementation:** `sklearn.linear_model.LogisticRegression(max_iter=1000, multi_class='multinomial')`

### Tier 3: Ensemble/Margin-Based Models
- **Models:** Random Forest, SVM (RBF kernel)
- **Purpose:** Test whether ensemble/non-linear methods exploit injected features better
- **Implementation:** `sklearn.ensemble.RandomForestClassifier`, `sklearn.svm.SVC(kernel='rbf')`

### Tier 4: Advanced Model
- **Models:** XGBoost Classifier (primary), MLP (if needed)
- **Purpose:** Justified only if Tiers 1-3 fail to meet KPI
- **Implementation:** `xgboost.XGBClassifier`

---

## 3. Evaluation Methodology

### 3.1 Temporal Fold Evaluation
Using the 5-fold expanding window from `temporal_folds.json`:
```
Fold 1: Train [1-6], Embargo [7], Test [8]
Fold 2: Train [1-7], Embargo [8], Test [9]
Fold 3: Train [1-8], Embargo [9], Test [10]
Fold 4: Train [1-9], Embargo [10], Test [11]
Fold 5: Train [1-10], Embargo [11], Test [12]
```

### 3.2 Metrics
- **Primary:** Macro-F1 Score (handles class imbalance, treats all classes equally)
- **Secondary:** Accuracy, per-class precision/recall/F1, confusion matrix

### 3.3 Pre-Registered Decision Rule
```python
PRE_REGISTERED_MARGIN = 0.02  # 2 points of Macro-F1

if best_learned_f1 - tier1_f1 > PRE_REGISTERED_MARGIN:
    winner = best_learned_model  # Learned model wins
else:
    winner = "tier1_rule_based"  # Rule-based wins (valid result)
```

Ties favor the simpler/more interpretable model (Tier 1 over Tier 3/4).

### 3.4 Per-Class Accuracy Requirement
Must maintain >75% per-class accuracy across all PFP classes.

---

## 4. Input/Output

### Input
- `datasets/engineered/train.parquet` — Engineered features (train split)
- `datasets/engineered/val.parquet` — Engineered features (validation split)
- `datasets/engineered/test.parquet` — Engineered features (test split)
- `datasets/engineered/feature_columns.json` — Feature column definitions
- `datasets/processed/temporal_folds.json` — Temporal fold definitions

### Output
- `models/fbp/` — Trained model artifacts
  - `tier0_majority.joblib` — Tier 0 model
  - `tier1_rule_based.joblib` — Tier 1 model (thresholds)
  - `tier2_logistic_regression.joblib` — Tier 2 model
  - `tier3_random_forest.joblib` — Tier 3 RF model
  - `tier3_svm.joblib` — Tier 3 SVM model
  - `tier4_xgboost.joblib` — Tier 4 XGB model
- `models/fbp/evaluation.json` — Cross-fold metrics for all tiers
- `models/fbp/evaluation_report.md` — Human-readable comparison report
- `models/fbp/confusion_matrices/` — Confusion matrix plots per fold

---

## 5. Usage

```bash
# Train all tiers and evaluate
python scripts/train_fbp.py --input datasets/engineered/ --output models/fbp/

# Train with custom options
python scripts/train_fbp.py \
  --input datasets/engineered/ \
  --output models/fbp/ \
  --temporal-folds datasets/processed/temporal_folds.json \
  --seed 42
```

---

## 6. Feature Sets per Tier

| Tier | Features Used |
|------|--------------|
| Tier 0 | None (majority class) |
| Tier 1 | `income_stability_cv`, `obligation_ratio` (2 features) |
| Tier 2 | All 19 engineered + 11 raw features (30 features) |
| Tier 3 | All 30 features |
| Tier 4 | All 30 features |

---

## 7. Dependencies

- scikit-learn ( classifiers, metrics, preprocessing)
- xgboost (Tier 4)
- joblib (model serialization)
- numpy, pandas (data handling)
- matplotlib, seaborn (confusion matrix plots)

---

## 8. Forecaster Training

See [forecaster-training.md](forecaster-training.md) for full details.

**Summary:** Random Forest (Tier 2) wins at MAPE 9.63%, R² 0.831, beating naive baseline by ~77% MAPE reduction. LSTM/GRU/BiLSTM skipped due to TensorFlow unavailability on this machine.

## 9. Anomaly Detector Training

See [anomaly-training.md](anomaly-training.md) for full details.

**Summary:** One-Class SVM (Tier 2) wins at PR-AUC 0.1478 (416.8% improvement over baseline), with ~3% anomaly rate in synthetic data (corrected from 0.3%). Final test PR-AUC: 0.1215.

---

*Document created: 2026-07-23*
*Updated: 2026-07-24 — added forecaster and anomaly detector results*
