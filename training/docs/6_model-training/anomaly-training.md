# Anomaly Detector Training Documentation

## Overview

This document describes the training pipeline for the Isolation Forest Anomaly Detector module, which detects anomalous transactions at the individual transaction level using unsupervised and semi-supervised methods.

**MDD Reference:** `module-design-document.md` lines 771-1104 (Model Design Document - Anomaly Detector)

## Feature Engineering

**Script:** `scripts/feature_engineering_anomaly.py`  
**Output:** `datasets/anomaly/{train,val,test}.parquet`

### Features (24 per transaction)

| Category | Features | Count |
|----------|----------|-------|
| Baseline | mean/std income/expenses, category_dist, txn_frequency, avg_txn_size, category_entropy, volatility_index, spending_concentration | 10 |
| Detection | amount_deviation, category_deviation, frequency_deviation, income_deviation, expense_deviation, is_novel_category, amount_vs_category_mean, amount_vs_category_std, category_frequency_change, amount_percentile_in_category, days_since_last_txn, is_weekend, amount_zscore_overall, amount_zscore_category | 14 |

### Pipeline

1. Load `synth/transactions.parquet` (35,568 rows, 300 personas, 12 months)
2. Build per-persona category history (monthly counts, amounts by category)
3. For each transaction, compute baseline stats from previous months
4. Compute detection features (deviations, novelty flags, z-scores)
5. Standardize features using train-only statistics
6. Export train/val/test splits

### Dataset Statistics

| Split | Rows | Anomaly Rate |
|-------|------|-------------|
| Train | 25,135 | 2.94% |
| Val | 4,980 | 2.87% |
| Test | 5,446 | 3.23% |

**Total features:** 24  
**Anomaly types in synthetic data:** amount_spike, category_mismatch, frequency_change, new_merchant

## Training Pipeline

**Script:** `scripts/train_anomaly.py`  
**Output:** `models/anomaly/`

### Model Tiers

| Tier | Model | Configuration |
|------|-------|---------------|
| Tier 0 | Majority-class baseline | Always predict normal |
| Tier 1 | IQR Detector | iqr_multiplier=1.5, per-feature bounds |
| Tier 2 | Isolation Forest | n_estimators=200, contamination=0.003 |
| Tier 2 | One-Class SVM | RBF kernel, gamma=scale, nu=0.01 |
| Tier 2 | Autoencoder (Keras) | 24→7→3→7→24 architecture, MSE loss |
| Tier 3 | Hybrid Ensemble | Average of IQR + IF + OCSVM scores |

### Walk-Forward Validation

- **Strategy:** Expanding window, 5 folds
- **Primary metric:** PR-AUC (area under precision-recall curve)
- **Secondary:** F1 at optimal threshold, precision, recall

| Fold | Train Months | Test Month |
|------|-------------|------------|
| 1 | [1-6] | 8 |
| 2 | [1-7] | 9 |
| 3 | [1-8] | 10 |
| 4 | [1-9] | 11 |
| 5 | [1-10] | 12 |

### Fold Results

| Model | PR-AUC (mean±std) | F1 (mean±std) |
|-------|-------------------|---------------|
| **tier2_ocsvm** | **0.1478 ± 0.0358** | **0.2401 ± 0.0366** |
| tier3_ensemble | 0.1113 ± 0.0293 | 0.1876 ± 0.0228 |
| tier2_isolation_forest | 0.0958 ± 0.0314 | 0.1880 ± 0.0345 |
| tier1_iqr | 0.0569 ± 0.0113 | 0.1261 ± 0.0293 |
| baseline | 0.0286 ± 0.0025 | N/A |

### Final Test Set Results (Winner: OCSVM)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.1215 |
| Best F1 | 0.1780 |
| Best Threshold | 0.4726 |
| Precision | 0.1863 |
| Recall | 0.1705 |
| TP/FP/FN/TN | 30/131/146/5139 |
| PR-AUC improvement over baseline | 416.8% |

### Analysis

#### Key Findings

1. **OCSVM wins** with PR-AUC 0.1478 (416.8% improvement over baseline) by learning the normal transaction boundary in feature space
2. **~3% anomaly rate** (109 normal per 1 anomalous) — much improved from 0.3% initial rate
3. **IQR provides interpretable per-feature bounds** but misses multi-feature anomalies
4. **Ensemble** doesn't outperform individual OCSVM, suggesting the models are redundant
5. **Autoencoder skipped** — TensorFlow unavailable on this machine (no GPU)

#### Anomaly Type Detection

| Type | Count | Detectability |
|------|-------|---------------|
| amount_spike | High | Clear z-score signal |
| category_mismatch | Medium | Novel categories (luxury/gambling/investment) |
| frequency_change | Low | Subtle behavioral shift |
| new_merchant | Low | Same categories, different patterns |

### Recommendations

1. **Deploy OCSVM** as primary detector with threshold 0.47 for production
2. **Monitor false positive rate** — 131 FP on 5,270 normal transactions (2.5%)
3. **Add more features:** merchant embeddings, temporal sequence patterns, cross-persona baselines
4. **Consider semi-supervised approach** with labeled anomaly examples
5. **Retrain quarterly** as spending patterns evolve

## Artifacts

- `models/anomaly/anomaly_detector.joblib` — trained OCSVM model
- `models/anomaly/evaluation.json` — full evaluation metrics and fold results
- `models/anomaly/evaluation_report.md` — markdown evaluation report
- `models/anomaly/confusion_matrices.png` — confusion matrix heatmaps
- `models/anomaly/pr_curves.png` — precision-recall curves
