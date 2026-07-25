# MDD Gaps — Implementation To-Do List

**Created:** 2026-07-17
**Updated:** 2026-07-18
**Status:** In Progress
**Deadline:** 2026-07-27 (10 days to start training)

---

## Context

This document tracks the implementation gaps identified during the MDD robustness analysis. All items are required before model training can begin. The MDD (v1.0 for PFP Classifier, v2.1 for Forecaster, v2.0 for Anomaly Detector) is 90% complete. These documents fill the remaining 10%.

---

## Completed

### 1. Synthetic Data Injection Rules
- **File:** `synthetic-injection-rules.md`
- **Status:** Complete
- **Completed:** 2026-07-17
- **Purpose:** Specify exact rules for generating behavioral features from FIES NCR aggregates
- **Contents:**
  - CV calculation methodology
  - Obligation ratio thresholds per PFP
  - Spending timing injection relative to payday (15th/30th)
  - Category diversity rules per income decile
  - Inter-transaction gap generation
  - Temporal pattern injection (payday spikes, petsa de peligro)

### 2. ROC-Based Cutoff Selection
- **File:** `roc-cutoff-selection.md`
- **Status:** Complete
- **Completed:** 2026-07-17
- **Purpose:** Define how income-stability and obligation-weight thresholds are calibrated

### 3. Partial Window Split Methodology
- **File:** `partial-window-splits.md`
- **Status:** Complete
- **Completed:** 2026-07-17
- **Purpose:** Define how models are evaluated under partial transaction history

### 4. Walk-Forward Validation Protocol
- **File:** `walk-forward-validation.md`
- **Status:** Complete
- **Completed:** 2026-07-17
- **Purpose:** Define window/step sizes for time-series cross-validation

### 5. Final Feature Set
- **File:** `feature-set.md`
- **Status:** Complete
- **Completed:** 2026-07-17
- **Purpose:** Commit to final features derived from RRL benchmarks

### 6. Module Integration Spec
- **File:** `module-integration.md`
- **Status:** Complete
- **Completed:** 2026-07-17
- **Purpose:** Define how modules interact

### 7. Deployment Architecture
- **File:** `deployment-architecture.md`
- **Status:** Complete
- **Completed:** 2026-07-17
- **Purpose:** Define container strategy and deployment topology

### 8. Synthetic Data Generation Pipeline
- **Files:** `scripts/run_pipeline.py`, `scripts/generate_personas.py`, `scripts/generate_transactions.py`
- **Status:** Complete
- **Completed:** 2026-07-18
- **Purpose:** Build the FIES NCR → Persona → Transaction pipeline
- **Output:** 12,000 personas (12 archetypes × 1,000 each), 12-month transaction histories, anomaly injection

### 9. Persona List for SME Validation
- **File:** `persona-validation-list.md`
- **Status:** Complete
- **Completed:** 2026-07-17
- **Purpose:** Exhaustive list of personas for financial science professor to validate
- **Note:** Pending SME review — archetypes may be added/edited/removed

### 10. Data Preprocessing Pipeline
- **File:** `scripts/preprocessor.py`, `docs/3_data-preprocessing/preprocessing.md`
- **Status:** Complete
- **Completed:** 2026-07-18
- **Purpose:** Synthesis, splitting, raw data export with leakage prevention
- **Output:** train.parquet, val.parquet, test.parquet (raw), temporal_folds.json,
  split_metadata.json

### 11. Feature Engineering Pipeline
- **File:** `scripts/feature_engineering.py`, `docs/5_feature-engineering/feature-engineering.md`
- **Status:** Complete
- **Completed:** 2026-07-20
- **Purpose:** 17 derived features, cyclical encoding, interaction features, feature
  selection, PCA dimensionality reduction
- **Output:** train.parquet, val.parquet, test.parquet (engineered),
  feature_columns.json, pipeline_report.json

### 12. PFP Dimension & Threshold Discovery (Phase 4.5)
- **File:** `docs/4.5_dimension-threshold-discovery/README.md`
- **Status:** Complete
- **Purpose:** Define classification thresholds for Financial Stability, Financial
  Obligation, and Financial Tolerance; validate binary splits via clustering;
  compute overlay indicators (Financial Trajectory, Financial Margin)
- **Completed:**
  - 8-class PFP label format implemented (Stable/Volatile × Obligated/Flexible × Tolerant/Tight)
  - 12 SME archetypes (A-L) aligned across generate_personas.py and persona-validation-list.md
  - Clustering analysis confirms binary splits are sufficient (silhouette k=2 competitive with k=8)
  - `dimension_discovery.py` created with K-means/GMM clustering + threshold candidates
- **Remaining:**
  - Final threshold values pending SME review + real user data (currently: CV < 0.5, ratio > 0.6, runway ≥ 3 months)
  - Financial Trajectory and Financial Margin features are overlay indicators, not classifying dimensions

### 12b. synthetic-injection-rules.md §3.3 Archetype Table
- **File:** `docs/prerequisites/synthetic-injection-rules.md`
- **Status:** Complete
- **Purpose:** Update §3.3 from 14 demographic archetypes to 12 A-L archetypes; update PFP labels from 4-class to 8-class
- **Completed:**
  - §3.3 rewritten with 12 A-L archetypes and 8-class PFP labels
  - §3.1 PFP-to-FIES mapping updated for 8 classes
  - §3.2 employment mapping updated for 8-class system
  - Rule 19 label generation rewritten for 3 dimensions
  - Rule 20 threshold constants updated to match generate_personas.py
  - Pipeline diagram updated to "12 archetypes"
  - Output spec updated to 12,000 personas and 8 PFP labels

### 12c. obligatory_expenses hardcode fix
- **File:** `generate_transactions.py`, `feature_engineering.py`
- **Status:** Complete
- **Purpose:** Replace `obligatory_expenses = 0.0` with actual debt_payment computation per MDD §4
- **Completed:**
  - `generate_transactions.py`: `obligatory_expenses = debt_payment` (was 0.0)
  - `feature_engineering.py`: `obligatory = float(arr_debt.sum())` (was 0.0), added `debt_list` accumulator
  - Obligation ratio now correctly includes debt payments (mean 0.88 vs old 0.78)
  - Note: Label distribution is skewed toward Obligated because essential+debt naturally exceeds 0.6 for most personas — this is correct behavior, not a bug

---

## Pending — Before Thesis Submission

### 11. Monitoring Consolidation
- **File:** `monitoring-consolidation.md`
- **Status:** Pending
- **Owner:** Guevarra
- **Deadline:** Before submission
- **Purpose:** Define shared monitoring service
- **Contents:**
  - Unified drift detection (ADWIN/EDDM)
  - Shared retraining triggers
  - Model version coordination
  - Performance dashboard

### 12. Explainability Plan
- **File:** `explainability-plan.md`
- **Status:** Pending
- **Owner:** Guevarra
- **Deadline:** Before submission
- **Purpose:** Define explainability approach per model
- **Contents:**
  - PFP Classifier: SHAP feature importance
  - Forecaster: Attention visualization
  - Anomaly Detector: Feature deviation explanations

---

## Pending — Model Training (Next)

### 13. PFP Classifier Training Script
- **File:** `scripts/train_fbp.py`
- **Status:** Complete
- **Purpose:** Train the Financial Behavioral Profile Classifier (8-class PFP)
- **Depends on:** `datasets/engineered/` (complete), `docs/prerequisites/roc-cutoff-selection.md`
- **Result:** Random Forest wins at 0.6241 Macro-F1, beating Tier 1 rule-based by 0.62 points (well above 2-point margin)

### 14. Forecaster Training Script
- **File:** `scripts/train_forecaster.py`
- **Status:** Complete
- **Purpose:** Train the LSTM Spending Forecaster
- **Depends on:** `datasets/forecaster/` (complete), `docs/prerequisites/walk-forward-validation.md`
- **Result:** Random Forest wins at MAPE 9.63%, R² 0.831, beating naive baseline by ~77% MAPE reduction. LSTM/GRU/BiLSTM skipped (TF unavailable on this machine).

### 15. Anomaly Detector Training Script
- **File:** `scripts/train_anomaly.py`
- **Status:** Complete
- **Purpose:** Train the Isolation Forest Anomaly Detector
- **Depends on:** `datasets/anomaly/` (complete)
- **Result:** One-Class SVM wins at PR-AUC 0.1478 (416.8% improvement over baseline), ~3% anomaly rate (corrected from 0.3%). Final test PR-AUC: 0.1215.

---

## Timeline

| Day | Task | Status |
|-----|------|--------|
| 1-2 | Synthetic Data Injection Rules | Complete |
| 2-3 | ROC-Based Cutoff Selection | Complete |
| 3-4 | Partial Window Split Methodology | Complete |
| 4-5 | Walk-Forward Validation Protocol | Complete |
| 5-6 | Final Feature Set from RRL | Complete |
| 6-7 | Module Integration Spec | Complete |
| 7-8 | Deployment Architecture | Complete |
| 8-9 | Persona List for SME | Complete |
| 9-10 | Synthetic Data Pipeline | Complete |
| 10 | Data Preprocessing Pipeline | Complete |
| 11 | Feature Engineering Pipeline | Complete |
| 11.5 | PFP Dimension & Threshold Discovery (Phase 4.5) | Complete |
| 12 | PFP Classifier Training | Complete |
| 13 | Forecaster Pipeline (FE + Training) | Complete |
| 14 | Anomaly Detector Pipeline (FE + Training) | Complete |
| 15 | Model Documentation Reconciliation | Complete |

---

## Notes

- All design documents are in `docs/prerequisites/`
- Preprocessing pipeline documentation: `docs/3_data-preprocessing/preprocessing.md`
- Synthetic data pipeline code: `scripts/` (run_pipeline.py, generate_personas.py, generate_transactions.py)
- Preprocessing code: `scripts/preprocess.py`
- Google Colab recommended for LSTM training (free GPU)
- Cloud deployment target (AWS/GCP/Azure TBD)
- NCR subset already extracted from FIES
- SME validation pending — archetypes may be added/edited/removed after professor review

### Classification (PFP Classifier)

| Algorithm | Benchmark | Source |
|-----------|-----------|--------|
| XGBoost | 90.66% accuracy, F1=90.72 | Delena et al. |
| CatBoost | 90.93% accuracy | Salvador |
| LightGBM | 88.52% accuracy, 2.17s training | Salvador |
| Random Forest | 73% accuracy | Pagliaro |
| Logistic Regression | Baseline | Multiple |

### Forecasting (Forecaster)

| Algorithm | Benchmark | Source |
|-----------|-----------|--------|
| ARIMA | Baseline (struggles with non-linear) | D'Souza et al. |
| LSTM | 72% accuracy (5-day) | Pagliaro |
| Hybrid ARIMA-LSTM | Improved robustness | D'Souza et al. |
| XGBoost Regressor | MSE 360.0 | Sonkavde et al. |
| LightGBM | Best accuracy/efficiency | Paper 85 |
| Prophet | Lowest weighted error (10.456) | Mariano & Monreal |

### Anomaly Detection (Anomaly Detector)

| Algorithm | Benchmark | Source |
|-----------|-----------|--------|
| Isolation Forest | 95.3% detection, 4.8% FPR | Zhong |
| Autoencoder | AUC-ROC 0.971 | Fariha et al. |
| One-Class SVM | Cold-start utility | Bader & Haraty |
| IQR Rule-based | F1 < 0.50 (expected) | MDD §6 |
| GRA+Ensemble | ROC AUC +3-6pp | Adlermann |
