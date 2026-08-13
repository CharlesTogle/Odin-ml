# Phase 1: Problem Statement

## What This Phase Is

This phase defines the ML problem space for Odin's four intelligent modules. It produces the Model Design Document (MDD) — the complete specification for what each model does, how it's trained, and how it integrates with the app.

## Inputs

| Input | Source | Purpose |
|-------|--------|---------|
| PSA 2023 FIES data | PSA Data Archive microdata | Income/expense distributions for persona generation; scoped to NCR |
| RRL benchmarks | Academic papers | Baseline model performance (XGBoost 90.66%, CatBoost 90.93%, LSTM, etc.) |
| Thesis requirements | `Odin-Paper` repository | Alignment with research objectives |
| SME domain knowledge | Asst. Prof. Pamela A. Go (CBFS) | Business logic validation |

## Process

### 1. Problem Definition

Four models are designed to work together:

| Model | Type | Classes/Targets | Purpose |
|-------|------|-----------------|---------|
| **PFP Classifier** | Multi-class classification | 8 (Stable-Flexible-Tolerant, Stable-Flexible-At-Risk, Stable-Obligated-Tolerant, Stable-Obligated-At-Risk, Variable-Flexible-Tolerant, Variable-Flexible-At-Risk, Variable-Obligated-Tolerant, Variable-Obligated-At-Risk) | Assign personal financial profile |
| **Forecaster** | Time-series regression | Continuous (monthly expenses) | Predict future income and expense trends |
| **Anomaly Detector** | Anomaly detection | Binary (normal/anomalous) | Detect unusual spending patterns |
| **Budget Optimizer** | Constraint optimization | Continuous (category allocations) | Recommend budget allocations within constraints |

### 2. Semi-Supervised Approach

The three learned models (PFP, Forecaster, Anomaly Detector) follow a semi-supervised design; the Budget Optimizer is a constraint-optimization module (LP) that consumes their outputs rather than learning from data:

- **Training**: Synthetic personas generated from FIES NCR aggregates (12 archetypes × 1,000 personas = 12,000 total)
- **Validation**: SME review of archetype list and classification thresholds
- **Deployment**: Real user data from the app

### 3. Design Decisions

Key architectural choices documented in MDD:

- **8-class PFP**: Derived from three binary dimensions (income stability × obligation weight × financial tolerance)
- **1-month embargo gap**: Prevents temporal leakage between train/test splits
- **Expanding window WFV**: Rolling-origin evaluation across months 8-12
- **2 classification modes**: STANDARD (transactions), QUESTIONNAIRE (self-reported). An earlier `ENSEMBLE` mode (both inputs) was dropped in v1.4 — no such path exists in the system spec or the training pipeline.

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| `module-design-document.md` | `1_problem-statement/` | Complete model design document (1,629 lines) |
| `bsp-fies-crosswalk.md` | `1_problem-statement/` | BSP CFS ↔ FIES NCR field mapping + archetype justification |
| `feature-set.md` | `1_problem-statement/` | Complete feature definitions for all 3 models |
| `walk-forward-validation.md` | `1_problem-statement/` | Temporal validation methodology |
| `partial-window-splits.md` | `1_problem-statement/` | Train/val/test splitting strategy |
| `roc-cutoff-selection.md` | `1_problem-statement/` | Threshold calibration for PFP |
| `module-integration.md` | `1_problem-statement/` | Inter-module API contracts |
| `deployment-architecture.md` | `1_problem-statement/` | Container/k8s/CI-CD design |
| `synthetic-injection-rules.md` | `1_problem-statement/` | 20 rules for FIES→Persona→Transaction |
| `persona-validation-list.md` | `1_problem-statement/` | 12 archetypes for SME review |
| `persona-validation-list-SME-draft.md` | `1_problem-statement/` | SME review draft of the archetype roster |
| Open tasks & gaps | `training/TODO.md` | MDD follow-up tasks and known data/version gaps |

## Key Design Rationale

### Why Semi-Supervised?

FIES microdata is anonymized per RA 10173 (Data Privacy Act) and PSA disclosure policy. It contains only aggregate/geographic fields — it was never designed to capture person-level behavioral signals. Persona-based synthesis is the only feasible route to a labeled dataset with both a defensible ground-truth class and the indirect features the learned candidates are meant to exploit.

### Why 8 Classes?

The PFP is derived from exactly three **classifying** dimensions:
- **Income stability** *(classifying)*: Stable vs. Variable (based on income coefficient of variation)
- **Obligation weight** *(classifying)*: Flexible vs. Obligated (based on (Essential + Obligatory) / Total Expenses)
- **Financial tolerance** *(classifying)*: Tolerant vs. At-Risk (based on emergency runway — months of expenses covered by savings)

This creates 2×2×2 = **8 balanced classes** that capture meaningful personal financial profile differences across stability, obligation burden, and financial resilience.

Two additional features — **Financial Behavior** (user preference captured via questionnaire) and **Surplus/Deficit Rate** (internal tracking from transaction data) — are **non-classifying** features that do not affect class count. They modify system urgency and tone without changing the PFP class itself. See the persona-validation-list-SME-draft.md for their full definitions.

> **Note:** Classification thresholds (CV for income stability, obligation ratio for obligation weight, runway months for financial tolerance) are being derived by the research team via reasoning and the Phase 4.5 discovery process (unsupervised clustering + candidate threshold documentation), not sourced from any existing local paper (none exists). Final thresholds are pending SME validation.

### RRL Benchmark Summary

| Category | Best Model | Accuracy/AUC |
|----------|------------|--------------|
| Classification | CatBoost | 90.93% |
| Forecasting | LSTM | (pending evaluation) |
| Anomaly Detection | Autoencoder | AUC-ROC 0.971 |

## Next Steps

1. Build model training scripts (`train_pfp.py`, `train_forecaster.py`, `train_anomaly.py`)
2. Set up Google Colab notebook for LSTM training (free GPU access)
3. Complete SME validation of persona archetypes
4. Validate synthetic→real generalization with prototype users
