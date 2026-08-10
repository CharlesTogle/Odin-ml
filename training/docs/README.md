# Odin ML — Documentation Index

## ML Development Phases

| Phase | Status | Key Deliverables | Documentation |
| :--- | :---: | :--- | :--- |
| **1. Problem Statement** | Complete | Model Design Document (MDD), 4 module designs, feasibility analysis | `1_problem-statement/README.md` |
| **2. Data Collection** | Complete | FIES 2023 NCR data, PSA/BSP datasets, FIES dictionary | `2_data-collection/`, `training/datasets/` |
| **3. Data Preprocessing** | Complete | Preprocessed raw data (train/val/test), temporal fold metadata | `3_data-preprocessing/preprocessing.md` |
| **4. Exploratory Data Analysis** | Complete | EDA report, plots, class balance, correlations, data quality | `4_eda/eda_report.md` |
| **4.5. Dimension & Threshold Discovery** | Complete | Dimension threshold candidates, clustering analysis, binary split validation | `4.5_dimension-threshold-discovery/` |
| **5. Feature Engineering** | Complete | 17 derived features, cyclical encoding, interaction features, feature selection, PCA | `5_feature-engineering/` |
| **6. Model Training** | Complete | PFP (Tier 0–4), Forecaster (RF + PyTorch LSTM/GRU/BiLSTM), Anomaly (IF + AE) | `6_model-training/` |
| **7. Model Evaluation** | Partial | PFP/Forecaster/Anomaly evaluation JSONs + reports | `6_model-training/`, `training/models/*/evaluation.json` |
| **9. Deployment** | Partial | Serving API + budget optimizer stubbed; containerization/CI-CD pending | `1_problem-statement/deployment-architecture.md` |
| **10. Model Monitoring** | Pending | Drift detection, retraining triggers | — |

## Data Sources

| Source | Year | Scope | Use |
|--------|------|-------|-----|
| **PSA 2023 FIES NCR** | 2023 | 41,380 households, 87 columns | Financial numerical baselines (income, expenditure, savings rates) |
| **BSP 2021 Consumer Finance Survey** | 2021 | 267 pages, nationwide | Behavioral/attitudinal patterns (savings behavior, debt patterns, financial inclusion) |

## Documentation Structure

```
docs/
├─ README.md                              # This file — documentation index
├─ 1_problem-statement/
│  ├─ README.md                          # Phase 1 in-depth guide
│  ├─ module-design-document.md          # MDD — PFP, Forecaster, Anomaly, Budget Optimizer
│  ├─ module-integration.md              # Inter-module API contracts (v1.1)
│  ├─ deployment-architecture.md         # Container/k8s/CI-CD design (v1.1)
│  ├─ feature-set.md                     # Complete feature definitions (3 models)
│  ├─ walk-forward-validation.md         # Temporal validation methodology
│  ├─ partial-window-splits.md           # Train/val/test splitting strategy
│  ├─ roc-cutoff-selection.md            # Threshold calibration for PFP
│  ├─ bsp-fies-crosswalk.md              # BSP CFS ↔ FIES NCR field mapping + archetype justification
│  ├─ synthetic-injection-rules.md       # Rules for FIES→Persona→Transaction generation
│  └─ persona-validation-list.md         # 12 archetypes (A-L) for SME review
├─ 2_data-collection/
│  ├─ README.md                          # Phase 2 in-depth guide
│  └─ FIES Dictionary & Valueset.csv     # FIES variable ID mapping
├─ 3_data-preprocessing/
│  ├─ README.md                          # Phase 3 in-depth guide
│  └─ preprocessing.md                   # Preprocessing pipeline documentation
├─ 4_eda/
│  ├─ README.md                          # Phase 4 in-depth guide
│  └─ eda_report.md                      # Generated EDA report
├─ 4.5_dimension-threshold-discovery/
│  └─ README.md                          # Phase 4.5: PFP dimension & threshold discovery
├─ 5_feature-engineering/
│  ├─ README.md                          # Phase 5 in-depth guide
│  └─ feature-engineering.md             # Feature engineering pipeline documentation
└─ 6_model-training/
   ├─ README.md                          # Phase 6: PFP Classifier training (Tier 0-4)
   ├─ forecaster-training.md             # Forecaster training (RF + PyTorch LSTM/GRU/BiLSTM)
   └─ anomaly-training.md                # Anomaly Detector training (IF + AE)
```

## Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `training/scripts/collector.py` | CSV to Parquet conversion for raw datasets | Complete |
| `training/scripts/generate_personas.py` | 12-archetype (A-L) persona generator from FIES data | Complete |
| `training/scripts/generate_transactions.py` | 12-month transaction history generator | Complete |
| `training/scripts/preprocessor.py` | Synthesis + splitting + raw data export | Complete |
| `training/scripts/feature_engineering.py` | 17 derived features + encoding + selection + PCA | Complete |
| `training/scripts/feature_engineering_forecaster.py` | Forecaster-specific feature engineering (RFM, STL, lags) | Complete |
| `training/scripts/feature_engineering_anomaly.py` | Anomaly-specific per-transaction feature engineering | Complete |
| `training/scripts/eda.py` | Exploratory data analysis with plots + report | Complete |
| `training/scripts/train_pfp.py` | PFP Classifier training (Tier 0-4), temporal fold evaluation | Complete |
| `training/scripts/train_forecaster.py` | Forecaster training (RF + PyTorch LSTM/GRU/BiLSTM) | Complete |
| `training/scripts/train_anomaly.py` | Anomaly Detector training (IF + AE) | Complete |
| `training/scripts/dimension_discovery.py` | Phase 4.5: clustering analysis + overlay feature computation | Complete |
| `training/scripts/fies_columns.py` | FIES variable ID ↔ CSV column mapping | Complete |
| `training/scripts/synthesizer.py` | Deprecated — use `preprocessor.py` instead | Deprecated |

## Pipeline Execution Order

```
1. python training/scripts/collector.py --input training/datasets/raw/ --output training/datasets/unprocessed/
   → Converts CSV files to Parquet format

2. python training/scripts/preprocessor.py --input training/datasets/unprocessed/puf.parquet --output training/datasets/processed/
   → Runs synthesis (personas + transactions → training/synth/)
   → Generates: train.parquet, val.parquet, test.parquet (raw data), temporal_folds.json

3. python training/scripts/feature_engineering.py --input training/datasets/processed/ --output training/datasets/engineered/
   → Computes 17 derived features + cyclical encoding + interaction features
   → Optional: feature selection, PCA dimensionality reduction
   → Generates: train.parquet, val.parquet, test.parquet (engineered), feature_columns.json

4. python training/scripts/eda.py --input training/datasets/engineered/ --output figures/
   → Generates: eda_report.md, *.png plots

4.5. python training/scripts/dimension_discovery.py --input training/datasets/processed/ --output training/docs/4.5_dimension-threshold-discovery/
   → Unsupervised clustering on Stability/Necessity scores
   → Computes Financial Trajectory and Financial Margin features

5. python training/scripts/train_pfp.py --input training/datasets/engineered/ --output training/models/pfp/
   → Trains PFP Classifier model (Tier 0-4)
   → Generates: training/models/pfp/ (artifacts + evaluation.json + evaluation_report.md)

6. python training/scripts/feature_engineering_forecaster.py
   → Generates forecaster feature sets (training/datasets/forecaster/)

7. python training/scripts/train_forecaster.py
   → Trains Forecaster model (RF + PyTorch LSTM/GRU/BiLSTM)
   → Generates: training/models/forecaster/ (artifacts + evaluation)

8. python training/scripts/feature_engineering_anomaly.py
   → Generates per-transaction anomaly features

9. python training/scripts/train_anomaly.py
   → Trains Anomaly Detector model (IF + AE)
   → Generates: training/models/anomaly/ (artifacts + evaluation)

10. Serving: uvicorn app.main:app --reload --port 8000
    → Loads PFP, Forecaster, Anomaly artifacts; Budget Optimizer runs scipy LP
```

## Known Data & Version Gaps

- Persona source count 300 vs. 12,000 generated personas (see `training/TODO.md`).
- PFP per-class zero support for some labels (labeling artifact of synth data).
- Training artifacts built with scikit-learn 1.9.0; serving venv has 1.8.0 (warnings, non-fatal).
- System spec targets Python 3.14; runtime pinned to 3.13.14.
