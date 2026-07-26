# Odin ML — Documentation Index

## ML Development Phases

| Phase | Status | Key Deliverables | Documentation |
| :--- | :---: | :--- | :--- |
| **1. Problem Statement** | Complete | Model Design Document (MDD), 3 module designs, feasibility analysis | `1_problem-statement/MDD.md` |
| **2. Data Collection** | Complete | FIES 2023 NCR data, PSA/BSP datasets, FIES dictionary | `2_data-collection/`, `datasets/` |
| **3. Data Preprocessing** | Complete | Preprocessed raw data (train/val/test), temporal fold metadata | `3_data-preprocessing/preprocessing.md`, `scripts/preprocessor.py` |
| **4. Exploratory Data Analysis** | Complete | EDA report, 16 plots, class balance, correlations, data quality | `4_eda/eda_report.md`, `scripts/eda.py` |
| **4.5. Dimension & Threshold Discovery** | Complete | Dimension threshold candidates, clustering analysis, binary split validation | `4.5_dimension-threshold-discovery/`, `scripts/dimension_discovery.py` |
| **5. Feature Engineering** | Complete | 17 derived features, cyclical encoding, interaction features, feature selection, PCA | `5_feature-engineering/`, `scripts/feature_engineering.py` |
| **6. Model Design** | Complete | Feature sets, ROC cutoffs, walk-forward validation, partial window splits | `prerequisites/` (9 documents) |
| **6. Model Training** | Complete | PFP Classifier training (Tier 0-4), model artifacts | `6_model-training/`, `scripts/train_fbp.py` |
| **7. Model Evaluation** | Pending | Evaluation reports, confusion matrices, error analysis | — |
| **9. Deployment** | Pending | Containerized microservices, CI/CD pipeline | `prerequisites/deployment-architecture.md` |
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
│  ├─ MDD.md                             # Model Design Document (PFP Classifier)
│  ├─ MDD - Template.md                  # Blank MDD template
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
│  └─ eda_report.md                      # Generated EDA report (7 sections)
├─ 4.5_dimension-threshold-discovery/
│  └─ README.md                          # Phase 4.5: PFP dimension & threshold discovery
├─ 5_feature-engineering/
│  ├─ README.md                          # Phase 5 in-depth guide
│  └─ feature-engineering.md             # Feature engineering pipeline documentation
├─ 6_model-training/
│  ├─ README.md                          # Phase 6: PFP Classifier training (Tier 0-4)
│  ├─ forecaster-training.md             # Forecaster training (RF + PyTorch LSTM/GRU/BiLSTM)
│  └─ anomaly-training.md                # Anomaly Detector training (IF + OCSVM + AE + Ensemble)
├─ prerequisites/
│  ├─ feature-set.md                     # Complete feature definitions (3 models)
│  ├─ walk-forward-validation.md         # Temporal validation methodology
│  ├─ partial-window-splits.md           # Train/val/test splitting strategy
│  ├─ roc-cutoff-selection.md            # Threshold calibration for PFP
│  ├─ module-integration.md              # Inter-module API contracts
│  └─ deployment-architecture.md         # Container/k8s/CI-CD design
└─ standards/
   ├─ REPOSITORY-STANDARDS.md            # Coding standards
   ├─ design-standards.md                # UI design system
   └─ frontend-design-antipattern-standards.md
```

## Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/collector.py` | CSV to Parquet conversion for raw datasets | Complete |
| `scripts/generate_personas.py` | 12-archetype (A-L) persona generator from FIES data | Complete |
| `scripts/generate_transactions.py` | 12-month transaction history generator | Complete |
| `scripts/preprocessor.py` | Synthesis + splitting + raw data export | Complete |
| `scripts/feature_engineering.py` | 17 derived features + encoding + selection + PCA | Complete |
| `scripts/feature_engineering_forecaster.py` | Forecaster-specific feature engineering (RFM, STL, lags) | Complete |
| `scripts/feature_engineering_anomaly.py` | Anomaly-specific per-transaction feature engineering (22 features) | Complete |
| `scripts/eda.py` | Exploratory data analysis with plots + report | Complete |
| `scripts/train_fbp.py` | PFP Classifier training (Tier 0-4), temporal fold evaluation | Complete |
| `scripts/train_forecaster.py` | Forecaster training (RF + PyTorch LSTM/GRU/BiLSTM) | Complete |
| `scripts/train_anomaly.py` | Anomaly Detector training (IF + OCSVM + AE + Ensemble) | Complete |
| `scripts/dimension_discovery.py` | Phase 4.5: clustering analysis + overlay feature computation | Complete |
| `scripts/fies_columns.py` | FIES variable ID ↔ CSV column mapping | Complete |
| `scripts/synthesizer.py` | Deprecated — use `preprocessor.py` instead | Deprecated |

## Pipeline Execution Order

```
1. python scripts/collector.py --input datasets/raw/ --output datasets/unprocessed/
   → Converts CSV files to Parquet format

2. python scripts/preprocessor.py --input datasets/unprocessed/puf.parquet --output datasets/processed/
   → Runs synthesis (personas + transactions → synth/)
   → Generates: train.parquet, val.parquet, test.parquet (raw data), temporal_folds.json

3. python scripts/feature_engineering.py --input datasets/processed/ --output datasets/engineered/
   → Computes 17 derived features + cyclical encoding + interaction features
   → Optional: feature selection, PCA dimensionality reduction
   → Generates: train.parquet, val.parquet, test.parquet (engineered), feature_columns.json

4. python scripts/eda.py --input datasets/engineered/ --output figures/
   → Generates: eda_report.md, *.png plots
   → Note: EDA now reads from datasets/engineered/ for full feature analysis

4.5. python scripts/dimension_discovery.py --input datasets/processed/ --output docs/4.5_dimension-threshold-discovery/
   → Unsupervised clustering on Stability/Necessity scores
   → Computes Financial Trajectory and Financial Margin features
   → Generates: dimension-threshold-candidates.md

5. python scripts/train_fbp.py --input datasets/engineered/ --output models/fbp/
   → Trains PFP Classifier model (Tier 0-4)
   → Generates: models/fbp/ (trained models + evaluation.json + evaluation_report.md)

6. python scripts/feature_engineering_forecaster.py
   → Generates forecaster feature sets (datasets/forecaster/)

7. python scripts/train_forecaster.py
   → Trains Forecaster model (RF + PyTorch LSTM/GRU/BiLSTM)
   → Generates: models/forecaster/ (trained models + evaluation)

8. [Next] python scripts/train_anomaly.py --input datasets/engineered/ --output models/anomaly
   → Trains Anomaly Detector model
```
