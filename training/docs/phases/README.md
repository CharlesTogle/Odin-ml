# ML Development Phases (Odin-ML)

Complete model development lifecycle reference for the revised BUDI ML pipeline. Design
documents and phase-by-phase narratives live in `Odin-Paper/docs/ml/`; this directory holds
the Odin-ML-side operational artifacts and runbooks.

Phase numbers match `Odin-Paper/docs/ml/README.md`.

| Phase | Name | Status | Odin-ML artifact |
|-------|------|--------|------------------|
| 1 | Problem Statement | Complete | (design docs in `Odin-Paper/docs/ml/1_problem-statement/`) |
| 2 | Data Collection | Complete | `training/scripts/collector.py`, `raw-datasets/` |
| 3 | Data Preprocessing | Complete | `training/scripts/preprocessor.py`, `training/datasets/processed/` |
| 4 | EDA | Complete | `training/scripts/eda.py`, `training/figures/eda_report.md` |
| 4.5 | Dimension & Threshold Discovery | Complete | `training/scripts/dimension_discovery.py`, `training/docs/dimension-threshold-discovery/` |
| 5 | Feature Engineering | Complete | `training/scripts/feature_engineering*.py`, `training/datasets/{engineered,forecaster,anomaly}/` |
| 6 | Model Training | Complete | `training/scripts/train_*.py` (tier engines) |
| 7 | Model Evaluation | **Partial** | `models/*/metadata.json`, training-run `evaluation.json` (see `phases/07-model-evaluation.md`) |
| 8 | Model Selection & Versioning | **Pending (scaffolded)** | `models/` (top-level), `phases/08-model-selection-versioning.md` |
| 9 | Deployment | **Partial** | `app/` FastAPI serving; `phases/09-deployment.md` |
| 10 | Model Monitoring | **Pending** | `phases/10-model-monitoring.md` |

## Phase documents

- [`07-model-evaluation.md`](07-model-evaluation.md)
- [`08-model-selection-versioning.md`](08-model-selection-versioning.md)
- [`09-deployment.md`](09-deployment.md)
- [`10-model-monitoring.md`](10-model-monitoring.md)

## Notes

- The previous training run produced evaluation artifacts committed under
  `training/figures/models/`. Those belong to the **old scope**; the phase docs below define
  the structure for the **new scope** work (see `../models/README.md`).
- Pre-registered decision rules are the source of truth for acceptance and are restated
  per-phase here so a reviewer never has to open the training scripts to know the rule.