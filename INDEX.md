# Odin-ML — Repository Index

- **Project:** Development of BUDI: A Personalized Intelligent Finance Management Application for Filipinos Using Classification, Forecasting, Optimization, and Anomaly Detection Models for Improving Savings and Debt
- **Institution:** University of Makati | Group 4, III-DCSAD
- **Last indexed:** 2026-08-31

---

## How to Use This Index

| Need | Go to |
| :--- | :--- |
| Project overview, setup, and usage | `README.md` |
| FastAPI service code (endpoints, models, schemas) | `app/` |
| Run the test suite | `tests/` (run `pytest`) |
| Model training pipeline and phase docs | `training/` |
| ML design documents (data collection, EDA, thresholds) | `training/docs/` |
| Engineering and documentation standards | `docs/standards/` |
| Enforceable Python/ML code standards | `docs/standards/REPOSITORY-STANDARDS.md` |
| Git commit message conventions | `docs/standards/git-commit-standards.md` |
| Thesis documentation (system spec, PRD, chapters) | **Odin-Paper** |
| RRL corpus and scoring | **Odin-Literature** |

---

## Repository Map

| Path | Purpose |
| :--- | :--- |
| `AGENTS.md` | Agent navigation guide, standards, and repository conventions. |
| `INDEX.md` | This file. Master navigation index. |
| `README.md` | Project overview, pipeline, setup, and usage. |
| `app/` | FastAPI microservice for model serving (PFP, forecaster, anomaly, budget). |
| `tests/` | Pytest coverage for the service. |
| `training/` | Model development pipeline (scripts, datasets, models, docs). |
| `docs/` | Standards and documentation. |

---

## app/ — FastAPI Microservice

| Path | Purpose |
| :--- | :--- |
| `app/main.py` | FastAPI entrypoint. |
| `app/api/` | Route modules (health, pfp, forecast, anomaly, budget). |
| `app/services/` | Inference and business logic. |
| `app/models/` | Model loading and artifact registry. |
| `app/schemas/` | Pydantic request and response models. |
| `app/core/` | Settings and startup wiring. |

---

## tests/

Pytest coverage for the serving API (health, PFP, forecast, anomaly, budget). See `conftest.py` for shared fixtures. Run with `pytest`.

---

## training/

The model development pipeline. Large generated artifacts are gitignored; scripts, docs, and evaluation reports are committed.

| Path | Purpose |
| :--- | :--- |
| `training/scripts/` | Collector, preprocessor, feature engineering, and training scripts. |
| `training/docs/` | ML design documents (data collection, EDA, dimension discovery). |
| `training/figures/` | EDA plots and `eda_report.md`. |
| `training/datasets/` | Processed/engineered feature matrices (gitignored). |
| `training/synth/` | Generated personas and transactions (gitignored). |
| `training/models/` | Trained model artifacts: `pfp/`, `forecaster/`, `anomaly/` (gitignored). |

---

## docs/

| Path | Purpose |
| :--- | :--- |
| `docs/README.md` | Clarifies the split between `docs/standards/` and `training/docs/`. |
| `docs/standards/REPOSITORY-STANDARDS.md` | Enforceable Python/ML engineering standards. |
| `docs/standards/git-commit-standards.md` | Git commit message format and scopes. |
| `docs/standards/documentation-format.md` | Shared documentation formatting rules. |
| `docs/standards/archive/` | Deprecated frontend/design standards (for the `odin` app). |

---

## Cross-References

| Task | Use |
| :--- | :--- |
| Model design documents (MDD, feature sets) | `../Odin-Paper/docs/ml/` |
| System specification and PRD | `../Odin-Paper/docs/requirements-engineering/` |
| RRL corpus, scoring, and pipeline | **Odin-Literature** |
