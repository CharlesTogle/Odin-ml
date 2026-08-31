# Odin-ML — Agent Guide

**Thesis**: Development of BUDI: A Personalized Intelligent Finance Management Application for Filipinos Using Classification, Forecasting, Optimization, and Anomaly Detection Models for Improving Savings and Debt
**Group 4, III-DCSAD, University of Makati**

---

## Repository Role

This repository contains the **machine learning service** for the BUDI application: the FastAPI microservice (`app/`), the model training pipeline (`training/`), and their tests. It does **not** contain thesis documents (those live in **Odin-Paper**) or the RRL corpus (that lives in **Odin-Literature**).

---

## Coding Standards

| Standard | Location |
| :--- | :--- |
| Repository standards (Python/ML) | `docs/standards/REPOSITORY-STANDARDS.md` |
| Shared documentation format | `docs/standards/documentation-format.md` |
| Git commit message standards | `docs/standards/git-commit-standards.md` |

**Note:** This repository is a Python/FastAPI service. The frontend/TypeScript standards that apply to the main `odin` application repository do **not** apply here. Keep Python/ML-specific rules in `docs/standards/REPOSITORY-STANDARDS.md`.

---

## Repository Access and Collaboration

- `Odin` and `Odin-QA` are **read-only** to this agent. Do not modify, commit, or delete files in them.
- `Odin-ML` is fully writable.

---

## Top-Level Directory Layout

```text
Odin-ML/
  AGENTS.md              # This file — agent navigation and standards
  INDEX.md               # Master navigation index (authoritative)
  README.md              # Project overview and setup
  app/                   # FastAPI microservice (api, services, models, schemas, core)
  tests/                 # Pytest coverage
  training/              # Model development pipeline (scripts, datasets, models, docs)
  docs/                  # Standards and documentation
    └─ standards/        # Enforceable standards (Python/ML, format, commit)
```

---

## Navigation

| Document | Purpose |
| :--- | :--- |
| `INDEX.md` | Master index. Authoritative navigation for all files. |
| `README.md` | Project overview, setup, pipeline, and usage. |
| `app/` | FastAPI service code. Entry point: `app/main.py`. |
| `training/` | Model training pipeline (scripts, datasets, artifacts). |
| `training/docs/` | ML design documents (data collection, EDA, dimension discovery). |
| `tests/` | Pytest coverage for the service. |
| `docs/standards/` | Enforceable engineering and documentation standards. |

---

## Python Environment

- Runtime pinned by `.python-version` (Python 3.14.4).
- Install dependencies in a virtual environment only:

```bash
python -m venv .venv
source .venv/bin/activate   # Bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # test/dev deps
```

- Use `.\.venv\Scripts\Activate.ps1` on Windows PowerShell.

---

## Testing

Run the test suite:

```bash
pytest
```

Tests live in `tests/` and cover the FastAPI service (health, PFP, forecast, anomaly, budget).

---

## Git Commit Message Standards

Use the format and conventions in `docs/standards/git-commit-standards.md`. This repository uses the `ml`, `api`, `data`, `training`, `docs`, `config`, `deps`, `tests`, and `standards` scopes.

---

## Important Gotchas

- Large generated artifacts (`training/datasets/`, `training/synth/`, `training/models/`, `figures/`) are gitignored. Only scripts, docs, and evaluation reports are committed.
- Generated EDA and evaluation reports under `training/figures/` and `training/models/*/` are tracked; containerization and CI/CD are still pending.
- `app/core/config.py` holds the service version; keep it in sync with release changes.
- The frontend standards in the main `odin` repository do not apply here. See `docs/standards/documentation-format.md` and `docs/standards/REPOSITORY-STANDARDS.md` for the Python/ML conventions that govern this repository.
