# Odin ML

Python microservice for machine learning APIs and inference, plus the complete model development pipeline for the Odin thesis.

## What's Here

| Directory | Purpose |
|-----------|---------|
| `app/` | FastAPI microservice for model serving (PFP, forecaster, anomaly, budget) |
| `tests/` | Pytest coverage for the serving API |
| `training/scripts/` | Data collection, preprocessing, feature engineering, and training pipeline |
| `training/docs/` | ML design documents, feature specs, preprocessing docs |
| `training/datasets/` | Processed + engineered feature matrices (Parquet, gitignored) |
| `training/synth/` | Generated personas and transactions (Parquet, gitignored) |
| `training/models/` | Trained model artifacts (joblib, gitignored) |
| `figures/` | EDA plots and analysis outputs |

## Model Development Pipeline

```
FIES 2023 Data → collector.py → preprocessor.py → feature_engineering.py → eda.py → [train scripts]
     ↓                ↓               ↓                    ↓                  ↓
 training/datasets/ training/     training/synth/ +    training/          figures/
      raw/       unprocessed/       datasets/          datasets/
                                   processed/          engineered/
```

All pipeline commands run from the repository root with the virtualenv activated:

```bash
python training/scripts/collector.py \
  --input training/datasets/raw/ \
  --output training/datasets/unprocessed/
```

```bash
python training/scripts/preprocessor.py \
  --input training/datasets/unprocessed/puf.parquet \
  --output training/datasets/processed/
```

```bash
python training/scripts/feature_engineering.py \
  --input training/datasets/processed/ \
  --output training/datasets/engineered/
```

```bash
python training/scripts/eda.py \
  --input training/datasets/processed/ \
  --output figures/ \
  --seed 42
```

```bash
python training/scripts/train_pfp.py --input training/datasets/processed/ --output training/models/pfp
python training/scripts/train_forecaster.py --input training/datasets/processed/ --output training/models/forecaster
python training/scripts/train_anomaly.py --input training/datasets/processed/ --output training/models/anomaly
```

The Budget Optimizer is a constraint-optimization module (LP via `scipy.linprog`); see the Budget Optimizer MDD v1.0 in `training/docs/1_problem-statement/module-design-document.md`.

## Tech Stack

- Python `3.14.4` (runtime pinned by `.python-version`)
- FastAPI `0.135.3`
- Uvicorn
- PyTorch `2.13.0` (primary deep learning framework — forecaster LSTM/GRU, anomaly autoencoder)
- scikit-learn `1.9.0`
- scipy, joblib, pandas, numpy, pyarrow
- matplotlib, seaborn (visualization)
- Pytest, HTTPX

## Prerequisites

- Python `3.14.4`
- `pip`
- `venv`

## Repository Layout

```text
odin-ml/
├─ app/
│  ├─ api/                        # FastAPI route modules (health, pfp, forecast, anomaly, budget)
│  ├─ services/                   # Inference and business logic (reuses training feature builders)
│  ├─ models/                     # Model loading and artifact registry
│  ├─ schemas/                    # Pydantic request and response models
│  ├─ core/                       # Settings, startup wiring
│  └─ main.py                     # FastAPI entrypoint
├─ tests/                         # Pytest coverage
├─ training/
│  ├─ scripts/                    # collector, preprocessor, feature engineering, train_* scripts
│  ├─ docs/                       # ML design documents and phase docs
│  ├─ datasets/                   # raw/, unprocessed/, processed/, engineered/ (Parquet, gitignored)
│  ├─ synth/                      # Generated personas + transactions (Parquet, gitignored)
│  └─ models/                     # Trained artifacts: pfp/, forecaster/, anomaly/ (gitignored)
├─ figures/                       # EDA plots
├─ requirements.txt
├─ requirements-dev.txt
├─ AGENTS.md
└─ README.md
```

## First-Time Setup

### Windows

```powershell
cd C:\path\to\App\odin-ml
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Bash

```bash
cd /path/to/App/odin-ml
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Fish

```fish
cd /path/to/App/odin-ml
python3.14 -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Start the FastAPI dev server:

```bash
uvicorn app.main:app --reload --port 8000
```

## Common Commands

```bash
python3.14 -m venv .venv
source .venv/bin/activate        # Fish: source .venv/bin/activate.fish
pip install -r requirements.txt
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
pytest
python -m py_compile app/main.py
```

## Current Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service banner |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (models loaded) |
| `GET` | `/metrics` | Module/winner metadata |
| `POST` | `/api/v1/pfp/classify` | PFP classification (STANDARD / QUESTIONNAIRE) |
| `POST` | `/api/v1/pfp/classify/batch` | Batch PFP classification |
| `POST` | `/api/v1/forecast/predict` | Next-month expense forecast |
| `POST` | `/api/v1/forecast/predict/batch` | Batch forecast |
| `POST` | `/api/v1/anomaly/detect` | Transaction anomaly detection |
| `POST` | `/api/v1/anomaly/detect/batch` | Batch anomaly detection |
| `POST` | `/api/v1/budget/recommend` | Budget allocation recommendation (LP) |
| `POST` | `/api/v1/budget/recommend/batch` | Batch budget recommendation |

Default local URL:

```text
http://localhost:8000
```

Interactive docs: `http://localhost:8000/docs`

## Troubleshooting

### `source .venv/bin/activate` fails in Fish

That is expected. Fish cannot source the Bash activation script; use `source .venv/bin/activate.fish`.

### PowerShell blocks script activation

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the venv again.

### `python3.14` is not found

Confirm Python `3.14.4` is installed and on your `PATH`. On Windows use `py -3.14 --version`; on Bash or Fish use `python3.14 --version`.

### PyTorch not installed

The anomaly autoencoder artifact requires PyTorch at serve time. Install the CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Recommended Next Steps

- Add per-module `Dockerfile` + `docker-compose.yml` (ports 8000–8005) matching `deployment-architecture.md` v1.1
- Add model artifact versioning (training-data hash + feature columns) to `training/models/*/metadata.json`
- Wire the Budget Optimizer to forecast + PFP outputs (end-to-end `/api/v1/analyze`)
- Persist prediction history for `/user/{id}/history` and `/latest` endpoints
