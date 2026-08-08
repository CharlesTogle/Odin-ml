# Odin ML

Python microservice for machine learning APIs and inference, plus the complete model development pipeline for the Odin thesis.

## What's Here

| Directory | Purpose |
|-----------|---------|
| `app/` | FastAPI microservice for model serving |
| `scripts/` | Data collection, preprocessing, and analysis pipeline |
| `docs/` | ML design documents, feature specs, preprocessing docs |
| `datasets/raw/` | FIES 2023, PSA, BSP source data (CSV input) |
| `datasets/processed/` | Preprocessed raw feature matrices (Parquet) |
| `datasets/engineered/` | Engineered feature matrices (Parquet) |
| `synth/` | Generated personas and transactions (Parquet) |
| `figures/` | EDA plots and analysis outputs |

## Model Development Pipeline

```
FIES 2023 Data → collector.py → preprocessor.py → feature_engineering.py → eda.py → [train scripts]
     ↓                ↓               ↓                    ↓                  ↓
  datasets/raw/    datasets/    synth/ + datasets/    datasets/          figures/
                  unprocessed/    processed/          engineered/
```

### Step 1: Collect Data

```bash
python scripts/collector.py \
  --input datasets/raw/ \
  --output datasets/unprocessed/
```

Converts raw CSV datasets to Parquet format for pipeline consumption.

### Step 2: Preprocess Data

```bash
python scripts/preprocessor.py \
  --input datasets/unprocessed/puf.parquet \
  --output datasets/processed/
```

Runs synthesis (persona + transaction generation) internally, then splits personas into train/val/test and exports raw feature matrices (metadata + 11 raw columns). No feature engineering or normalization — that happens in the next step.

### Step 3: Engineer Features

```bash
python scripts/feature_engineering.py \
  --input datasets/processed/ \
  --output datasets/engineered/
```

Computes 17 derived financial features, cyclical encoding, interaction features, redundant feature removal, optional feature selection and PCA. Reads raw data from `datasets/processed/`, outputs engineered matrices to `datasets/engineered/`.

### Step 4: Run Exploratory Data Analysis

```bash
python scripts/eda.py \
  --input datasets/processed/ \
  --output figures/ \
  --seed 42
```

Produces a comprehensive EDA report with static plots covering distributions, correlations, class balance, temporal patterns, anomalies, and data quality. Works with both raw data (`datasets/processed/`) and engineered data (`datasets/engineered/`).

```bash
python scripts/train_pfp.py --input datasets/processed/ --output models/pfp
python scripts/train_forecaster.py --input datasets/processed/ --output models/forecaster
python scripts/train_anomaly.py --input datasets/processed/ --output models/anomaly
```

## Tech Stack

- Python `3.14.4`
- FastAPI `0.135.3`
- Uvicorn
- TensorFlow `2.21.0`
- scikit-learn `1.8.0`
- pandas, numpy
- pyarrow (Parquet support)
- matplotlib, seaborn (visualization)
- Pytest, HTTPX

## Prerequisites

Install these before working in this repository:

- Python `3.14.4`
- `pip`
- `venv`

## Repository Layout

```text
odin-ml/
├─ app/
│  └─ main.py                    # FastAPI entrypoint
├─ scripts/
│  ├─ collector.py               # CSV to Parquet conversion
│  ├─ generate_personas.py       # 14-archetype persona generator
│  ├─ generate_transactions.py   # 12-month transaction generator
│  ├─ preprocessor.py            # Synthesis + splitting + raw data export
│  ├─ feature_engineering.py     # 17 derived features + encoding + selection + PCA
│  ├─ synthesizer.py             # Deprecated — use preprocessor.py
│  ├─ eda.py                     # Exploratory data analysis (works with raw or engineered)
│  └─ fies_columns.py            # FIES variable ID mapping
├─ docs/
│  ├─ README.md                  # Documentation index
│  ├─ 1_problem-statement/       # MDD and module designs
│  ├─ 2_data-collection/         # FIES dictionary
│  ├─ 3_data-preprocessing/      # Preprocessing pipeline docs
│  ├─ 4_eda/                     # EDA report and analysis
│  ├─ prerequisites/             # Feature sets, validation, deployment
│  └─ standards/                 # Coding standards
├─ datasets/
│  ├─ raw/                       # FIES, PSA, BSP source data (CSV)
│  ├─ unprocessed/               # Parquet format of raw data (collector output)
│  ├─ processed/                 # Preprocessed raw feature matrices (Parquet)
│  └─ engineered/                # Engineered feature matrices (Parquet)
├─ synth/                        # Generated personas + transactions (Parquet)
├─ figures/                      # EDA plots and analysis outputs
├─ requirements.txt
└─ requirements-dev.txt
```

## First-Time Setup

### Windows

Use PowerShell:

```powershell
cd C:\path\to\App\odin-ml
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Start the FastAPI dev server:

```powershell
uvicorn app.main:app --reload --port 8000
```

### Bash

Use this on Linux, macOS, WSL, or Git Bash:

```bash
cd /path/to/App/odin-ml
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Start the FastAPI dev server:

```bash
uvicorn app.main:app --reload --port 8000
```

### Fish

Use the Fish activation script, not the Bash one:

```fish
cd /path/to/App/odin-ml
python3.14 -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Start the FastAPI dev server:

```fish
uvicorn app.main:app --reload --port 8000
```

## Environment Variables

The current scaffold does not require secrets yet, but this service will likely need environment variables once model storage, Supabase, or external services are connected.

Expected future values:

```env
PORT=8000
MODEL_PATH=
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

Do not commit real secrets.

## Common Commands

Create the virtual environment:

```bash
python3.14 -m venv .venv
```

Activate the environment in Bash:

```bash
source .venv/bin/activate
```

Activate the environment in Fish:

```fish
source .venv/bin/activate.fish
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

Run tests:

```bash
pytest
```

Verify the FastAPI module compiles:

```bash
python -m py_compile app/main.py
```

## Current Endpoints

The service currently exposes:

- `GET /`
- `GET /health`

Default local URL:

```text
http://localhost:8000
```

## Troubleshooting

### `source .venv/bin/activate` fails in Fish

That is expected. Fish cannot source the Bash activation script.

Use:

```fish
source .venv/bin/activate.fish
```

### PowerShell blocks script activation

If PowerShell refuses to run `Activate.ps1`, open PowerShell as your user and run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the venv again.

### `python3.14` is not found

Confirm Python `3.14.4` is installed and on your `PATH`.

On Windows, use:

```powershell
py -3.14 --version
```

On Bash or Fish:

```bash
python3.14 --version
```

### TensorFlow install issues

TensorFlow support can vary by platform and Python build. If installation fails, verify that your Python `3.14.4` environment is compatible with the TensorFlow wheel available for your OS and CPU architecture.

## Recommended Next Steps

- Add `.env.example`
- Add API routers under `app/`
- Add request and response schemas
- Add model-loading lifecycle hooks
- Add pytest coverage for `/` and `/health`
- Add Dockerfile for Cloud Run deployment
