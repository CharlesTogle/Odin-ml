# Odin ML

Python microservice for machine learning APIs and inference.

This repository currently contains:

- `app/main.py`: FastAPI app entrypoint
- `requirements.txt`: runtime Python dependencies
- `requirements-dev.txt`: development and test dependencies

## Tech Stack

- Python `3.14.4`
- FastAPI `0.135.3`
- Uvicorn
- TensorFlow `2.21.0`
- scikit-learn `1.8.0`
- Pytest
- HTTPX

## Prerequisites

Install these before working in this repository:

- Python `3.14.4`
- `pip`
- `venv`

## Repository Layout

```text
odin-ml/
├─ .gitignore
├─ app/
│  └─ main.py
├─ README.md
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
