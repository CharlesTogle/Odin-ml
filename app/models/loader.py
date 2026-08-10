from __future__ import annotations

import json
from pathlib import Path

import joblib

from app.core.config import MODELS_DIR


class ModelLoader:
    """Loads model artifacts from `training/models/{module}/{version}/`.

    Mirrors the deployment-architecture.md v1.1 loading contract: artifacts are
    versioned directories with a `latest` symlink; feature columns and metadata
    travel alongside the estimator.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir

    def resolve(self, module: str, version: str = "latest") -> Path:
        """Resolve the directory containing a module's artifacts.

        Prefers a versioned subdirectory (`training/models/{module}/{version}/`,
        or a `latest` symlink). Falls back to the flat training-pipeline layout
        (`training/models/{module}/`), which is the current repo state.
        """
        root = self.models_dir / module / version
        if not root.exists():
            root = self.models_dir / module / "latest"
        if not root.exists():
            root = self.models_dir / module
        if not root.exists():
            raise FileNotFoundError(f"model artifacts missing: {module}/{version}")
        return root

    def load_joblib(self, module: str, name: str, version: str = "latest"):
        root = self.resolve(module, version)
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"artifact missing: {module}/{name}")
        return joblib.load(path)

    def load_json(self, module: str, name: str, version: str = "latest") -> dict:
        root = self.resolve(module, version)
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"artifact missing: {module}/{name}")
        return json.loads(path.read_text())
