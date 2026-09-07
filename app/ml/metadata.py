"""Reproducibility metadata generation for final model artifacts.

Every train pipeline emits a `metadata.json` next to its final artifact in the
tracked top-level `models/<family>/` directory (schema in `models/README.md`).
This module centralizes the bits that are identical across families: the
training-data hash, the training commit, environment pins, and the JSON writer,
so each `train_*.py` only supplies the family-specific fields.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex sha256 of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def data_hash(paths: list[Path]) -> str:
    """Combine hashes of multiple source files into a single provenance hash.

    `paths` should be the raw inputs that deterministically produce the feature
    matrix (e.g. the unprocessed/processed parquet, or the feature matrix
    itself). The returned value is a stable sha256 computed over each file's
    own sha256 plus its relative name.
    """
    hasher = hashlib.sha256()
    for p in paths:
        resolved = Path(p)
        if not resolved.exists():
            raise FileNotFoundError(f"cannot hash missing input: {resolved}")
        part = f"{resolved.name}:{sha256_file(resolved)}".encode()
        hasher.update(part)
    return hasher.hexdigest()


def training_commit() -> str:
    """Return the current git commit short sha, or 'dirty'/'unknown'.

    Best-effort: never raises. Falls back to 'unknown' when not in a git repo
    or git is unavailable.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
        return "unknown"
    except OSError, subprocess.SubprocessError:
        return "unknown"


def python_version() -> str:
    return platform.python_version()


def build_metadata(
    *,
    model_id: str,
    family: str,
    feature_columns: list[str],
    metrics: dict[str, Any],
    decision_rule: str,
    framework: str,
    framework_version: str,
    artifacts: list[str],
    data_sources: list[Path],
    winner_reason: str | None = None,
    serving_note: str = "loaded by app/models/loader.py",
    fitted: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a metadata.json dict per the models/README.md schema."""
    meta: dict[str, Any] = {
        "model_id": model_id,
        "family": family,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "training_commit": training_commit(),
        "training_data_hash": data_hash(data_sources),
        "framework": framework,
        "framework_version": framework_version,
        "python_version": python_version(),
        "artifacts": artifacts,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "decision_rule": decision_rule,
        "fitted": fitted,
        "serving_note": serving_note,
    }
    if winner_reason:
        meta["winner_reason"] = winner_reason
    if extra:
        meta.update(extra)
    return meta


def write_metadata(metadata: dict[str, Any], output_dir: Path) -> Path:
    """Write metadata.json into `output_dir`, returning the written path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, default=str))
    return path


def framework_version_of(pkg: str) -> str:
    """Return the installed version of `pkg`, or 'unknown'."""
    try:
        import importlib.metadata as md

        return md.version(pkg)
    except md.PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"
    except Exception:  # pragma: no cover - defensive
        return "unknown"


__all__ = [
    "sha256_file",
    "sha256_text",
    "data_hash",
    "training_commit",
    "python_version",
    "build_metadata",
    "write_metadata",
    "framework_version_of",
]
