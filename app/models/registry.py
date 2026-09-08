from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from app.models.artifact_classes import (
    _SequenceForecaster,
)
from app.models.loader import ModelLoader

PFP_MODULE = "pfp"
PFP_ARTIFACT = "tier3_svm.joblib"
FORECASTER_MODULE = "forecaster"
ANOMALY_MODULE = "anomaly"
ANOMALY_ARTIFACT = "anomaly_detector.joblib"

logger = logging.getLogger(__name__)


@dataclass
class ModuleModel:
    module: str
    model: Any
    evaluation: dict
    feature_columns: list[str]
    threshold: float | None = None


def _resolve_forecaster_artifact(evaluation: dict, output_dir) -> tuple[str, Any]:
    """Load the forecaster winner artifact from evaluation.json."""
    winner = evaluation.get("winner", "tier2_random_forest")
    if winner == "tier2_random_forest":
        artifact = "tier2_random_forest.joblib"
        model = ModelLoader().load_joblib(FORECASTER_MODULE, artifact)
        return artifact, model
    # Statsmodels ARIMA winner (pooled, user-normalized forecaster)
    if winner == "tier3_arima":
        artifact = "tier3_arima.joblib"
        model = ModelLoader().load_joblib(FORECASTER_MODULE, artifact)
        return artifact, model
    # PyTorch winner (tier3_gru, tier3_lstm, tier3_bilstm)
    pth_path = output_dir / f"{winner}.pth"
    meta_path = output_dir / f"{winner}_meta.joblib"
    if pth_path.exists() and meta_path.exists():
        import joblib

        meta = joblib.load(str(meta_path))
        state = torch.load(str(pth_path), map_location="cpu", weights_only=True)
        variant = state.get("model_type", winner.replace("tier3_", ""))
        input_size = state.get("input_size", 20)
        hidden_size = state.get("hidden_size", 32)
        seq_length = state.get("seq_length", 3)
        model = _SequenceForecaster(input_size, hidden_size, variant)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return winner, {
            "model": model,
            "scaler": meta["scaler"],
            "feature_cols": meta["feature_cols"],
            "seq_length": seq_length,
        }
    # Fallback to RF
    return "tier2_random_forest.joblib", ModelLoader().load_joblib(
        FORECASTER_MODULE, "tier2_random_forest.joblib"
    )


class ModelRegistry:
    """Loads and exposes all module artifacts at startup (lifespan)."""

    def __init__(self, loader: ModelLoader | None = None):
        self.loader = loader or ModelLoader()
        self.pfp: ModuleModel | None = None
        self.forecaster: ModuleModel | None = None
        self.anomaly: ModuleModel | None = None

    def load_all(self) -> None:
        self.pfp = self._load_optional(PFP_MODULE, PFP_ARTIFACT)
        try:
            self.forecaster = self._load_forecaster()
        except FileNotFoundError as exc:
            logger.warning("forecaster artifacts not found; skipping: %s", exc)
            self.forecaster = None
        try:
            self.anomaly = self._load_anomaly()
        except FileNotFoundError as exc:
            logger.warning("anomaly artifacts not found; skipping: %s", exc)
            self.anomaly = None

    def _load_optional(self, module: str, artifact: str) -> ModuleModel | None:
        try:
            return self._load_sklearn(module, artifact)
        except FileNotFoundError as exc:
            logger.warning("%s artifacts not found; skipping: %s", module, exc)
            return None

    def _load_sklearn(self, module: str, artifact: str) -> ModuleModel:
        evaluation = self.loader.load_json(module, "evaluation.json")
        model = self.loader.load_joblib(module, artifact)
        feature_columns = evaluation.get("feature_columns") or []
        if not feature_columns and isinstance(model, dict):
            feature_columns = list(model.get("feature_cols", []))
        return ModuleModel(
            module=module, model=model, evaluation=evaluation, feature_columns=feature_columns
        )

    def _load_forecaster(self) -> ModuleModel:
        evaluation = self.loader.load_json(FORECASTER_MODULE, "evaluation.json")
        feature_columns = evaluation.get("feature_columns", [])
        output_dir = self.loader.resolve(FORECASTER_MODULE)
        artifact_name, model = _resolve_forecaster_artifact(evaluation, output_dir)
        return ModuleModel(
            module=FORECASTER_MODULE,
            model=model,
            evaluation=evaluation,
            feature_columns=feature_columns,
        )

    def _load_anomaly(self) -> ModuleModel:
        evaluation = self.loader.load_json(ANOMALY_MODULE, "evaluation.json")
        feature_columns = evaluation.get("feature_columns", [])
        model = self.loader.load_joblib(ANOMALY_MODULE, ANOMALY_ARTIFACT)
        threshold = (
            evaluation.get("val_selected_threshold")
            or evaluation.get("final_test_metrics", {}).get("best_threshold")
            or evaluation.get("threshold")
        )
        return ModuleModel(
            module=ANOMALY_MODULE,
            model=model,
            evaluation=evaluation,
            feature_columns=feature_columns,
            threshold=threshold,
        )

    @property
    def loaded_modules(self) -> list[str]:
        return [
            name
            for name, module in (
                (PFP_MODULE, self.pfp),
                (FORECASTER_MODULE, self.forecaster),
                (ANOMALY_MODULE, self.anomaly),
            )
            if module is not None
        ]

    @property
    def is_ready(self) -> bool:
        core = (self.forecaster, self.anomaly)
        return all(m is not None for m in core)
