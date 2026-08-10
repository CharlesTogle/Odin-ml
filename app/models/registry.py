from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.artifact_classes import register_artifact_classes
from app.models.loader import ModelLoader

PFP_MODULE = "pfp"
PFP_ARTIFACT = "tier3_random_forest.joblib"
FORECASTER_MODULE = "forecaster"
FORECASTER_ARTIFACT = "tier2_random_forest.joblib"
ANOMALY_MODULE = "anomaly"
ANOMALY_ARTIFACT = "anomaly_detector.joblib"


@dataclass
class ModuleModel:
    module: str
    model: Any
    evaluation: dict
    feature_columns: list[str]


class ModelRegistry:
    """Loads and exposes all four module artifacts at startup (lifespan)."""

    def __init__(self, loader: ModelLoader | None = None):
        self.loader = loader or ModelLoader()
        self.pfp: ModuleModel | None = None
        self.forecaster: ModuleModel | None = None
        self.anomaly: ModuleModel | None = None

    def load_all(self) -> None:
        self.pfp = self._load_sklearn(PFP_MODULE, PFP_ARTIFACT)
        self.forecaster = self._load_sklearn(FORECASTER_MODULE, FORECASTER_ARTIFACT)
        self.anomaly = self._load_anomaly()

    def _load_sklearn(self, module: str, artifact: str) -> ModuleModel:
        evaluation = self.loader.load_json(module, "evaluation.json")
        model = self.loader.load_joblib(module, artifact)
        feature_columns = evaluation.get("feature_columns") or []
        if not feature_columns and isinstance(model, dict):
            feature_columns = list(model.get("feature_cols", []))
        return ModuleModel(module=module, model=model, evaluation=evaluation,
                           feature_columns=feature_columns)

    def _load_anomaly(self) -> ModuleModel:
        register_artifact_classes()
        evaluation = self.loader.load_json(ANOMALY_MODULE, "evaluation.json")
        feature_columns = evaluation.get("feature_columns", [])
        model = self.loader.load_joblib(ANOMALY_MODULE, ANOMALY_ARTIFACT)
        return ModuleModel(module=ANOMALY_MODULE, model=model, evaluation=evaluation,
                           feature_columns=feature_columns)

    @property
    def is_ready(self) -> bool:
        return all(m is not None for m in (self.pfp, self.forecaster, self.anomaly))
