from __future__ import annotations

import numpy as np

from app.models.registry import ModuleModel
from app.schemas.common import ModuleStatus
from app.schemas.pfp import PFPClassification, PFPClassifyRequest
from app.services.features import pfp_feature_vector


def _calibrated_scores(vector: np.ndarray, feature_cols: list[str]) -> dict[str, float]:
    values = {name: float(vector[0][i]) for i, name in enumerate(feature_cols)}
    stability_cv = values.get("income_stability_cv", 1.0)
    obligation = values.get("obligation_ratio", 0.5)
    savings_rate = values.get("savings_rate", 0.0)

    stability = max(0.0, min(1.0, 1.0 - stability_cv / 0.5))
    weight = max(0.0, min(1.0, obligation))
    tolerance = max(0.0, min(1.0, savings_rate / 0.5))
    return {
        "stability": round(stability, 4),
        "weight": round(weight, 4),
        "tolerance": round(tolerance, 4),
    }


def classify_standard(model: ModuleModel, request: PFPClassifyRequest) -> PFPClassification:
    transactions = request.payload.get("historical_transactions") or []
    vector = pfp_feature_vector(transactions, model.feature_columns)

    estimator = model.model
    prediction = str(estimator.predict(vector)[0])
    proba = estimator.predict_proba(vector)[0]
    classes = list(estimator.classes_)

    scores = _calibrated_scores(vector, model.feature_columns)
    return PFPClassification(
        prediction=prediction,
        financial_stability_score=scores["stability"],
        financial_weight_score=scores["weight"],
        financial_tolerance_score=scores["tolerance"],
        confidence=round(float(max(proba)), 4),
        status=ModuleStatus.SUCCESS,
        tier_used=3,
        model_name="random_forest",
    )


def classify_questionnaire(request: PFPClassifyRequest) -> PFPClassification:
    """Deterministic mapping from questionnaire answers to a PFP class.

    Tier 0 fallback (no model inference) for cold-start users, per the
    PFP MDD: QUESTIONNAIRE mode is the system's cold-start answer.
    """
    answers = request.payload.get("questionnaire_answers") or {}
    variability = str(answers.get("income_variability", "")).lower()
    obligation = str(answers.get("obligation_level", "")).lower()
    runway = str(answers.get("emergency_runway", "")).lower()

    stability = 0.2 if "variable" in variability or "irregular" in variability else 0.8
    weight = 0.8 if "high" in obligation else (0.4 if "medium" in obligation else 0.2)
    tolerance = 0.2 if "low" in runway else (0.8 if "high" in runway or "3" in runway else 0.5)

    label_parts = []
    label_parts.append("Variable" if stability < 0.5 else "Stable")
    label_parts.append("Obligated" if weight >= 0.5 else "Flexible")
    label_parts.append("At-Risk" if tolerance < 0.5 else "Tolerant")
    prediction = "/".join(label_parts)

    confidence = round(max(stability, weight, tolerance), 4)
    return PFPClassification(
        prediction=prediction,
        financial_stability_score=round(stability, 4),
        financial_weight_score=round(weight, 4),
        financial_tolerance_score=round(tolerance, 4),
        confidence=confidence,
        status=ModuleStatus.SUCCESS,
        tier_used=0,
        model_name="questionnaire_rule",
    )


def classify(model: ModuleModel, request: PFPClassifyRequest) -> PFPClassification:
    if request.classification_mode.value == "QUESTIONNAIRE":
        return classify_questionnaire(request)
    return classify_standard(model, request)
