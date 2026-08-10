from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.common import ApiMetadata, ModuleStatus, Transaction

CLASSIFIER_MODES = ("STANDARD", "QUESTIONNAIRE")


class ClassificationMode(str, Enum):
    STANDARD = "STANDARD"
    QUESTIONNAIRE = "QUESTIONNAIRE"


class QuestionnaireAnswers(BaseModel):
    income_variability: str | None = None
    obligation_level: str | None = None
    emergency_runway: str | None = None


class PFPClassifyRequest(BaseModel):
    user_id: str
    classification_mode: ClassificationMode
    payload: dict


class PFPClassification(BaseModel):
    prediction: str
    financial_stability_score: float = Field(ge=0, le=1)
    financial_weight_score: float = Field(ge=0, le=1)
    financial_tolerance_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    status: ModuleStatus
    tier_used: int
    model_name: str


class PFPClassifyResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    classification: PFPClassification
    metadata: ApiMetadata
