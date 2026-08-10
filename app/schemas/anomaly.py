from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import ApiMetadata, ModuleStatus, Transaction


class AnomalyRequest(BaseModel):
    user_id: str
    transactions: list[Transaction] = Field(min_length=1)


class AnomalyResult(BaseModel):
    transaction_id: str
    is_anomalous: bool
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    explanation: list[str]


class AnomalyResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    anomaly: AnomalyResult
    metadata: ApiMetadata
