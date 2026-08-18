from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.common import ApiMetadata, ModuleStatus, Transaction


class DetectionType(str, Enum):
    ANOMALOUS = "ANOMALOUS"
    OVERSPENDING = "OVERSPENDING"
    BOTH = "BOTH"


class BudgetAllocationInput(BaseModel):
    category_id: str
    budget_amount: float = Field(ge=0)


class WhitelistEntry(BaseModel):
    category: str | None = None
    transaction_id: str | None = None


class AnomalyRequest(BaseModel):
    user_id: str
    transactions: list[Transaction] = Field(min_length=1)
    detection_type: DetectionType = DetectionType.ANOMALOUS
    budget_allocations: list[BudgetAllocationInput] | None = None
    whitelist: list[WhitelistEntry] | None = None
    user_metadata: dict | None = None


class AnomalousTransaction(BaseModel):
    transaction_id: str
    anomaly_score: float = Field(ge=0, le=1)
    reason: str
    feature_contributions: list[str]


class OverspendingTransaction(BaseModel):
    transaction_id: str
    budget_excess: float
    category: str
    period: str | None = None


class AnomalyResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    anomalous_transactions: list[AnomalousTransaction]
    overspending_transactions: list[OverspendingTransaction]
    model_version: str
    confidence: float = Field(ge=0, le=1)
    status: ModuleStatus
    metadata: ApiMetadata
