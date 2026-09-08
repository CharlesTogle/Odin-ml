from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.common import ApiMetadata, ModuleStatus, Transaction


class ForecastHorizon(str, Enum):
    WEEKLY = "WEEKLY"
    SEMI_MONTHLY = "SEMI_MONTHLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"


class ForecastLevel(str, Enum):
    TOTAL = "TOTAL"
    CATEGORY_GROUP = "CATEGORY_GROUP"
    CATEGORY = "CATEGORY"


class ForecastRequest(BaseModel):
    user_id: str
    historical_transactions: list[Transaction] = Field(min_length=1)
    forecast_horizon: ForecastHorizon = ForecastHorizon.MONTHLY
    forecast_level: ForecastLevel = ForecastLevel.TOTAL
    target_categories: list[str] | None = None


class ForecastPoint(BaseModel):
    date: str
    amount: float
    category: str | None = None


class ConfidenceInterval(BaseModel):
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


class ForecastResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    forecasts: list[ForecastPoint]
    forecast_level: ForecastLevel
    forecast_horizon: ForecastHorizon
    confidence_intervals: ConfidenceInterval
    model_version: str
    status: ModuleStatus
    metadata: ApiMetadata
