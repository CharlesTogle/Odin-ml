from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_registry
from app.models.registry import ModelRegistry
from app.schemas.common import ApiMetadata, ModuleStatus
from app.schemas.forecast import (
    ConfidenceInterval,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
)
from app.services import forecast_service

router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])


def _run(registry: ModelRegistry, request: ForecastRequest) -> ForecastResponse:
    start = time.perf_counter()
    transactions = [t.model_dump() for t in request.historical_transactions]
    status = ModuleStatus.SUCCESS
    try:
        points, interval, level = forecast_service.forecast(registry.forecaster, request)
    except Exception:
        total, std = forecast_service.cold_start_estimate(transactions)
        points = [ForecastPoint(date="next", amount=round(total, 2))]
        interval = ConfidenceInterval(
            lower_80=round(total - std, 2),
            upper_80=round(total + std, 2),
            lower_95=round(total - 1.96 * std, 2),
            upper_95=round(total + 1.96 * std, 2),
        )
        level = "fallback"
        status = ModuleStatus.FALLBACK

    return ForecastResponse(
        response_id=str(uuid.uuid4()),
        request_id=request.user_id,
        user_id=request.user_id,
        forecasts=points,
        forecast_level=request.forecast_level,
        forecast_horizon=request.forecast_horizon,
        confidence_intervals=interval,
        model_version="v2.4.0",
        status=status,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version="v2.4.0",
            strategy_used=level,
        ),
    )


class BatchRequest(BaseModel):
    requests: list[ForecastRequest] = Field(min_length=1)


class BatchResponse(BaseModel):
    results: list[ForecastResponse]


@router.post("/predict", response_model=ForecastResponse)
async def predict(
    request: ForecastRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> ForecastResponse:
    return _run(registry, request)


@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    batch: BatchRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BatchResponse:
    return BatchResponse(results=[_run(registry, req) for req in batch.requests])
