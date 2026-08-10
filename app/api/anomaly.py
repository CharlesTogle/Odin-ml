from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_registry
from app.models.registry import ModelRegistry
from app.schemas.anomaly import AnomalyRequest, AnomalyResponse, AnomalyResult
from app.schemas.common import ApiMetadata, ModuleStatus
from app.services import anomaly_service

router = APIRouter(prefix="/api/v1/anomaly", tags=["anomaly"])


def _run(registry: ModelRegistry, request: AnomalyRequest) -> list[AnomalyResult]:
    start = time.perf_counter()
    try:
        results = anomaly_service.detect(registry.anomaly, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="anomaly detection failed") from exc
    return results


class BatchRequest(BaseModel):
    requests: list[AnomalyRequest] = Field(min_length=1)


class BatchResponse(BaseModel):
    results: list[list[AnomalyResult]]


@router.post("/detect", response_model=AnomalyResponse)
async def detect(
    request: AnomalyRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> AnomalyResponse:
    start = time.perf_counter()
    results = _run(registry, request)
    latest = results[-1] if results else AnomalyResult(
        transaction_id="", is_anomalous=False, score=0.0, threshold=0.0, explanation=[]
    )
    return AnomalyResponse(
        response_id=str(uuid.uuid4()),
        request_id=request.user_id,
        user_id=request.user_id,
        anomaly=latest,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version="v2.3.0",
        ),
    )


@router.post("/detect/batch", response_model=BatchResponse)
async def detect_batch(
    batch: BatchRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BatchResponse:
    return BatchResponse(results=[_run(registry, req) for req in batch.requests])
