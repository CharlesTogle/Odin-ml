from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_registry
from app.models.registry import ModelRegistry
from app.schemas.anomaly import (
    AnomalyRequest,
    AnomalyResponse,
    DetectionType,
    OverspendingTransaction,
)
from app.schemas.common import ApiMetadata, ModuleStatus
from app.services import anomaly_service
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/anomaly", tags=["anomaly"])


def _run(registry: ModelRegistry, request: AnomalyRequest) -> AnomalyResponse:
    start = time.perf_counter()
    anomalous = []
    overspending: list[OverspendingTransaction] = []
    status = ModuleStatus.SUCCESS

    try:
        if request.detection_type in (DetectionType.ANOMALOUS, DetectionType.BOTH):
            anomalous = anomaly_service.detect(registry.anomaly, request)

        if request.detection_type in (DetectionType.OVERSPENDING, DetectionType.BOTH):
            if not request.budget_allocations:
                raise ValueError("budget_allocations required when detection_type is OVERSPENDING or BOTH")
            transactions = [t.model_dump() for t in request.transactions]
            overspending = anomaly_service._overspending_detect(
                transactions, [b.model_dump() for b in request.budget_allocations]
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        status = ModuleStatus.FALLBACK

    # Confidence: max anomaly score if any anomalies detected, else 0.0
    scores = [a.anomaly_score for a in anomalous]
    confidence = round(max(scores), 4) if scores else 0.0

    return AnomalyResponse(
        response_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        user_id=request.user_id,
        anomalous_transactions=anomalous,
        overspending_transactions=overspending,
        model_version=registry.anomaly.evaluation.get("winner", "unknown"),
        confidence=confidence,
        status=status,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version="v2.3.0",
        ),
    )


class BatchRequest(BaseModel):
    requests: list[AnomalyRequest] = Field(min_length=1)


class BatchResponse(BaseModel):
    results: list[AnomalyResponse]


@router.post("/detect", response_model=AnomalyResponse)
async def detect(
    request: AnomalyRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> AnomalyResponse:
    return _run(registry, request)


@router.post("/detect/batch", response_model=BatchResponse)
async def detect_batch(
    batch: BatchRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BatchResponse:
    return BatchResponse(results=[_run(registry, req) for req in batch.requests])
