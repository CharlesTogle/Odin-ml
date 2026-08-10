from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app.core.config import SERVICE_NAME, SERVICE_VERSION
from app.models.registry import ModelRegistry
from app.api.deps import get_registry

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"service": SERVICE_NAME, "status": "ok"}


@router.get("/ready")
async def ready(registry: ModelRegistry = Depends(get_registry)) -> dict:
    if registry.is_ready:
        return {"status": "ready", "models": ["pfp", "forecaster", "anomaly"]}
    return {"status": "not ready", "reason": "model artifacts not loaded"}


@router.get("/metrics")
async def metrics(registry: ModelRegistry = Depends(get_registry)) -> dict:
    start = time.perf_counter()
    modules = {}
    for name, module in (
        ("pfp", registry.pfp),
        ("forecaster", registry.forecaster),
        ("anomaly", registry.anomaly),
    ):
        if module is not None:
            modules[name] = {
                "model_version": module.evaluation.get("timestamp", ""),
                "winner": module.evaluation.get("winner", ""),
            }
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "modules": modules,
        "check_latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
