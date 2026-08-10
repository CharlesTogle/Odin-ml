from __future__ import annotations

from fastapi import Request

from app.models.registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry
