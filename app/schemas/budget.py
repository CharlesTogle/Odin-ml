from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.common import ApiMetadata, ModuleStatus


class RestrictionLevel(str, Enum):
    PROTECTED = "PROTECTED"
    FIXED = "FIXED"
    FREE = "FREE"


class BudgetCategory(BaseModel):
    category_id: str
    restriction_level: RestrictionLevel
    floor: float = Field(ge=0)
    ceiling: float = Field(ge=0)
    priority_weight: float = Field(ge=0, le=1)
    current_spend: float = Field(ge=0, default=0)


class BudgetRequest(BaseModel):
    user_id: str
    available_funds: float = Field(gt=0)
    categories: list[BudgetCategory] = Field(min_length=1)
    target_ratios: dict[str, float] | None = None
    forecast: dict | None = None
    include_reasoning: bool = True


class BudgetAllocation(BaseModel):
    category_id: str
    amount: float = Field(ge=0)


class BudgetRecommendation(BaseModel):
    allocations: list[BudgetAllocation]
    utilization_rate: float = Field(ge=0, le=1)
    constraint_satisfaction: float = Field(ge=0, le=1)
    feasibility: str


class BudgetExplanation(BaseModel):
    category_id: str
    reason: str


class BudgetResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    recommendation: BudgetRecommendation
    explanations: list[BudgetExplanation] | None = None
    status: ModuleStatus
    metadata: ApiMetadata
