from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from app.schemas.budget import (
    BudgetAllocation,
    BudgetExplanation,
    BudgetRecommendation,
    BudgetRequest,
    RestrictionLevel,
)

EPS = 1e-9


def optimize(request: BudgetRequest) -> tuple[BudgetRecommendation, list[BudgetExplanation]]:
    """Solve the allocation LP (Budget Optimizer MDD v1.0, Tier 2).

    Decision variables: allocation `x_i` plus deviation `d_i` per category,
    where `d_i >= |x_i - target_i|`. The LP minimizes the weighted sum of
    deviations, so recommendations stay as close as possible to the user's
    target ratios (Deviation-from-User-Preferences KPI).

    Constraints:
      - sum of allocations == available funds,
      - LOCKED categories held at their fixed amount (floor = ceiling = fixed),
      - PROTECTED categories keep at least current spend,
      - FREE categories respect [floor, ceiling].

    Feasibility:
      - FEASIBLE: all constraints met, allocations sum to available funds.
      - REDUCED: LP finds a solution but ceiling constraints prevent full
        utilization of available funds.
      - INFEASIBLE: no allocation satisfies all hard constraints.
    """
    categories = request.categories
    n = len(categories)
    funds = request.available_funds

    target_ratios = request.target_ratios or {}
    targets = np.array([funds * target_ratios.get(c.category_id, 0.0) for c in categories])
    weights = np.array([max(cat.priority_weight, EPS) for cat in categories])

    for cat in categories:
        if cat.restriction_level == RestrictionLevel.LOCKED:
            if cat.ceiling < cat.floor:
                raise ValueError(f"category {cat.category_id}: ceiling below floor")
        elif cat.restriction_level == RestrictionLevel.PROTECTED:
            if cat.ceiling < cat.floor:
                raise ValueError(f"category {cat.category_id}: ceiling below floor")

    # Objective: minimize sum(w_i * d_i)  -> c = [0]*n allocations, [w] deviations
    c = np.concatenate([np.zeros(n), weights])

    # Inequality constraints A_ub @ [x, d] <= b_ub:
    #   x_i - d_i <= target_i   (under-shoot)
    #   -x_i - d_i <= -target_i (over-shoot)
    A_ub, b_ub = [], []
    for i in range(n):
        row_lo = [0.0] * (2 * n)
        row_lo[i] = 1.0
        row_lo[n + i] = -1.0
        A_ub.append(row_lo)
        b_ub.append(targets[i])

        row_hi = [0.0] * (2 * n)
        row_hi[i] = -1.0
        row_hi[n + i] = -1.0
        A_ub.append(row_hi)
        b_ub.append(-targets[i])

    # Bounds: allocations respect restriction levels; deviations are free.
    bounds = []
    for cat in categories:
        if cat.restriction_level == RestrictionLevel.LOCKED:
            lo = hi = cat.current_spend
        elif cat.restriction_level == RestrictionLevel.PROTECTED:
            lo = max(cat.floor, cat.current_spend)
            hi = max(cat.ceiling, lo)
        else:
            lo = cat.floor
            hi = cat.ceiling
        bounds.append((lo, hi))
    bounds.extend([(0.0, None)] * n)

    A_eq = [[1.0 if j < n else 0.0 for j in range(2 * n)]]
    b_eq = [funds]

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise ValueError(f"budget LP infeasible: {result.message}")

    allocations = np.maximum(result.x[:n], 0.0)
    utilization = float(allocations.sum()) / funds if funds > 0 else 0.0

    constraint_satisfaction = 1.0
    for i, cat in enumerate(categories):
        if cat.restriction_level == RestrictionLevel.LOCKED and abs(allocations[i] - cat.current_spend) > EPS:
            constraint_satisfaction = 0.0
        if allocations[i] < cat.floor - EPS or allocations[i] > cat.ceiling + EPS:
            constraint_satisfaction = 0.0

    # Determine feasibility per MDD: FEASIBLE / REDUCED / INFEASIBLE
    fully_utilized = utilization > 1.0 - EPS
    if constraint_satisfaction < 0.99:
        feasibility = "INFEASIBLE"
    elif not fully_utilized:
        feasibility = "REDUCED"
    else:
        feasibility = "FEASIBLE"
    recommendation = BudgetRecommendation(
        allocations=[
            BudgetAllocation(category_id=cat.category_id, amount=round(float(a), 2))
            for cat, a in zip(categories, allocations)
        ],
        utilization_rate=round(utilization, 4),
        constraint_satisfaction=round(constraint_satisfaction, 4),
        feasibility=feasibility,
    )

    explanations = []
    for cat, a in zip(categories, allocations):
        if cat.restriction_level == RestrictionLevel.LOCKED:
            explanations.append(BudgetExplanation(
                category_id=cat.category_id,
                reason=f"LOCKED category — held at {a:.2f}",
            ))
        elif cat.restriction_level == RestrictionLevel.PROTECTED:
            explanations.append(BudgetExplanation(
                category_id=cat.category_id,
                reason=f"PROTECTED category — held at {a:.2f}",
            ))
        elif a > 0:
            explanations.append(BudgetExplanation(
                category_id=cat.category_id,
                reason="Allocation set within floor/ceiling bounds",
            ))

    return recommendation, explanations
