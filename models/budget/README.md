# models/budget/

Canonical home for the **new-scope** budget optimizer configuration (`metadata.json` + config).
Budget optimization is a deterministic LP (`scipy.linprog`) so there is no trained artifact;
the "model" is the constraint formulation + configuration.

## Decision rule / metrics

- Constraint Satisfaction Rate, Budget Utilization Rate, Deviation from User Preferences.

## Target artifact

- `budget_config.json` + `metadata.json` — the current v1 configuration (`app/services/budget_service.py`).

## Skeleton

- `metadata.example.json` — schema template; copy to `metadata.json` on promotion.
- `.gitkeep` — keeps the empty family dir tracked until the config lands.
