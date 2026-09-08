# Phase 7 — Model Evaluation

**Status:** Complete (new-scope evaluation executed for all four families; artifacts live in
`models/<family>/`).

## Acceptance criteria (pre-registered decision rules)

| Family | Rule | Primary metric | Source of truth |
|---|---|---|---|
| PFP | Winner must beat Tier-1 rule-based floor by ≥ 0.02 Macro-F1 | Macro-F1 | `train_pfp.py` docstring + `models/pfp/evaluation.json` |
| Forecaster | Winner must beat naive baseline by ≥ 20% MAPE reduction | MAE, SMAPE, MDA, RMSE | `train_forecaster.py` docstring + `models/forecaster/evaluation.json` |
| Anomaly | Winner must beat IQR baseline by ≥ 50% F1 and reach F1 ≥ 0.85; else fall back to IQR | Accuracy, Precision, Recall, F1 | `train_anomaly.py` docstring + `models/anomaly/evaluation.json` |
| Budget | Constraint Satisfaction Rate, Budget Utilization Rate, Deviation from User Preferences | (LP feasibility / utilization) | Budget Optimizer MDD v1.0 |

## Evaluation protocol (fixed for all new-scope runs)

1. **Split integrity:** 5-fold expanding window from `training/datasets/processed/temporal_folds.json`; embargo months excluded from training labels (forecaster).
2. **No test leakage:** operating thresholds selected on the held-out val split only; test used exactly once at the end.
3. **Write `evaluation.json` + `evaluation_report.md`** next to the artifact in `models/<family>/`.
4. **Record:** per-tier metrics, winner, winner reason (rule restated), feature columns, timestamp, training-data hash, git commit.

## New-scope deliverable

Re-run `train_*.py` after any feature/label change. Output lands in `models/<family>/` with
`evaluation.json`, `evaluation_report.md`, and its `metadata.json` (schema in
`models/README.md`). To re-emit reports/metadata from an existing `evaluation.json` without
retraining: `python training/scripts/regenerate_artifacts.py`.