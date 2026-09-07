# models/

Canonical home for **new-scope final model artifacts** for the revised BUDI ML service.

## Relationship to old-scope artifacts

The previous training run produced "winner" artifacts for the original scope. Those artifacts
live in `training/figures/models/` and are **not** moved — they belong to the superseded scope
and are kept in place for reference only. This `models/` directory is the target for the
revised pipeline (model candidate roster + phase 7–10 process).

## Layout

```text
models/
  README.md        # This file
  pfp/             # PFP classifier — final .joblib + metadata.json
  forecaster/      # Spending forecaster — final artifact + metadata.json
  anomaly/         # Anomaly detector — final artifact + metadata.json
  budget/          # Budget optimizer (LP scipy) — config + metadata.json
```

## Model artifact standards

Every final model committed to `models/` **must** be accompanied by a `metadata.json` with:

```json
{
  "model_id": "pfp-v1.0.0",
  "family": "pfp",
  "created_at": "2026-09-07T00:00:00Z",
  "training_commit": "<git sha of training code>",
  "training_data_hash": "<sha256 of source parquet or feature matrix>",
  "framework": "scikit-learn",
  "framework_version": "1.9.0",
  "python_version": "3.14.4",
  "artifacts": ["pfp_classifier.joblib"],
  "feature_columns": ["income_stability_cv", "obligation_ratio", "..."],
  "metrics": {
    "primary": {"name": "macro_f1", "value": 0.675, "folds": 5},
    "secondary": {"accuracy": 0.678}
  },
  "decision_rule": "winner must beat tier1_rule_based by 0.02 macro-F1",
  "fitted": true,
  "serving_note": "loaded by app/models/loader.py"
}
```

Minimum required keys: `model_id`, `family`, `created_at`, `training_commit`,
`training_data_hash`, `framework`, `feature_columns`, `metrics`.

## Gitignoring

Large checkpoints and intermediate artifacts belong in `training/models/` (gitignored).
Only the **final selected model** for each family plus `metadata.json` is committed here.
Keep total committed artifact size small (< ~50 MB per model) — else prefer a release
binary store and reference it from `metadata.json`.