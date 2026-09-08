# models/anomaly/

Canonical home for the **new-scope** anomaly detector artifact (`metadata.json` + final
`.joblib`). Current winner: **`tier1_iqr`** (`anomaly_detector.joblib`) — IQR baseline kept
under the pre-registered fallback rule after adaptive-threshold + hybrid tiers were evaluated
(see `evaluation.json` / `evaluation_report.md`).

## Decision rule (pre-registered)

Winner must beat the IQR baseline by **≥ 50% F1** improvement **and** reach **F1 ≥ 0.85**;
otherwise fall back to IQR.

## Primary metrics

- Accuracy, Precision, Recall, F1 (+ PR-AUC, ROC-AUC)

## Target artifact

- `anomaly_detector.joblib` — committed here (winner resolved from `metadata.json` at serve
  time).
- `app/models/registry.py` (`ANOMALY_MODULE = "anomaly"`) loads it at serve time via
  `_resolve_anomaly`/metadata-based resolution.

## Committed state

- `metadata.json` — winner contract (`winner`, `winner_artifact`, `winner_params`, `threshold`).
- `evaluation.json` — raw fold metrics (source of truth for the report).
- `evaluation_report.md` — regenerated from `evaluation.json`.

Regenerate reports/metadata from existing `evaluation.json` without retraining:

```bash
python training/scripts/regenerate_artifacts.py
```
