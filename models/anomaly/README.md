# models/anomaly/

Canonical home for the **new-scope** anomaly detector artifact (`metadata.json` + final
`.joblib`). Old-scope winners (tier1_iqr, tier2_isoforest/ocsvm/autoencoder, tier3_hybrid) stay
in `training/figures/models/anomaly/` and are reference only.

## Decision rule (pre-registered)

Winner must beat the IQR baseline by **≥ 50% F1** improvement **and** reach **F1 ≥ 0.85**;
otherwise fall back to IQR.

## Primary metrics

- Accuracy, Precision, Recall, F1 (+ PR-AUC, ROC-AUC)

## Target artifact

- `anomaly_detector.joblib` — committed here once trained.
- `app/models/registry.py` (`ANOMALY_MODULE = "anomaly"`, `ANOMALY_ARTIFACT`) loads it at serve
  time.

## Skeleton

- `metadata.example.json` — schema template; copy to `metadata.json` on promotion.
- `.gitkeep` — keeps the empty family dir tracked until the trained artifact lands.
