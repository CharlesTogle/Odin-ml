# models/forecaster/

Canonical home for the **new-scope** spending forecaster artifact (`metadata.json` + final
`.joblib` / `.pth` + scaler). Old-scope winners (tier2_rf, tier3_gru/lstm/bilstm) stay in
`training/figures/models/forecaster/` and are reference only.

## Decision rule (pre-registered)

Winner must beat the naive baseline by **≥ 20% MAPE reduction**.

## Primary metrics

- MAE, SMAPE, MDA, RMSE

## Target artifact

- `forecaster.joblib` (or `captain.pth` + `_meta.joblib` for a PyTorch winner) — committed here
  once trained.
- `app/models/registry.py` (`FORECASTER_MODULE = "forecaster"`) loads it at serve time.

## Skeleton

- `metadata.example.json` — schema template; copy to `metadata.json` on promotion.
- `.gitkeep` — keeps the empty family dir tracked until the trained artifact lands.
