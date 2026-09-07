# models/pfp/

Canonical home for the **new-scope** PFP classification artifact (`metadata.json` + final
`.joblib`). Old-scope winners (tier0/tier1/tier2/tier3) stay in `training/figures/models/pfp/`
and are reference only.

## Decision rule (pre-registered)

Winner must beat the Tier-1 rule-based floor by **≥ 0.02 Macro-F1** (5-fold expanding window).

## Primary metric

- Macro-F1 (primary); Accuracy (secondary)

## Target artifact

- `pfp_classifier.joblib` — committed here once trained.
- `app/models/registry.py` (`PFP_MODULE = "pfp"`, `PFP_ARTIFACT`) loads it at serve time.

## Skeleton

- `metadata.example.json` — schema template; copy to `metadata.json` on promotion.
- `.gitkeep` — keeps the empty family dir tracked until the trained artifact lands.
