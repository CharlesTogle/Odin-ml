# Odin-ML Documentation

## Structure

Documentation in this repository is split by purpose:

| Path | Contains |
| :--- | :--- |
| `docs/standards/` | Enforceable engineering and documentation standards for this repository (Python/ML). |
| `docs/models/` | Model candidate roster (RRL-grounded) and model-lifecycle guidance (see `models/README.md`). |
| `training/docs/` | ML design documents produced by the training pipeline (data collection, EDA, dimension discovery). |
| `training/docs/phases/` | Model development runbooks: Phase 7 evaluation, 8 selection/versioning, 9 deployment, 10 monitoring. |

## Conventions

- Standards that govern code quality live in `docs/standards/`.
- Model scope and artifacts follow the guidance in `docs/models/` and top-level `models/README.md`.
- Generated ML design reports (EDA, dimension-threshold candidates, data collection) live under `training/docs/` alongside the artifacts they describe.
- Follow the shared formatting rules in `docs/standards/documentation-format.md`.
