# Forecaster — Financial Forecasting Evaluation Report

**Generated:** 2026-09-08T03:44:27.361005Z
**Folds:** 5
**Decision rule:** best learned model must cut MAPE by ≥ 20% vs naive baseline, else naive

## Winner

- **Tier:** tier3_sarima
- **Artifact:** tier3_sarima.joblib
- **Reason:** tier3_sarima reduces MAPE by 74.9% (>20% threshold)

## Approval Criteria Result

**Result:** PASS — tier3_sarima reduces MAPE by 74.9% (>20% threshold)
MAPE 9.40% vs naive 37.45% (-74.9%).

## Aggregate Results

| Tier | MAE (mean ± std) | SMAPE (mean ± std) | MDA (mean ± std) | RMSE (mean ± std) | MAPE (mean ± std) | R² (mean ± std) |
|------|-----------------|-------------------|-----------------|------------------|----------------|--------------|
| naive_baseline | 9350.3820 ± 8.7519 | 30.0178 ± 0.0317 | 0.5669 ± 0.0025 | 11807.96 ± 30.43 | 37.4519 ± 0.0569% | 0.0000 ± 0.0000 |
| tier3_arima | 3162.6796 ± 74.9620 | 9.3813 ± 0.2297 | 0.7126 ± 0.0076 | 5257.27 ± 99.44 | 9.3957 ± 0.2457% | 0.8017 ± 0.0066 |
| tier3_sarima | 3162.6796 ± 74.9620 | 9.3813 ± 0.2297 | 0.7126 ± 0.0076 | 5257.27 ± 99.44 | 9.3957 ± 0.2457% | 0.8017 ± 0.0066 |

## Per-Fold Results

### Fold 1

- Train months: [1, 2, 3, 4, 5, 6]
- Test months: [8]
- Train samples: 72000
- Test samples: 12000

| Tier | MAE | SMAPE | MDA | RMSE | MAPE | R² |
|------|-----|-------|-----|------|------|-----|
| naive_baseline | 9340.04 | 29.99% | 0.5711 | 11865.55 | 37.50% | -0.0000 |
| tier3_arima | 3291.39 | 9.77% | 0.6981 | 5447.06 | 9.82% | 0.7893 |
| tier3_sarima | 3291.39 | 9.77% | 0.6981 | 5447.06 | 9.82% | 0.7893 |

### Fold 2

- Train months: [1, 2, 3, 4, 5, 6, 7]
- Test months: [9]
- Train samples: 84000
- Test samples: 12000

| Tier | MAE | SMAPE | MDA | RMSE | MAPE | R² |
|------|-----|-------|-----|------|------|-----|
| naive_baseline | 9366.60 | 30.08% | 0.5643 | 11801.55 | 37.52% | -0.0000 |
| tier3_arima | 3166.55 | 9.40% | 0.7171 | 5240.45 | 9.42% | 0.8028 |
| tier3_sarima | 3166.55 | 9.40% | 0.7171 | 5240.45 | 9.42% | 0.8028 |

### Fold 3

- Train months: [1, 2, 3, 4, 5, 6, 7, 8]
- Test months: [10]
- Train samples: 96000
- Test samples: 12000

| Tier | MAE | SMAPE | MDA | RMSE | MAPE | R² |
|------|-----|-------|-----|------|------|-----|
| naive_baseline | 9348.89 | 30.02% | 0.5661 | 11775.79 | 37.45% | -0.0000 |
| tier3_arima | 3177.89 | 9.43% | 0.7118 | 5237.14 | 9.44% | 0.8022 |
| tier3_sarima | 3177.89 | 9.43% | 0.7118 | 5237.14 | 9.44% | 0.8022 |

### Fold 4

- Train months: [1, 2, 3, 4, 5, 6, 7, 8, 9]
- Test months: [11]
- Train samples: 108000
- Test samples: 12000

| Tier | MAE | SMAPE | MDA | RMSE | MAPE | R² |
|------|-----|-------|-----|------|------|-----|
| naive_baseline | 9349.01 | 29.99% | 0.5685 | 11803.80 | 37.42% | -0.0000 |
| tier3_arima | 3102.29 | 9.19% | 0.7185 | 5203.80 | 9.18% | 0.8056 |
| tier3_sarima | 3102.29 | 9.19% | 0.7185 | 5203.80 | 9.18% | 0.8056 |

### Fold 5

- Train months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
- Test months: [12]
- Train samples: 120000
- Test samples: 12000

| Tier | MAE | SMAPE | MDA | RMSE | MAPE | R² |
|------|-----|-------|-----|------|------|-----|
| naive_baseline | 9347.37 | 30.01% | 0.5647 | 11793.13 | 37.36% | -0.0000 |
| tier3_arima | 3075.29 | 9.11% | 0.7177 | 5157.88 | 9.13% | 0.8087 |
| tier3_sarima | 3075.29 | 9.11% | 0.7177 | 5157.88 | 9.13% | 0.8087 |

