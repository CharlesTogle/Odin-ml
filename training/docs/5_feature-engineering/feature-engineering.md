# Feature Engineering Pipeline

**Module:** Feature Engineering
**Script:** `scripts/feature_engineering.py`
**Input:** `datasets/processed/` (preprocessed raw data)
**Output:** `datasets/engineered/` (engineered feature matrices)

---

## Purpose

The feature engineering pipeline transforms preprocessed raw data into enriched feature matrices ready for model training. It is a separate step from preprocessing, ensuring clear separation of concerns.

Key design principles:
1. **Features computed incrementally** — each month uses only historical data (months 1..M)
2. **Select/transform fit on train only** — no data leakage to val/test
3. **Two computation modes** — incremental (full synthesis data) or direct (preprocessed rows)
4. **Drop-in redundant feature removal** — addresses multicollinearity identified in EDA
5. **Feature selection + PCA** — dimensionality reduction options

## Architecture

```
scripts/feature_engineering.py
│
├── Step 1: Load Preprocessed Data
│   ├── train.parquet, val.parquet, test.parquet
│   └── Metadata (scalers, folds, split, feature columns)
│
├── Step 2: Load Synthesis Data (for incremental computation)
│   ├── monthly_summaries.parquet (per-persona-month raw data)
│   ├── personas.parquet (fbp_label lookup)
│   └── transactions.parquet (anomaly info)
│
├── Step 3: Compute Derived Features
│   ├── Mode A (incremental): compute_features_for_persona()
│   │   For each persona, for each month M (1..12):
│   │   ├── Compute features using only months 1..M
│   │   └── 17 derived features
│   ├── Mode B (direct): compute_features_from_row()
│   │   └── Single-pass computation from cumulative raw columns
│
├── Step 4: Impute + Cap Outliers (fit on train only)
│   ├── impute_features(): median for critical, 0 for others
│   └── cap_outliers(): 99th percentile for debt_to_income, savings_rate
│
├── Step 5: Cyclical Encoding (sin/cos)
│   └── month → month_sin, month_cos (period=12)
│
├── Step 6: Interaction Features
│   ├── income_volatility_interaction: income_stability_cv × total_income
│   └── obligation_volatility_interaction: obligation_ratio × income_stability_cv
│
├── Step 7: Redundant Feature Removal
│   └── Drop highly correlated pairs (|r| > 0.98):
│       ├── essential_ratio (redundant with obligation_ratio)
│       ├── savings_income_ratio (redundant with savings_rate)
│       └── expense_regularity (redundant with category_entropy)
│
├── Step 8: Feature Selection
│   ├── mutual_info: SelectKBest with mutual information
│   └── anova: SelectKBest with ANOVA F-test
│
├── Step 9: PCA Dimensionality Reduction (optional)
│   └── Configurable variance threshold or component count
│
└── Step 10: Export
    ├── train.parquet, val.parquet, test.parquet
    ├── feature_columns.json
    ├── pipeline_report.json
    └── Carried forward: scalers.json, split_metadata.json, temporal_folds.json
```

## CLI Usage

```bash
# Minimal (17 derived features + cyclical encoding + drop redundant)
python scripts/feature_engineering.py

# Feature selection with mutual information
python scripts/feature_engineering.py \
  --select-method mutual_info \
  --select-k 12

# Feature selection with ANOVA
python scripts/feature_engineering.py \
  --select-method anova \
  --select-k 10

# PCA dimensionality reduction
python scripts/feature_engineering.py \
  --pca-variance 0.95

# PCA with fixed components
python scripts/feature_engineering.py \
  --pca-components 8

# Full pipeline
python scripts/feature_engineering.py \
  --input datasets/processed/ \
  --synth-dir synth/ \
  --output datasets/engineered/ \
  --encoding cyclical \
  --select-method mutual_info \
  --select-k 12 \
  --pca-variance 0.95 \
  --seed 42

# Skip redundant feature removal
python scripts/feature_engineering.py --no-drop-redundant

# No encoding (just 17 derived features)
python scripts/feature_engineering.py --encoding none
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `datasets/processed/` | Input directory with preprocessed data |
| `--synth-dir` | `synth/` | Synthesis data directory (for incremental computation) |
| `--output` | `datasets/engineered/` | Output directory for engineered features |
| `--encoding` | `cyclical` | Temporal encoding: `cyclical` or `none` |
| `--select-method` | `None` | Feature selection: `mutual_info` or `anova` |
| `--select-k` | `None` | Number of features to select |
| `--pca-variance` | `None` | PCA variance threshold (e.g., 0.95 for 95%) |
| `--pca-components` | `None` | Number of PCA components |
| `--no-drop-redundant` | `False` | Skip dropping redundant features |
| `--seed` | `42` | Random seed for reproducibility |

## Feature Sets

### Derived Features (17 finite-window features)

| Feature | Formula | Computation Window | Expected Range |
|---------|---------|-------------------|----------------|
| `income_stability_cv` | σ(income) / μ(income) | Cumulative (min 2 months) | [0, 5] |
| `obligation_ratio` | essential / total expenses | Cumulative | [0, 1] |
| `savings_rate` | savings / income | Cumulative | [-1, 1] |
| `debt_to_income` | debt / income | Cumulative | [0, 10] |
| `essential_ratio` | essential / total | Cumulative | [0, 1] |
| `discretionary_ratio` | discretionary / total | Cumulative | [0, 1] |
| `income_trend` | linear slope of income | Cumulative (min 3 months) | unbounded |
| `expense_trend` | linear slope of expenses | Cumulative (min 3 months) | unbounded |
| `volatility_index` | rolling std of expenses | Cumulative (min 2 months) | unbounded |
| `category_entropy` | Shannon entropy of categories | Cumulative | [0, 2] |
| `transaction_frequency` | count / months active | Cumulative | unbounded |
| `avg_transaction_size` | mean transaction amount | Cumulative | unbounded |
| `income_regularity` | % months with income > 0 | Cumulative | [0, 1] |
| `expense_regularity` | % months with expenses > 0 | Cumulative | [0, 1] |
| `income_expense_gap` | income - expenses | Cumulative | unbounded |
| `essential_income_ratio` | essential / income | Cumulative | unbounded |
| `savings_income_ratio` | savings / income | Cumulative | unbounded |

### New Features Added in Phase 5

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `month_sin`, `month_cos` | Cyclical | Sin/cos encoding of month (period=12) | MDD §5 |
| `income_volatility_interaction` | Interaction | income_stability_cv × total_income | feature-set.md §2.3 |
| `obligation_volatility_interaction` | Interaction | obligation_ratio × income_stability_cv | feature-set.md §2.3 |

### Redundant Features (auto-dropped when `--drop-redundant` is enabled)

Based on EDA correlation analysis (|r| > 0.98 threshold):

| Kept | Dropped | Correlation |
|------|---------|-------------|
| `obligation_ratio` | `essential_ratio` | r = 1.000 |
| `savings_rate` | `savings_income_ratio` | r = 0.999 |
| `category_entropy` | `expense_regularity` | r = 0.992 |

## Normalization Strategy

The preprocessor exports scalers (`scalers.json`) fitted on the raw training data. These scalers are carried forward to `datasets/engineered/` unchanged.

Future work: Move normalization into this module once the separation is fully validated.

## Two Computation Modes

### Mode A: Incremental (preferred)

Requires synthesis data in `synth/`. Computes features per persona per month using only historical data (months 1..M). This is the same methodology as the original preprocessor.

```bash
python scripts/feature_engineering.py --synth-dir synth/
```

### Mode B: Direct (fallback)

Works without synthesis data. Computes features directly from each row's cumulative raw columns in the preprocessed data. Less accurate for trend/volatility features but useful when synth data is unavailable.

*Automatic fallback when `monthly_summaries.parquet` is missing.*

## Output Files

### `train.parquet`, `val.parquet`, `test.parquet`

Feature matrices with metadata, raw, and engineered columns. Schema includes:
- `user_id`, `month`, `fbp_label`, `is_anomalous`, `anomaly_type`
- `total_income`, `total_expenses`, `food_expense`, ... (raw)
- All selected engineered features (derived + cyclical + interaction)

### `feature_columns.json`

Column metadata including feature names, ranges, selection scores, and redundant drop info.

### `pipeline_report.json`

Quality report with feature counts, selection info, PCA variance, and split sizes.

### Carried Forward from Preprocessor

- `scalers.json` — normalization parameters (train-fitted)
- `split_metadata.json` — persona assignments per split
- `temporal_folds.json` — walk-forward validation folds

## Missing Value Handling

| Feature Type | Imputation Strategy |
|-------------|-------------------|
| Critical (CV, obligation_ratio) | Median from training set |
| Other derived features | 0 |
| Interaction features | computed from imputed inputs |

## Outlier Handling

| Feature | Method |
|---------|--------|
| `debt_to_income` | Capped at 99th percentile (from training set) |
| `savings_rate` | Capped at 99th percentile (from training set) |

## Leakage Prevention

| # | Check | Implementation |
|---|-------|---------------|
| 1 | Incremental features | Each month's features use only months 1..M |
| 2 | Train-only imputation | Medians computed from train only |
| 3 | Train-only outlier caps | 99th percentile from train only |
| 4 | Train-only selection | Feature selection fit on train only |
| 5 | Train-only PCA | PCA fit on train only |
| 6 | No lookahead | Features never use future data |

## Dependencies

```
pandas
numpy
scikit-learn
pyarrow
```

No new dependencies — all are already in `requirements.txt`.

## Testing

```bash
# Full feature engineering pipeline
python scripts/feature_engineering.py \
  --input datasets/processed/ \
  --output datasets/engineered/ \
  --encoding cyclical \
  --seed 42

# Verify outputs
python -c "
import pandas as pd
for split in ['train', 'val', 'test']:
    df = pd.read_parquet(f'datasets/engineered/{split}.parquet')
    print(f'{split}: {df.shape[0]} rows, {df[\"user_id\"].nunique()} personas, {len(df.columns)} columns')
"

# Quick test with feature selection
python scripts/feature_engineering.py \
  --select-method mutual_info \
  --select-k 10 \
  --output datasets/engineered_test/
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Separate module | Feature engineering decoupled from preprocessing | Separation of concerns, independent development |
| Two computation modes | Incremental + direct | Flexibility for different data availability scenarios |
| Cyclical encoding | Sin/cos for month | Preserves cyclic nature (month 12 adjacent to month 1) |
| Redundant feature drop | |r| > 0.98 threshold | Based on EDA findings; prevents multicollinearity |
| Mutual info selection | Default feature selection method | Handles non-linear relationships |
| PCA as optional | Variance threshold or component count | User chooses based on model requirements |
| Parquet output | Columnar, compressed | Fast I/O for large datasets |
