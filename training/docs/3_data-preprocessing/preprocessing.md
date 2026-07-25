# Data Preprocessing Pipeline

**Module:** Data Preprocessing
**Script:** `scripts/preprocessor.py`
**Input:** FIES data file (CSV or Parquet)
**Output:** `datasets/processed/` (preprocessed raw data ready for feature engineering)

> **Note:** Feature engineering has been moved to Phase 5 (`docs/5_feature-engineering/`).
> The preprocessor now outputs raw data only. Run `scripts/feature_engineering.py`
> after preprocessing to produce engineered feature matrices.

---

## Purpose

The preprocessing pipeline runs synthesis (persona + transaction generation) and then transforms the output into labeled, normalized feature matrices ready for model training. It enforces strict data leakage prevention by:

1. Generating synthetic personas and transactions from FIES data
2. Splitting personas **before** any feature engineering
3. Computing features **incrementally** (each month uses only historical data)
4. Fitting normalization scalers **on training data only**
5. Generating temporal fold metadata for walk-forward validation

## Architecture

```
scripts/preprocessor.py
│
├── Step 1: Synthesis
│   ├── Load FIES data (CSV or Parquet)
│   ├── Validate columns (fies_columns.py)
│   ├── Filter NCR region
│   ├── Compute FIES statistics + expense ratios
│   ├── Generate 12 archetypes × 1,000 personas
│   ├── Generate 12-month transaction histories
│   └── Export to synth/ (personas, transactions, summaries)
│
├── Step 2: Load & Validate
│   ├── Load monthly_summaries.parquet from synth/
│   ├── Load personas.parquet from synth/
│   ├── Load anomaly info from transactions.parquet
│   └── Validate schema, data types, completeness
│
├── Step 3: Persona-Level Split (stratified by PFP label)
│   ├── Train 70% / Val 15% / Test 15% (configurable)
│   ├── StratifiedShuffleSplit to preserve PFP distribution
│   ├── Entire persona → one split (no persona-level mixing)
│   └── Save: split_metadata.json
│
├── Step 4: Build Raw Feature Matrices
│   ├── For each split, merge summaries + personas + anomaly info
│   └── Output: metadata columns + 11 raw columns per split
│
├── Step 5: Temporal Fold Metadata
│   ├── Expanding window (default) or rolling window
│   ├── Configurable embargo gap, min train, test horizon
│   └── Save: temporal_folds.json
│
└── Step 6: Export
    ├── train.parquet, val.parquet, test.parquet (raw data)
    ├── split_metadata.json
    ├── temporal_folds.json
    ├── feature_columns.json
    └── pipeline_report.json

=== Next: Run feature_engineering.py ===

scripts/feature_engineering.py (see docs/5_feature-engineering/)
│
├── Step 1: Load preprocessed data from datasets/processed/
├── Step 2: Compute 17 derived features
├── Step 3: Impute + cap outliers (fit on train only)
├── Step 4: Cyclical encoding (month → sin/cos)
├── Step 5: Interaction features
├── Step 6: Drop redundant features
├── Step 7: Feature selection (mutual info / ANOVA)
├── Step 8: PCA dimensionality reduction (optional)
└── Step 9: Export to datasets/engineered/
```

## CLI Usage

```bash
python scripts/preprocessor.py \
  --input datasets/unprocessed/puf.parquet \
  --output datasets/processed/ \
  --personas-per-archetype 1000 \
  --months 12 \
  --seed 42 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --embargo-months 1 \
  --min-train-months 6 \
  --test-horizon-days 30 \
  --wfv-strategy expanding \
  --strict
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `datasets/unprocessed/puf.parquet` | Path to FIES data file (CSV or Parquet) |
| `--output` | `datasets/processed/` | Output directory for processed data |
| `--personas-per-archetype` | `1000` | Number of personas to generate per archetype |
| `--months` | `12` | Number of months of transaction history to generate |
| `--seed` | `42` | Random seed (applies to both synthesis and splitting) |
| `--skip-fies` | `False` | Skip FIES data loading (use default statistics) |
| `--train-ratio` | `0.70` | Fraction of personas for training |
| `--val-ratio` | `0.15` | Fraction of personas for validation |
| `--test-ratio` | `0.15` | Fraction of personas for testing |
| `--embargo-months` | `1` | Months to skip between train and test (prevents leakage) |
| `--min-train-months` | `6` | Minimum months of training data per fold |
| `--test-horizon-days` | `30` | Test window size in days (converted to months) |
| `--wfv-strategy` | `expanding` | Walk-forward strategy: `expanding` or `rolling` |
| `--strict` | `False` | Raise errors on warnings |

## Output Files

### `train.parquet`, `val.parquet`, `test.parquet`

Raw data matrices with one row per persona-month. Schema includes metadata and raw FIES columns:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | str | Persona identifier |
| `month` | int | Month number (1-12) |
| `fbp_label` | str | Personal Financial Profile class |
| `is_anomalous` | bool | Whether this month contains anomalous transactions |
| `anomaly_type` | str | Type of anomaly |
| `total_income` | float | Cumulative income |
| `total_expenses` | float | Cumulative expenses |
| `food_expense` | float | Cumulative food expenses |
| `housing_expense` | float | Cumulative housing expenses |
| `transport_expense` | float | Cumulative transport expenses |
| `health_expense` | float | Cumulative health expenses |
| `education_expense` | float | Cumulative education expenses |
| `other_expense` | float | Cumulative other expenses |
| `savings` | float | Cumulative savings |
| `debt_payment` | float | Cumulative debt payment |
| `transaction_count` | int | Total transaction count |

> Run `scripts/feature_engineering.py` next to compute derived features (income stability, ratios, trends, etc.)

### `split_metadata.json`

```json
{
  "seed": 42,
  "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
  "personas": {
    "train": ["persona_A1_0001", ...],
    "val": ["persona_B1_0001", ...],
    "test": ["persona_C1_0001", ...]
  },
  "distributions": {
    "train": {"Stable/Obligated": 4900, "Stable/Flexible": 2800, ...},
    "val": {"Stable/Obligated": 1050, ...},
    "test": {"Stable/Obligated": 1050, ...}
  },
  "embargo_months": 1,
  "min_train_months": 6,
  "test_horizon_days": 30,
  "wfv_strategy": "expanding"
}
```

### `temporal_folds.json`

Walk-forward validation fold definitions:

```json
[
  {
    "fold": 1,
    "train_months": [1, 2, 3, 4, 5, 6],
    "embargo_months": [7],
    "test_months": [8]
  },
  {
    "fold": 2,
    "train_months": [1, 2, 3, 4, 5, 6, 7],
    "embargo_months": [8],
    "test_months": [9]
  },
  ...
]
```

### `feature_columns.json`

Feature metadata including names, types, and column lists.

### `pipeline_report.json`

Quality report with validation warnings, split sizes, and persona counts.

## Feature Engineering

Feature engineering is handled by the separate **Phase 5** pipeline (`scripts/feature_engineering.py`).
See the full documentation at `docs/5_feature-engineering/feature-engineering.md`.

The feature engineering pipeline computes:
- **17 derived features** — income stability, obligation ratio, savings rate, trends, entropy, etc.
- **Cyclical encoding** — sin/cos transformation for month (period=12)
- **Interaction features** — income×volatility, obligation×volatility
- **Feature selection** — mutual information or ANOVA F-test (SelectKBest)
- **PCA dimensionality reduction** — configurable variance threshold or component count

All feature engineering follows strict leakage prevention: fit on train only, transform val/test.

## Leakage Prevention

The pipeline implements 3 layers of leakage prevention:

| # | Check | Implementation |
|---|-------|---------------|
| 1 | Persona-level split | Entire persona goes to one split (no mixing) |
| 2 | Embargo gap | 1-month gap between train and test windows |
| 3 | Split-before-export | All splitting completes before any feature engineering |

Feature engineering adds additional leakage prevention (see `docs/5_feature-engineering/feature-engineering.md`).

## Walk-Forward Validation

The temporal fold metadata enables rolling-origin evaluation across all calendar months, avoiding the bias of a fixed test window.

### Expanding Window (default)

Training window grows with each fold:

```
Fold 1: Train [1-6],  Embargo [7],  Test [8]
Fold 2: Train [1-7],  Embargo [8],  Test [9]
Fold 3: Train [1-8],  Embargo [9],  Test [10]
Fold 4: Train [1-9],  Embargo [10], Test [11]
Fold 5: Train [1-10], Embargo [11], Test [12]
```

Test months: 8, 9, 10, 11, 12 — covers Aug through Dec, not just holidays.

### Rolling Window

Fixed-size training window:

```
Fold 1: Train [1-6],  Embargo [7],  Test [8]
Fold 2: Train [2-7],  Embargo [8],  Test [9]
Fold 3: Train [3-8],  Embargo [9],  Test [10]
Fold 4: Train [4-9],  Embargo [10], Test [11]
Fold 5: Train [5-10], Embargo [11], Test [12]
```

### Embargo Gap

The embargo gap prevents feature leakage where training samples at the end of the train window might use data from the embargo period. This follows Lopez de Prado's *Advances in Financial Machine Learning* (Ch. 7).

For monthly features on daily data, the embargo should equal the feature aggregation window (1 month = 31 days).

## Input Data Format

The preprocessor generates these files internally during the synthesis step (step 1). They are written to `synth/` and consumed by the subsequent preprocessing steps.

### `monthly_summaries.parquet`

Generated by `generate_transactions.py` (called by `preprocessor.py`):

| Column | Type | Description |
|--------|------|-------------|
| `persona_id` | str | Persona identifier |
| `month` | int | Month number (1-12) |
| `year` | int | Year (2023) |
| `total_income` | float | Total income for the month |
| `total_expenses` | float | Total expenses for the month |
| `food_expense` | float | Food category total |
| `housing_expense` | float | Housing category total |
| `transport_expense` | float | Transport category total |
| `health_expense` | float | Health category total |
| `education_expense` | float | Education category total |
| `other_expense` | float | Other category total |
| `savings` | float | max(0, income - expenses) |
| `debt_payment` | float | 10% of total expenses |
| `balance` | float | Running balance |
| `transaction_count` | int | Number of transactions |
| `income_stability_cv` | float | Income coefficient of variation |
| `obligation_ratio` | float | Essential expense ratio |
| `is_anomalous` | bool | Contains anomalous transactions |

### `personas.parquet`

Generated by `generate_personas.py` (called by `preprocessor.py`):

| Column | Type | Description |
|--------|------|-------------|
| `persona_id` | str | Unique identifier |
| `archetype_id` | str | Archetype code (A-L) |
| `archetype_name` | str | Human-readable archetype name |
| `monthly_income` | float | Base monthly income (PHP) |
| `income_cv` | float | Income coefficient of variation |
| `obligation_ratio` | float | Target obligation ratio |
| `savings_rate` | float | Target savings rate |
| `household_size` | int | Number of household members |
| `employment_type` | str | Employment category |
| `fbp_label` | str | PFP classification |
| `food_expense` | float | Monthly food budget |
| `housing_expense` | float | Monthly housing budget |
| `transport_expense` | float | Monthly transport budget |
| `health_expense` | float | Monthly health budget |
| `education_expense` | float | Monthly education budget |
| `other_expense` | float | Monthly other budget |
| `debt_amount` | float | Outstanding debt |
| `savings_amount` | float | Current savings |
| `emergency_fund` | float | Emergency fund |

## Dependencies

```
pandas
numpy
scikit-learn
pyarrow
```

No new dependencies required — all are already in `requirements.txt`.

## Testing

Run the full pipeline with a small dataset:

```bash
# Run preprocessing (synthesis + splitting + raw data export)
python scripts/preprocessor.py \
  --input datasets/unprocessed/puf.parquet \
  --output datasets/processed/ \
  --personas-per-archetype 2 \
  --seed 42

# Verify outputs
python -c "
import pandas as pd
for split in ['train', 'val', 'test']:
    df = pd.read_parquet(f'datasets/processed/{split}.parquet')
    print(f'{split}: {df.shape[0]} rows, {df[\"user_id\"].nunique()} personas')
"
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Separate feature engineering | Moved to Phase 5 (`feature_engineering.py`) | Separation of concerns, independent development |
| Split before features | No modification on unsplit data | Prevents any form of data leakage |
| Raw-only export | Preprocessed data = metadata + raw columns | Feature engineering reads this as input |
| Expanding window | Growing training set | Maximizes data for short 12-month series |
| Embargo gap | 1 month between train and test | Prevents autocorrelation leakage |
| Parquet output | Columnar, compressed, fast I/O | Better performance than CSV for large datasets |
| Configurable seed | Default 42, CLI arg allows tuning | Reproducibility with flexibility |
