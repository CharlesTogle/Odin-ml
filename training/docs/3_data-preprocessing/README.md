# Phase 3: Data Preprocessing

## What This Phase Is

This phase transforms raw synthetic transaction data into labeled, split feature matrices ready for feature engineering. It enforces strict data leakage prevention through persona-level splitting and incremental feature computation. Feature engineering is handled separately in Phase 5 (`feature_engineering.py`).

## Inputs

| Input | Location | Description |
|-------|----------|-------------|
| FIES data (CSV or Parquet) | `datasets/unprocessed/` | Raw FIES data collected by `collector.py` |

The preprocessor runs synthesis internally as step 1, generating personas and transactions into `synth/` before proceeding with splitting.

## Process

### 8-Step Pipeline

```
Step 1: Synthesis (persona + transaction generation → synth/)
    ↓
Step 2: Load & Validate
    ↓
Step 3: Validate Data Quality
    ↓
Step 4: Persona-Level Split (stratified by PFP label)
    ↓
Step 5: Build Raw Feature Matrices (per split)
    ↓
Step 6: Temporal Fold Metadata
    ↓
Step 7: Export
```

### Step 1: Synthesis

- Load FIES data from the `--input` path (CSV or Parquet)
- Generate 12 archetypes × 1,000 personas = 12,000 synthetic personas
- Generate 12-month transaction histories for each persona
- Export to `synth/` directory (personas, transactions, monthly summaries)

### Step 2: Load & Validate

- Load `monthly_summaries.parquet`, `personas.parquet`, and anomaly info from `transactions.parquet`
- Validate schema, data types, completeness
- Check for required columns: `persona_id`, `month`, `total_income`, `total_expenses`, `transaction_count`

### Step 3: Validate Data Quality

- Check persona counts, month counts, row completeness
- Flag zero-income rows and personas with insufficient data

### Step 4: Persona-Level Split

- **Split ratios**: 70% train / 15% val / 15% test (configurable)
- **Method**: `StratifiedShuffleSplit` to preserve PFP class distribution
- **Critical rule**: Entire persona goes to one split (no persona-level mixing)
- **Output**: `split_metadata.json` with persona assignments

### Step 5: Build Raw Feature Matrices

For each split (train, val, test):
1. Filter summaries to split's persona IDs
2. Merge with persona labels (`pfp_label`)
3. Merge with anomaly info (`is_anomalous`, `anomaly_type`)
4. Rename `persona_id` → `user_id`
5. Select metadata + raw columns only

**5 Metadata Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | str | Persona identifier |
| `month` | int | Month number (1-12) |
| `pfp_label` | str | Personal Financial Profile class |
| `is_anomalous` | bool | Whether this month contains anomalous transactions |
| `anomaly_type` | str | Type of anomaly |

**11 Raw Cumulative Features:**

| Column | Type | Description |
|--------|------|-------------|
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

Total output schema: `user_id, month, pfp_label, is_anomalous, anomaly_type, 11 raw columns` = 16 columns.

> Run `scripts/feature_engineering.py` next to compute derived features (income stability, ratios, trends, etc.)

### Step 6: Temporal Fold Metadata

- **Strategy**: Expanding window (default) or rolling window
- **Configuration**: 1-month embargo gap, min 6 train months, 30-day test horizon
- **Folds**: 5 folds covering months 8-12, each with embargo gap to prevent leakage
- **Output**: `temporal_folds.json` with fold boundaries

### Step 7: Export

Generate all output files with metadata and diagnostics.

## Outputs

| Output | Description |
|--------|-------------|
| `train.parquet` | Training raw feature matrix |
| `val.parquet` | Validation raw feature matrix |
| `test.parquet` | Test raw feature matrix |
| `split_metadata.json` | Persona-to-split assignments |
| `temporal_folds.json` | Fold boundaries for WFV |
| `feature_columns.json` | Feature names and types |
| `pipeline_report.json` | Execution summary and diagnostics |

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

Minimum usage:

```bash
python scripts/preprocessor.py --input datasets/unprocessed/puf.parquet
```

## Key Design Decisions

### Why Persona-Level Splitting?

If we split by rows instead of personas, the same persona could appear in both train and test sets with different months. This creates data leakage because the model could learn persona-specific patterns that don't generalize.

### Why Raw-Only Export?

The preprocessor intentionally outputs only raw cumulative features (metadata + 11 raw columns). Derived features, normalization, encoding, and feature selection are handled by the separate `feature_engineering.py` pipeline. This separation:
- Allows independent development and testing of each stage
- Enables EDA on raw data before engineering decisions are made
- Prevents premature normalization that could leak test statistics

## Integration

```
datasets/processed/ (Phase 3 output)
    ↓ scripts/feature_engineering.py
datasets/engineered/ (Phase 5 output)
    ↓ scripts/eda.py (or directly from processed/)
figures/eda_report.md, figures/*.png
    ↓
Phase 7: Model Training (pending)
```
