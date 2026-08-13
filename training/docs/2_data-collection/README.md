# Phase 2: Data Collection

## What This Phase Is

This phase acquires, documents, and organizes all source data needed for model development. The data primarily comes from the Philippine Statistics Authority (PSA) and is used to build the synthetic persona pipeline that trains all three models.

## Inputs

| Input | Source | Location |
|-------|--------|----------|
| FIES 2023 CSV | PSA | `datasets/raw/Family Income and Expenditure.csv` |
| FIES Dictionary | PSA | `docs/2_data-collection/FIES Dictionary & Valueset.csv` |

> **Note**: Raw data files are not committed to Git (too large). Download from PSA/BSP and place in `datasets/raw/` before running the pipeline.

## Process

### 1. Data Acquisition

- **FIES 2023**: Downloaded from PSA OpenStat portal. Contains household income/expense data for 56,000+ households nationwide.
- **NCR Subset**: Extracted from FIES 2023 for National Capital Region (NCR) only — the target deployment area.
- **BSP Data**: Banco Sentral ng Pilipinas data on Filipino financial behavior in Metro Manila.

### 2. Column Mapping (`fies_columns.py`)

The FIES CSV uses variable IDs (e.g., `TOINC`, `FOOD`, `HEALTH`) rather than human-readable names. The mapping script:

- Maps 56 of 115 expected FIES columns to readable names
- Provides safe accessor functions for column lookup
- Handles missing columns gracefully (returns defaults)

Key mapped columns:

| FIES ID | Description | Used For |
|---------|-------------|----------|
| `TOINC` | Total Income | Income stability calculation |
| `FOOD` | Food Expenditure | Essential expense ratio |
| `HEALTH` | Health Expenditure | Essential expense ratio |
| `EDUC` | Education Expenditure | Essential expense ratio |
| `TRANS` | Transportation | Discretionary spending |
| `RENTVAL` | Rental Value | Housing cost proxy |

### 2.5 Persona & Transaction Synthesis (`preprocessor.py`)

After archetype definitions are finalized, the preprocessor runs synthesis as its first step:

1. **Persona generation** (`generate_personas.py`): Samples 1,000 personas per archetype from FIES NCR decile ranges, producing persona metadata (income, expenses, PFP label, employment type).
2. **Transaction generation** (`generate_transactions.py`): For each persona, generates 12 months of daily transaction history with income patterns, expense allocation, temporal modifiers, and anomaly injection.
3. **Preprocessing** (`preprocessor.py`): Transforms raw transactions into labeled, normalized feature matrices with persona-level splitting.

This step bridges Phase 2 (data collection) and Phase 3 (preprocessing) — it is the primary data production mechanism for all three models.

> **Note**: The standalone `synthesizer.py` script is deprecated. All synthesis logic is now handled by `preprocessor.py`.

### 3. Archetype Definition

12 financial archetypes defined based on FIES NCR aggregates (3 binary dimensions: Stability × Obligation × Tolerance):

| Archetype | Stability | Obligation | Tolerance | Description |
|-----------|-----------|------------|-----------|-------------|
| A | Stable | Obligated | Tolerant | Steady income, high debt, healthy runway |
| B | Stable | Obligated | At-Risk | Steady income, high debt, limited runway |
| C | Stable | Flexible | Tolerant | Steady income, low debt, healthy runway |
| D | Stable | Flexible | At-Risk | Steady income, low debt, limited runway |
| E | Variable | Obligated | Tolerant | Irregular income, high debt, healthy runway |
| F | Variable | Obligated | At-Risk | Irregular income, high debt, limited runway |
| G | Variable | Flexible | Tolerant | Irregular income, low debt, healthy runway |
| H | Variable | Flexible | At-Risk | Irregular income, low debt, limited runway |
| I–L | ... | ... | ... | Additional archetypes for edge cases (self-employed, seasonal, etc.) |

> **Note:** Income Stability is derived from income *consistency* (coefficient of variation), not income *magnitude* (level). High/low income levels are a separate attribute that does not determine Stability. The earlier draft's use of "Income Level: High/Low" as a proxy for Stability was a design bug — magnitude and consistency are different things.

Each archetype is parameterized with:
- Income mean and standard deviation
- Expense category distributions
- Obligation ratios
- Volatility parameters

### 4. SME Validation (Pending)

The archetype list and classification thresholds need validation from a financial science professor before production use.

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| `FIES Dictionary & Valueset.csv` | `docs/2_data-collection/` | Variable ID mapping (93 lines) |
| `persona-validation-list.md` | `1_problem-statement/` | 12 archetypes for SME review |
| `synthetic-injection-rules.md` | `1_problem-statement/` | 20 rules for FIES→Persona→Transaction |
| `fies_columns.py` | `scripts/` | Python module for FIES column access |

## Data Quality Notes

- **Missing columns**: 59 of 115 FIES variables not mapped (not needed for current models)
- **Anonymization**: FIES data is anonymized per RA 10173 (Data Privacy Act)
- **Geographic scope**: NCR subset only (Metro Manila)
- **Temporal scope**: 2023 cross-sectional data (not longitudinal)

## Pipeline Integration

The data collected in this phase feeds directly into preprocessing:

```
datasets/raw/Family Income and Expenditure.csv
    ↓ scripts/collector.py
datasets/unprocessed/puf.parquet
    ↓ scripts/preprocessor.py (synthesis + preprocessing)
synth/ → datasets/processed/
```

## Legal Constraints

Per PSA Terms & Conditions:
- Raw FIES data cannot be modified or redistributed
- Derivative works (synthetic personas) are permitted for research
- No personally identifiable information in the source data
