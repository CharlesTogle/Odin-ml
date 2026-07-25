# Phase 4: Exploratory Data Analysis

## What This Phase Is

This phase analyzes the processed feature matrices to understand data distributions, class balance, correlations, temporal patterns, anomalies, and data quality issues before model training. The EDA script is designed to work with both raw data (`datasets/processed/`) and engineered data (`datasets/engineered/`), dynamically detecting which features are available.

## Inputs

| Input | Location | Description |
|-------|----------|-------------|
| `train.parquet` | `datasets/processed/` or `datasets/engineered/` | Training feature matrix |
| `val.parquet` | `datasets/processed/` or `datasets/engineered/` | Validation feature matrix |
| `test.parquet` | `datasets/processed/` or `datasets/engineered/` | Test feature matrix |
| `pipeline_report.json` | input directory | Feature statistics and range violations |

## Process

### 7-Step Analysis

| Step | Analysis | Key Findings |
|------|----------|--------------|
| 1 | Dataset Overview | Row counts, persona counts, column counts, memory usage |
| 2 | Feature Distributions | Histograms, skewness, kurtosis for available features |
| 3 | Class Balance | PFP class distribution (21-29% per class) |
| 4 | Correlation Analysis | Pearson/Spearman matrices, multicollinearity |
| 5 | Temporal Patterns | Monthly trajectories, month-1 degeneracy |
| 6 | Anomaly Analysis | Anomaly rate, type breakdown, feature comparisons |
| 7 | Data Quality | Zero-income rows, outliers, range violations |

### Analysis Scope

The EDA dynamically detects which features are present in the input data:

- **When run on `datasets/processed/`** (raw data): 11 raw features + 5 metadata columns = 16 columns. Engineered feature sections are skipped gracefully.
- **When run on `datasets/engineered/`** (engineered data): 17 engineered features + 11 raw features + 5 metadata columns = 33 columns. Full analysis including engineered feature distributions, correlations, and temporal patterns.

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| `eda_report.md` | output directory | Full markdown report with all 7 sections |
| `*.png` | output directory | Static plots (regenerate with script) |

### Generated Plots

Plots are generated for features that exist in the data. When run on raw data only, engineered-feature plots are skipped.

**Always generated (raw + engineered):**
| Plot | Description |
|------|-------------|
| `class_distribution.png` | PFP class counts per split |
| `class_balance_by_split.png` | Class proportion comparison |
| `correlation_pearson.png` | Pearson heatmap |
| `correlation_spearman.png` | Spearman heatmap |
| `temporal_income_expense_by_class.png` | Income vs expense by class |
| `anomaly_rate_by_month.png` | Anomaly rate across months |
| `anomaly_rate_by_class.png` | Anomaly rate by PFP class |
| `anomaly_type_breakdown.png` | Anomaly type bar chart |
| `feature_distributions_raw.png` | Raw feature histograms |

**Only with engineered features:**
| Plot | Description |
|------|-------------|
| `feature_distributions_engineered.png` | Engineered feature histograms |
| `feature_by_class_boxplots.png` | Top features by PFP class |
| `correlation_with_target.png` | Point-biserial with PFP classes |
| `temporal_feature_means.png` | Monthly trajectories |
| `anomaly_feature_comparison.png` | Normal vs anomalous distributions |
| `zero_income_analysis.png` | Impact of zero-income rows |
| `outlier_summary.png` | IQR outlier counts |

## CLI Usage

```bash
python scripts/eda.py \
  --input datasets/processed/ \
  --output figures/ \
  --seed 42
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `datasets/processed/` | Directory with train.parquet, val.parquet, test.parquet |
| `--output` | `figures/` | Output directory for plots |
| `--seed` | `42` | Random seed for reproducibility |

## Dependencies

Added to `requirements.txt`:
- `matplotlib>=3.8.0` — Static plot generation
- `seaborn>=0.13.0` — Statistical visualization
- `scipy>=1.10.0` — Hypothesis testing (Kruskal-Wallis, Mann-Whitney, Chi-square)
- `scikit-learn>=1.3.0` — Feature importance (Random Forest, permutation importance, MI, ANOVA)

## Key Findings Summary

1. **Class balance is good:** 21-29% per class across all splits (imbalance ratio 1.34)
2. **Zero-income rows (~15%):** Expected behavior — intentional zero-income months from irregular/project-based income patterns
3. **Anomaly rate (~3%):** Well-distributed across months and classes
4. **Multicollinearity:** Strong correlations among raw expense categories (expected — all derive from total_expenses)
5. **No missing values:** Training set is complete
6. **Hypothesis testing:** No significant association between PFP class and anomaly status (chi-square p=0.71). Most features differ significantly across PFP classes (Kruskal-Wallis p<0.05)
7. **Top predictive features:** `income_stability_cv`, `savings_rate`, `essential_income_ratio` ranked top-5 by all 4 importance methods
8. **Low-importance features:** `expense_trend`, `expense_regularity`, `category_entropy` have near-zero importance — candidates for removal

## Pipeline Integration

```
datasets/processed/ (Phase 3 output)
    ↓ scripts/eda.py
figures/eda_report.md, figures/*.png
    ↓ (inform feature selection, preprocessing improvements)
Phase 4.5: Dimension & Threshold Discovery → dimension-threshold-candidates.md
Phase 5: Feature Engineering → datasets/engineered/
Phase 7: Model Training (pending)
```
