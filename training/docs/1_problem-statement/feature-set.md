# Final Feature Set

**Document Version:** v1.0
**Author(s):** Guevarra
**Date:** 2026-07-17
**Status:** Draft
**Purpose:** Define the complete feature set for all Odin models, derived from RRL benchmarks and FIES data

---

## 1. Overview

This document defines the final feature set for all three Odin models:

1. **PFP Classifier** — 8-class financial behavior classification (3 binary dimensions)
2. **Forecaster** — Monthly spending prediction
3. **Anomaly Detector** — Unusual transaction detection

Features are derived from:
- **FIES 2023 data dictionary** (~90 variables, 405 columns)
- **RRL benchmarked algorithms** (XGBoost, LightGBM, LSTM, etc.)
- **Domain expertise** (financial behavior patterns)

---

## 2. Feature Categories

### 2.1 Raw Features (from FIES)

These are the original FIES variables that serve as inputs:

| Feature | FIES Code | Type | Description |
|---------|-----------|------|-------------|
| Household Income | TFI | Float | Total family income (monthly) |
| Per Capita Income | TPCI | Float | Income per household member |
| Food Expenditure | TFOOD | Float | Monthly food spending |
| Housing Expenditure | THOUSING | Float | Monthly housing costs |
| Transport Expenditure | TTRANS | Float | Monthly transportation costs |
| Health Expenditure | THEALTH | Float | Monthly health costs |
| Education Expenditure | TEDUC | Float | Monthly education costs |
| Total Expenditure | TEXP | Float | Sum of all expenses |
| Savings | TSAVINGS | Float | Monthly savings amount |
| Debt | TDEBT | Float | Outstanding debt |
| Household Size | HHSIZE | Int | Number of household members |
| Employment Status | EMPSTAT | Categorical | Employment category |
| Occupation | OCCUP | Categorical | Primary occupation |
| Region | REGION | Categorical | Geographic region |
| Urban/Rural | URBAN | Binary | Urban vs rural residence |

### 2.2 Derived Features (Computed)

These are engineered features computed from raw FIES data:

| Feature | Formula | Module | Description |
|---------|---------|--------|-------------|
| Income Stability (CV) | σ(income) / μ(income) | PFP | Coefficient of variation |
| Obligation Ratio | (Essential + Obligatory) / Total | PFP | Fixed cost proportion |
| Savings Rate | Savings / Income | All | Financial health indicator |
| Debt-to-Income | Debt / Income | All | Debt burden |
| Essential Ratio | Essential Expenses / Total | PFP | Non-discretionary spending |
| Discretionary Ratio | Discretionary / Total | PFP | Optional spending |
| Income Trend | Linear slope of income | PFP, LSTM | Growth/decline pattern |
| Expense Trend | Linear slope of expenses | LSTM | Spending trajectory |
| Volatility Index | Rolling std of expenses | Anomaly | Spending stability |
| Seasonality Score | Fourier analysis of expenses | LSTM | Seasonal patterns |
| Category Entropy | Shannon entropy of categories | Anomaly | Spending diversity |
| Transaction Frequency | Count / month | Anomaly | Activity level |
| Average Transaction Size | Mean transaction amount | Anomaly | Spending magnitude |
| Income Regularity | % months with income > 0 | PFP | Income consistency |
| Expense Regularity | % months with expenses > 0 | PFP | Spending consistency |
| Financial Trajectory | Linear slope of income-expense gap | PFP (overlay) | Trend of surplus/deficit over time (Article IV §5) |
| Financial Margin | (Income − Expenses) / Income | PFP (overlay) | Point-in-time surplus/deficit level (Article IV §6) |

### 2.3 Interaction Features

| Feature | Formula | Module | Description |
|---------|---------|--------|-------------|
| Income-Expense Gap | Income - Expenses | All | Surplus/deficit |
| Essential-Income Ratio | Essential / Income | PFP | Basic needs burden |
| Savings-Income Ratio | Savings / Income | All | Savings capacity |
| Debt-Income Ratio | Debt / Income | All | Debt burden |
| Income-volatility Interaction | CV * Income | PFP | Volatility impact |
| Obligation-volatility Interaction | Ratio * CV | PFP | Fixed cost volatility |

---

## 3. Feature Sets per Module

### 3.1 PFP Classifier Feature Set

**Primary Features (Tier 1 - Rule-Based):**
1. Income Stability (CV)
2. Obligation Ratio

**Extended Features (Tier 2-4 - Learned):**
3. Savings Rate
4. Debt-to-Income
5. Essential Ratio
6. Income Trend
7. Expense Trend
8. Income Regularity
9. Expense Regularity
10. Income-Expense Gap
11. Essential-Income Ratio
12. Savings-Income Ratio

**Overlay Indicator Features (do not affect PFP class):**
13. Financial Trajectory (slope of income-expense gap over time)
14. Financial Margin (point-in-time (Income − Expenses) / Income)

**Total:** 12 classifying features + 2 overlay features

### 3.2 Forecaster Feature Set

**Input Features (LSTM):**
1. Monthly Income (normalized)
2. Monthly Expenses (normalized)
3. Expense Categories (food, housing, transport, health, education)
4. Savings Amount
5. Debt Amount
6. Transaction Count
7. Day of Month (for intra-month patterns)
8. Income Stability (CV, rolling)
9. Obligation Ratio (rolling)
10. Savings Rate (rolling)

**Output Targets:**
1. Next Month Total Expenses
2. Next Month Category Breakdown (5 categories)
3. Next Month Income (if variable)

**Total:** 10 input features, 7 output targets

### 3.3 Anomaly Detector Feature Set

**Baseline Features:**
1. Mean Income (rolling)
2. Std Income (rolling)
3. Mean Expenses (rolling)
4. Std Expenses (rolling)
5. Category Distributions (5 categories)
6. Transaction Frequency (rolling)
7. Average Transaction Size (rolling)

**Detection Features:**
1. Current Income (vs baseline)
2. Current Expenses (vs baseline)
3. Category Deviation (5 categories)
4. Transaction Frequency Deviation
5. Amount Deviation
6. Category Entropy
7. Volatility Index

**Total:** 14 features (7 baseline + 7 detection)

---

## 4. Feature Engineering Pipeline

### 4.1 Normalization

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler for Gaussian-like features
standard_features = ['income_stability_cv', 'obligation_ratio', 'savings_rate',
                     'debt_to_income', 'essential_ratio', 'discretionary_ratio']

# MinMaxScaler for bounded features
minmax_features = ['income_regularity', 'expense_regularity', 'category_entropy']

scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

def normalize_features(X, feature_names):
    X_normalized = X.copy()
    for i, name in enumerate(feature_names):
        if name in standard_features:
            X_normalized[:, i] = scaler_standard.fit_transform(X[:, i:i+1])
        elif name in minmax_features:
            X_normalized[:, i] = scaler_minmax.fit_transform(X[:, i:i+1])
    return X_normalized
```

### 4.2 Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def select_features(X, y, k=10, method='mutual_info'):
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == 'anova':
        selector = SelectKBest(f_classif, k=k)
    
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    
    return X_selected, selected_indices

def feature_importance(X, y, model='xgboost'):
    from xgboost import XGBClassifier
    
    model = XGBClassifier()
    model.fit(X, y)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    return indices, importances
```

### 4.3 Feature Scaling for LSTM

```python
def prepare_lstm_features(transactions, seq_length=6):
    """Prepare features for LSTM input."""
    features = []
    for t in transactions:
        features.append([
            t.total_income,
            t.total_expenses,
            t.food_expense,
            t.housing_expense,
            t.transport_expense,
            t.health_expense,
            t.education_expense,
            t.savings,
            t.debt,
            t.transaction_count,
        ])
    
    # Normalize per feature
    features = np.array(features)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Reshape for LSTM: [samples, timesteps, features]
    X = features_scaled.reshape(1, seq_length, -1)
    
    return X, scaler
```

---

## 5. Feature Importance (from RRL Benchmarks)

### 5.1 XGBoost Classification (PFP)

Based on RRL Topic 5.C.VII:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Income Stability (CV) | 0.28 |
| 2 | Obligation Ratio | 0.24 |
| 3 | Savings Rate | 0.15 |
| 4 | Debt-to-Income | 0.12 |
| 5 | Essential Ratio | 0.08 |
| 6 | Income Trend | 0.05 |
| 7 | Expense Trend | 0.04 |
| 8 | Income Regularity | 0.02 |
| 9 | Expense Regularity | 0.01 |
| 10 | Income-Expense Gap | 0.01 |

### 5.2 LSTM Forecasting

Based on RRL Topic 7.B.I:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Monthly Expenses (lag-1) | 0.32 |
| 2 | Monthly Expenses (lag-2) | 0.18 |
| 3 | Monthly Income | 0.15 |
| 4 | Food Expenses | 0.12 |
| 5 | Housing Expenses | 0.08 |
| 6 | Savings Amount | 0.05 |
| 7 | Debt Amount | 0.04 |
| 8 | Transaction Count | 0.03 |
| 9 | Transport Expenses | 0.02 |
| 10 | Health Expenses | 0.01 |

### 5.3 Isolation Forest (Anomaly Detection)

Based on RRL Topic 12.B.II:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Transaction Amount (vs baseline) | 0.35 |
| 2 | Category Deviation | 0.22 |
| 3 | Transaction Frequency | 0.18 |
| 4 | Income Deviation | 0.12 |
| 5 | Expense Deviation | 0.08 |
| 6 | Category Entropy | 0.03 |
| 7 | Volatility Index | 0.02 |

---

## 6. Feature Validation Rules

### 6.1 Missing Values

```python
def handle_missing_features(X, feature_names):
    """Handle missing feature values."""
    X_clean = X.copy()
    
    for i, name in enumerate(feature_names):
        if name in ['income_stability_cv', 'obligation_ratio']:
            # Critical features: impute with median
            median_val = np.nanmedian(X_clean[:, i])
            X_clean[np.isnan(X_clean[:, i]), i] = median_val
        else:
            # Other features: impute with 0
            X_clean[np.isnan(X_clean[:, i]), i] = 0
    
    return X_clean
```

### 6.2 Outlier Handling

```python
def cap_outliers(X, feature_names, percentile=99):
    """Cap outliers at specified percentile."""
    X_capped = X.copy()
    
    for i, name in enumerate(feature_names):
        if name in ['debt_to_income', 'savings_rate']:
            # Cap at percentile
            cap_value = np.percentile(X_capped[:, i], percentile)
            X_capped[X_capped[:, i] > cap_value, i] = cap_value
    
    return X_capped
```

### 6.3 Feature Validation

```python
def validate_features(X, feature_names):
    """Validate feature values are within expected ranges."""
    ranges = {
        'income_stability_cv': (0, 5),
        'obligation_ratio': (0, 1),
        'savings_rate': (-1, 1),
        'debt_to_income': (0, 10),
        'essential_ratio': (0, 1),
        'discretionary_ratio': (0, 1),
        'income_regularity': (0, 1),
        'expense_regularity': (0, 1),
        'category_entropy': (0, 2),
    }
    
    violations = []
    for i, name in enumerate(feature_names):
        if name in ranges:
            min_val, max_val = ranges[name]
            if np.any(X[:, i] < min_val) or np.any(X[:, i] > max_val):
                violations.append(name)
    
    return violations
```

---

## 7. Expected Outputs

| Output | Description | Location |
|--------|-------------|----------|
| `feature_set.md` | This document | `1_problem-statement/` |
| `feature_importance.csv` | Importance scores per feature | `1_problem-statement/` |
| `feature_ranges.json` | Valid ranges per feature | `1_problem-statement/` |
| `normalization_params.json` | Scaler parameters | `1_problem-statement/` |

---

## 8. RRL Justifications

| Concept | RRL Support | Topic |
|---------|-------------|-------|
| Feature engineering | Standard ML practice | 5.C.VII |
| Mutual information feature selection | Non-linear feature importance | 8.B.V |
| XGBoost feature importance | Tree-based importance | 5.C.VII |
| LSTM input features | Sequential data representation | 7.B.I |
| Isolation Forest features | Anomaly detection features | 12.B.II |

---

*Document created: 2026-07-17*
