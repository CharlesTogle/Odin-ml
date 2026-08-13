# Partial Window Split Methodology

**Document Version:** v1.0
**Author(s):** Guevarra
**Date:** 2026-07-17
**Status:** Draft
**Purpose:** Define the temporal train/validation/test splitting strategy for FIES persona data

---

## 1. Overview

This document specifies the **Partial Window Split** methodology for dividing synthetic FIES persona data into train/validation/test sets. This approach is critical for:

1. **Preventing temporal leakage** — Future data must not leak into training
2. **Simulating production conditions** — Training on partial windows, deploying on full data
3. **Enabling walk-forward validation** — Rolling window splits for temporal generalization

---

## 2. Temporal Structure of FIES Data

### 2.1 Data Timeline

The FIES 2023 dataset contains survey responses collected across multiple rounds:

| Round | Period | Data Points |
|-------|--------|-------------|
| Round 1 | January-March 2023 | ~1,200 households |
| Round 2 | April-June 2023 | ~1,200 households |
| Round 3 | July-September 2023 | ~1,200 households |
| Round 4 | October-December 2023 | ~1,200 households |

### 2.2 Synthetic Persona Timeline

When we generate synthetic personas from FIES data, we create **12-month transaction histories**:

```
Month 1 (Jan) -> Month 2 (Feb) -> ... -> Month 12 (Dec)
```

Each month contains income transactions, expense transactions, and account balance snapshots.

---

## 3. Standard Split: 60/20/20

The standard split uses **temporal ordering** (not random shuffling):

```
[---------- Train (60%) ----------][-- Val (20%) --][-- Test (20%) --]
Month 1                        Month 7          Month 10         Month 12
```

### 3.1 Implementation

```python
def temporal_split(persona, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    n_months = len(persona.monthly_transactions)
    train_end = int(n_months * train_ratio)
    val_end = int(n_months * (train_ratio + val_ratio))
    return (persona.monthly_transactions[:train_end],
            persona.monthly_transactions[train_end:val_end],
            persona.monthly_transactions[val_end:])
```

### 3.2 Minimum Window Sizes

| Split | Minimum Months | Rationale |
|-------|----------------|-----------|
| Train | 6 | Need enough data to learn patterns |
| Val | 2 | Need enough data to evaluate thresholds |
| Test | 2 | Need enough data to assess generalization |
| **Total** | **10** | Minimum for meaningful evaluation |

**For 12-month personas:** Train: 7 months (58%), Val: 3 months (25%), Test: 2 months (17%).

---

## 4. Partial Window Splits for PFP Classification

### 4.1 Feature Computation Window

The PFP classifier computes features from a **3-month observation window**:

```python
def compute_features(transactions, window_size=3):
    window = transactions[-window_size:]
    incomes = [m.total_income for m in window]
    income_cv = np.std(incomes) / np.mean(incomes) if np.mean(incomes) > 0 else 0
    essential = sum(m.essential_expenses for m in window)
    obligatory = sum(m.obligatory_expenses for m in window)
    total = sum(m.total_expenses for m in window)
    obligation_ratio = (essential + obligatory) / total if total > 0 else 0
    return {
        'income_stability_cv': income_cv,
        'obligation_ratio': obligation_ratio,
        'window_size': window_size
    }
```

### 4.2 Training on Partial Windows

During training, we simulate **partial windows** to test robustness:

```python
def train_on_partial_windows(personas, window_sizes=[3, 4, 5, 6]):
    results = {}
    for window_size in window_sizes:
        X_train, y_train = [], []
        for p in personas:
            partial = p.monthly_transactions[-window_size:]
            features = compute_features(partial, window_size)
            X_train.append(features)
            y_train.append(p.label)
        model = train_pfp_classifier(X_train, y_train)
        X_val, y_val = prepare_validation_set(val_personas, window_size=None)
        results[window_size] = model.score(X_val, y_val)
    return results
```

### 4.3 Robustness to Missing Months

```python
def handle_missing_months(transactions, expected_months=12):
    n_months = len(transactions)
    if n_months < 3:
        return None, "Insufficient data (< 3 months)"
    if n_months < 6:
        window = transactions[-n_months:]
        features = compute_features(window, window_size=n_months)
        features['data_quality'] = 'partial'
        return features, f"Partial data ({n_months} months)"
    window = transactions[-3:]
    features = compute_features(window, window_size=3)
    features['data_quality'] = 'full'
    return features, "Full data (3 months)"
```

---

## 5. Partial Window Splits for Forecaster

### 5.1 Sequence Length for LSTM

The LSTM forecaster expects fixed-length sequences. Partial windows are handled via **padding**:

```python
def prepare_forecaster_input(transactions, seq_length=6):
    n_months = len(transactions)
    if n_months >= seq_length:
        sequence = transactions[-seq_length:]
    else:
        padding_needed = seq_length - n_months
        sequence = [ZeroTransaction()] * padding_needed + transactions
    return np.array([month_to_features(m) for m in sequence])
```

### 5.2 Walk-Forward Training

The forecaster uses rolling windows:

```
Split 1: Train [1-6], Test [7]
Split 2: Train [2-7], Test [8]
Split 3: Train [3-8], Test [9]
Split 4: Train [4-9], Test [10]
Split 5: Train [5-10], Test [11]
Split 6: Train [6-11], Test [12]
```

```python
def walk_forward_split(transactions, seq_length=6, horizon=1):
    splits = []
    n_months = len(transactions)
    for i in range(seq_length, n_months - horizon + 1):
        splits.append({
            'train': transactions[i - seq_length:i],
            'test': transactions[i + horizon - 1],
            'split_idx': len(splits)
        })
    return splits
```

---

## 6. Partial Window Splits for Anomaly Detector

### 6.1 Baseline Window

The anomaly detector establishes a baseline from historical data:

```python
def establish_baseline(transactions, baseline_months=6):
    baseline_data = transactions[:baseline_months]
    return {
        'mean_income': np.mean([m.total_income for m in baseline_data]),
        'std_income': np.std([m.total_income for m in baseline_data]),
        'mean_expenses': np.mean([m.total_expenses for m in baseline_data]),
        'std_expenses': np.std([m.total_expenses for m in baseline_data]),
        'typical_categories': compute_category_distributions(baseline_data),
    }
```

### 6.2 Detection Window

Anomaly detection uses a **sliding window**:

```python
def detect_anomalies(transactions, baseline, detection_window=3):
    scores = []
    for i in range(len(transactions)):
        window_start = max(0, i - detection_window + 1)
        window = transactions[window_start:i+1]
        scores.append(compute_anomaly_score(window, baseline))
    return scores
```

---

## 7. Cross-Persona Splits

### 7.1 Stratified Split by Label

Personas are split **stratified by PFP label**:

```python
from sklearn.model_selection import StratifiedShuffleSplit

labels = [p.label for p in personas]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in sss.split(personas, labels):
    train_personas = [personas[i] for i in train_idx]
    test_personas = [personas[i] for i in test_idx]
```

### 7.2 Leave-One-Persona-Out Cross-Validation

For small datasets, use leave-one-out:

```python
def leave_one_out_split(personas):
    splits = []
    for i in range(len(personas)):
        test_persona = personas[i]
        train_personas = personas[:i] + personas[i+1:]
        splits.append({'train': train_personas, 'test': test_persona})
    return splits
```

---

## 8. Leakage Prevention Checklist

| Check | Description |
|-------|-------------|
| Temporal ordering | Train months < Val months < Test months |
| No future data | Test data never appears in training features |
| Persona-level split | Entire persona goes to one split (not split by month) |
| Feature computation | Features use only historical data (no lookahead) |
| Threshold calibration | ROC thresholds computed on training set only |
| Walk-forward | Rolling windows respect temporal order |

---

## 9. Expected Outputs

| Output | Description | Location |
|--------|-------------|----------|
| `train_personas.json` | Training persona IDs and labels | `1_problem-statement/` |
| `val_personas.json` | Validation persona IDs and labels | `1_problem-statement/` |
| `test_personas.json` | Test persona IDs and labels | `1_problem-statement/` |
| `split_statistics.md` | Class distribution per split | `1_problem-statement/` |

---

## 10. RRL Justifications

| Concept | RRL Support | Topic |
|---------|-------------|-------|
| Temporal splitting | Prevents data leakage in time-series data | 12.B.I |
| Walk-forward validation | Standard for financial forecasting | 7.B.I |
| Partial window training | Tests robustness to incomplete data | 12.B.II |
| Persona-level split | Prevents data leakage across personas | 12.B.I |
| Stratified splitting | Maintains class balance across splits | 8.B.V |

---

*Document created: 2026-07-17*
