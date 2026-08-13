# Walk-Forward Validation Protocol

**Document Version:** v1.0
**Author(s):** Guevarra
**Date:** 2026-07-17
**Status:** Draft
**Purpose:** Define the walk-forward validation methodology for temporal model evaluation

---

## 1. Overview

Walk-forward validation (WFV) is the gold standard for evaluating time-series models. Unlike static train/test splits, WFV simulates real-world deployment by:

1. Training on historical data only
2. Predicting the next time step
3. Expanding the training window forward
4. Repeating until all data is exhausted

This methodology is critical for Odin because:
- Financial data has strong temporal dependencies
- Models must generalize to unseen future periods
- Static splits can overfit to specific time periods

---

## 2. Walk-Forward Variants

### 2.1 Expanding Window (EW)

The training window grows over time. All historical data is used:

```
Split 1: Train [1-6], Test [7]
Split 2: Train [1-7], Test [8]
Split 3: Train [1-8], Test [9]
Split 4: Train [1-9], Test [10]
Split 5: Train [1-10], Test [11]
Split 6: Train [1-11], Test [12]
```

**Pros:** Uses all available data. Better for long time series.
**Cons:** Computational cost grows. Older data may be irrelevant.

### 2.2 Rolling Window (RW)

The training window has fixed size. Only recent data is used:

```
Split 1: Train [1-6], Test [7]
Split 2: Train [2-7], Test [8]
Split 3: Train [3-8], Test [9]
Split 4: Train [4-9], Test [10]
Split 5: Train [5-10], Test [11]
Split 6: Train [6-11], Test [12]
```

**Pros:** Fixed computational cost. Adapts to concept drift.
**Cons:** Discards older data. May miss long-term patterns.

### 2.3 Anchored Window (AW)

Similar to rolling, but always anchored to the start:

```
Split 1: Train [1-6], Test [7]
Split 2: Train [1-6, 8], Test [9]
Split 3: Train [1-6, 8, 10], Test [11]
```

**Pros:** Preserves baseline period. Good for anomaly detection.
**Cons:** Complex implementation. May introduce gaps.

---

## 3. Recommended Strategy per Module

### 3.1 PFP Classifier: Rolling Window

The PFP classifier uses a fixed observation window (3 months). Rolling window is most appropriate:

```python
def pfp_walk_forward(personas, obs_window=3):
    results = []
    
    for persona in personas:
        months = persona.monthly_transactions
        n = len(months)
        
        # Generate rolling splits
        for i in range(obs_window, n):
            train_window = months[i - obs_window:i]
            test_month = months[i]
            
            # Compute features from training window
            features = compute_pfp_features(train_window)
            
            # Get ground truth label for test month
            label = get_pfp_label(test_month)
            
            results.append({
                'persona_id': persona.id,
                'split_idx': i - obs_window,
                'features': features,
                'label': label
            })
    
    return results
```

### 3.2 Forecaster: Expanding Window

The LSTM forecaster benefits from more data. Expanding window is recommended:

```python
def forecaster_walk_forward(personas, seq_length=6):
    results = []
    
    for persona in personas:
        months = persona.monthly_transactions
        n = len(months)
        
        # Generate expanding splits
        for i in range(seq_length, n):
            # Training: all data up to i
            train_sequence = months[:i]
            test_month = months[i]
            
            # Pad if needed
            if len(train_sequence) < seq_length:
                padded = [ZeroTransaction()] * (seq_length - len(train_sequence)) + list(train_sequence)
            else:
                padded = list(train_sequence)[-seq_length:]
            
            # Prepare input/output
            X = np.array([month_to_features(m) for m in padded])
            y = month_to_features(test_month)
            
            results.append({
                'persona_id': persona.id,
                'split_idx': i,
                'X': X,
                'y': y
            })
    
    return results
```

### 3.3 Anomaly Detector: Anchored Window

The anomaly detector needs a stable baseline. Anchored window preserves the baseline:

```python
def anomaly_walk_forward(personas, baseline_months=6):
    results = []
    
    for persona in personas:
        months = persona.monthly_transactions
        n = len(months)
        
        # Anchor to first baseline_months
        baseline = establish_baseline(months[:baseline_months])
        
        # Generate anchored splits
        for i in range(baseline_months, n):
            test_month = months[i]
            
            # Detect anomalies in test month
            score = compute_anomaly_score(test_month, baseline)
            is_anomalous = test_month.is_anomalous
            
            results.append({
                'persona_id': persona.id,
                'split_idx': i - baseline_months,
                'anomaly_score': score,
                'is_anomalous': is_anomalous
            })
    
    return results
```

---

## 4. Implementation Details

### 4.1 Walk-Forward Engine

```python
class WalkForwardEngine:
    def __init__(self, strategy='rolling', window_size=6, horizon=1):
        self.strategy = strategy
        self.window_size = window_size
        self.horizon = horizon
    
    def generate_splits(self, data):
        splits = []
        n = len(data)
        
        if self.strategy == 'expanding':
            for i in range(self.window_size, n - self.horizon + 1):
                splits.append({
                    'train': data[:i],
                    'test': data[i:i + self.horizon]
                })
        
        elif self.strategy == 'rolling':
            for i in range(self.window_size, n - self.horizon + 1):
                splits.append({
                    'train': data[i - self.window_size:i],
                    'test': data[i:i + self.horizon]
                })
        
        elif self.strategy == 'anchored':
            baseline = data[:self.window_size]
            for i in range(self.window_size, n - self.horizon + 1):
                splits.append({
                    'train': baseline + data[self.window_size:i],
                    'test': data[i:i + self.horizon]
                })
        
        return splits
    
    def evaluate(self, model, data, metric='accuracy'):
        splits = self.generate_splits(data)
        scores = []
        
        for split in splits:
            model.fit(split['train'])
            y_pred = model.predict(split['test'])
            y_true = [d['label'] for d in split['test']]
            
            if metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'f1':
                score = f1_score(y_true, y_pred, average='macro')
            elif metric == 'mse':
                score = mean_squared_error(y_true, y_pred)
            
            scores.append(score)
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'per_split': scores
        }
```

### 4.2 Cross-Validation Integration

WFV can be combined with cross-validation:

```python
def walk_forward_cross_validation(personas, n_folds=5, strategy='rolling'):
    fold_size = len(personas) // n_folds
    results = []
    
    for fold in range(n_folds):
        # Hold out one fold as test
        test_start = fold * fold_size
        test_end = test_start + fold_size
        test_personas = personas[test_start:test_end]
        
        # Remaining personas for training
        train_personas = personas[:test_start] + personas[test_end:]
        
        # Apply walk-forward within training set
        engine = WalkForwardEngine(strategy=strategy)
        train_results = engine.evaluate(model, train_personas)
        
        # Evaluate on held-out fold
        test_score = evaluate_on_test(model, test_personas)
        
        results.append({
            'fold': fold,
            'train_mean': train_results['mean'],
            'test_score': test_score
        })
    
    return results
```

---

## 5. Evaluation Metrics

### 5.1 Classification Metrics (PFP)

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Balanced classes |
| Macro-F1 | 2 * (precision * recall) / (precision + recall) | Imbalanced classes |
| Cohen's Kappa | (p_o - p_e) / (1 - p_e) | Agreement beyond chance |
| AUC-ROC | Area under ROC curve | Threshold-independent |

### 5.2 Regression Metrics (Forecaster)

| Metric | Formula | Use Case |
|--------|---------|----------|
| MAE | mean(|y_true - y_pred|) | Robust to outliers |
| RMSE | sqrt(mean((y_true - y_pred)^2)) | Penalizes large errors |
| MAPE | mean(|(y_true - y_pred) / y_true|) * 100 | Percentage error |
| R-squared | 1 - (SS_res / SS_tot) | Variance explained |

### 5.3 Anomaly Detection Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Detection Rate | TP / (TP + FN) | Catch rate |
| False Alarm Rate | FP / (FP + TN) | Precision |
| F1-Score | 2 * (precision * recall) / (precision + recall) | Balance |
| AUC-ROC | Area under ROC curve | Threshold-independent |

---

## 6. Statistical Significance Testing

### 6.1 Paired t-test

Compare two models across walk-forward splits:

```python
from scipy import stats

def paired_t_test(scores_a, scores_b):
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### 6.2 Wilcoxon Signed-Rank Test

Non-parametric alternative for small samples:

```python
def wilcoxon_test(scores_a, scores_b):
    stat, p_value = stats.wilcoxon(scores_a, scores_b)
    return {
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### 6.3 Bootstrap Confidence Intervals

Estimate confidence intervals for model performance:

```python
def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    
    return {
        'mean': np.mean(scores),
        'ci_lower': lower,
        'ci_upper': upper
    }
```

---

## 7. Expected Outputs

| Output | Description | Location |
|--------|-------------|----------|
| `walk_forward_results.csv` | Per-split performance metrics | `1_problem-statement/` |
| `model_comparison.md` | Statistical comparison of models | `1_problem-statement/` |
| `confidence_intervals.md` | Bootstrap CIs for best model | `1_problem-statement/` |
| `temporal_stability.png` | Performance over time plot | `1_problem-statement/` |

---

## 8. RRL Justifications

| Concept | RRL Support | Topic |
|---------|-------------|-------|
| Walk-forward validation | Standard for time-series forecasting | 7.B.I |
| Expanding window | Uses all available data | 12.B.I |
| Rolling window | Adapts to concept drift | 12.B.II |
| Anchored window | Preserves baseline for anomaly detection | 12.B.II |
| Bootstrap confidence intervals | Robust uncertainty estimation | 12.B.III |

---

*Document created: 2026-07-17*
