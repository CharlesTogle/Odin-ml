# ROC-Based Cutoff Selection

**Document Version:** v1.0
**Author(s):** Guevarra
**Date:** 2026-07-17
**Status:** Draft
**Purpose:** Define how income-stability and obligation-weight thresholds are calibrated for PFP classification

---

## 1. Overview

This document specifies the methodology for calibrating the two binary thresholds that define the eight PFP classes:

1. **Income Stability Threshold (CV Threshold):** Separates Stable from Variable profiles
2. **Obligation Weight Threshold (Obligation Ratio Threshold):** Separates Obligated from Flexible profiles

The key insight from MDD §6 Tier 1 is that these thresholds are **not fixed a priori** — they are calibrated on the training persona set using ROC-based cutoff selection, ensuring Tier 1 (rule-based) gets the same "fitting" opportunity as learned tiers, keeping the comparison fair.

---

## 2. Score Formulas

### 2.1 Income Stability Score (CV)

The income stability score is the **Coefficient of Variation (CV)** of monthly income over a 3-month observation window:

```
CV = σ(monthly_incomes) / μ(monthly_incomes)
```

Where:
- `monthly_incomes` = [month_1_income, month_2_income, month_3_income]
- `σ` = standard deviation
- `μ` = mean

**Interpretation:**
- Low CV → Stable income (predictable amounts)
- High CV → Variable income (unpredictable amounts)

**Range:** [0, ∞)
- CV = 0: Perfectly stable income
- CV = 0.2: Moderate variability
- CV > 0.5: Highly variable income

### 2.2 Obligation Weight Score (Obligation Ratio)

The obligation weight score is the ratio of essential+obligatory expenses to total expenses:

```
Obligation Ratio = (Essential Expenses + Obligatory Expenses) / Total Expenses
```

Where:
- **Essential Expenses:** Food, Housing, Transport, Health, Education
- **Obligatory Expenses:** Debt/loan repayments, remittances (obligations imposed on the user by other parties — not self-selected spending)
  > **Note:** Housing is excluded from Obligatory because it already belongs to Essential (double-count bug fixed). Insurance is excluded because it is typically self-selected rather than imposed; confirm with SME whether mandatory government contributions (SSS/PhilHealth/Pag-IBIG) should be reclassified as Obligatory once those fields are available.
- **Total Expenses:** Sum of all expense categories

**Interpretation:**
- High ratio → Obligated (fixed costs dominate)
- Low ratio → Flexible (discretionary spend dominates)

**Range:** [0, 1]
- Ratio = 1.0: All spending is essential/obligatory
- Ratio = 0.5: Equal split between essential/obligatory and discretionary
- Ratio = 0.0: All spending is discretionary

---

## 3. ROC-Based Threshold Optimization

### 3.1 Problem Framing

We have two binary classification problems:

1. **Income Stability:** Classify personas as Stable (CV < threshold) vs Variable (CV ≥ threshold)
2. **Obligation Weight:** Classify personas as Obligated (ratio > threshold) vs Flexible (ratio ≤ threshold)

For each problem, we need to find the optimal threshold that maximizes classification performance.

### 3.2 ROC Analysis for Income Stability

**Step 1: Compute CV for all training personas**
```python
cv_scores = []
for persona in training_personas:
    monthly_incomes = persona.get_monthly_incomes(window=3)
    cv = np.std(monthly_incomes) / np.mean(monthly_incomes)
    cv_scores.append(cv)
```

**Step 2: Define ground truth labels**
Use each persona's archetype-assigned PFP octant (from synthetic-injection-rules.md §3.3) as the independent ground truth label. Each persona was generated with a known income-stability class (Stable or Variable) based on its archetype's CV range and employment type — these labels were assigned during persona generation, not derived from the CV score itself.

- Stable: archetype assigned Stable income (CV range < 0.2 in PFP-to-FIES mapping)
- Variable: archetype assigned Variable income (CV range ≥ 0.2 in PFP-to-FIES mapping)

**Step 3: Compute ROC curve**
```python
from sklearn.metrics import roc_curve, auc

# For each candidate threshold
thresholds = np.arange(0.05, 0.50, 0.01)
tpr_list = []
fpr_list = []

for threshold in thresholds:
    # Predict: Stable if CV < threshold, Variable otherwise
    y_pred = (cv_scores < threshold).astype(int)
    
    # Compute TPR and FPR
    tp = np.sum((y_pred == 1) & (y_true_stable == 1))
    fn = np.sum((y_pred == 0) & (y_true_stable == 1))
    fp = np.sum((y_pred == 1) & (y_true_stable == 0))
    tn = np.sum((y_pred == 0) & (y_true_stable == 0))
    
    tpr = tp / (tp + fn)  # Sensitivity
    fpr = fp / (fp + tn)  # 1 - Specificity
    
    tpr_list.append(tpr)
    fpr_list.append(fpr)

# Compute AUC
roc_auc = auc(fpr_list, tpr_list)
```

**Step 4: Find optimal threshold**
```python
# Method 1: Youden's J statistic (maximizes TPR - FPR)
j_scores = np.array(tpr_list) - np.array(fpr_list)
optimal_idx = np.argmax(j_scores)
optimal_threshold_cv = thresholds[optimal_idx]

# Method 2: Closest to top-left corner (0, 1)
distances = np.sqrt((1 - np.array(tpr_list))**2 + np.array(fpr_list)**2)
closest_idx = np.argmin(distances)
optimal_threshold_cv = thresholds[closest_idx]

# Method 3: Maximize F1-score (if class imbalance exists)
f1_scores = []
for threshold in thresholds:
    y_pred = (cv_scores < threshold).astype(int)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

optimal_threshold_cv = thresholds[np.argmax(f1_scores)]
```

### 3.3 ROC Analysis for Obligation Weight

**Step 1: Compute obligation ratio for all training personas**
```python
obligation_ratios = []
for persona in training_personas:
    essential = persona.get_expenses(['food', 'housing', 'transport', 'health', 'education'])
    obligatory = persona.get_expenses(['debt_payment', 'remittance'])  # Fields TBD — placeholder until debt/remittance categories are added
    total = persona.get_total_expenses()
    
    ratio = (essential + obligatory) / total if total > 0 else 0
    obligation_ratios.append(ratio)
```

**Step 2: Define ground truth labels**
Use each persona's archetype-assigned PFP octant (from synthetic-injection-rules.md §3.3) as the independent ground truth label. Each persona was generated with a known obligation class (Obligated or Flexible) based on its archetype's obligation ratio range — these labels were assigned during persona generation, not derived from the ratio score itself.

- Obligated: archetype assigned Obligated obligation weight (ratio > 0.5 in PFP-to-FIES mapping)
- Flexible: archetype assigned Flexible obligation weight (ratio ≤ 0.5 in PFP-to-FIES mapping)

**Step 3-4:** Same ROC analysis as income stability, but with:
- Candidate thresholds: np.arange(0.30, 0.70, 0.01)
- Labels inverted (Obligated = 1 if ratio > threshold)

### 3.4 Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Income Stability ROC
axes[0].plot(fpr_list_cv, tpr_list_cv, label=f'ROC curve (AUC = {roc_auc_cv:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].scatter(fpr_list_cv[optimal_idx_cv], tpr_list_cv[optimal_idx_cv], 
                c='red', label=f'Optimal threshold = {optimal_threshold_cv:.3f}')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Income Stability ROC')
axes[0].legend()

# Obligation Weight ROC
axes[1].plot(fpr_list_obl, tpr_list_obl, label=f'ROC curve (AUC = {roc_auc_obl:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].scatter(fpr_list_obl[optimal_idx_obl], tpr_list_obl[optimal_idx_obl],
                c='red', label=f'Optimal threshold = {optimal_threshold_obl:.3f}')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Obligation Weight ROC')
axes[1].legend()

plt.tight_layout()
plt.savefig('roc_cutoff_selection.png', dpi=150)
plt.show()
```

---

## 4. Cross-Validation Strategy

### 4.1 Persona-Level Split

To prevent data leakage, entire personas are held out for validation:

```python
from sklearn.model_selection import StratifiedKFold

# Split personas into folds (not transactions)
persona_labels = [p.label for p in training_personas]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(training_personas)), persona_labels)):
    train_personas = [training_personas[i] for i in train_idx]
    val_personas = [training_personas[i] for i in val_idx]
    
    # Compute optimal threshold on training fold
    optimal_cv, optimal_obl = compute_optimal_thresholds(train_personas)
    
    # Evaluate on validation fold
    accuracy, f1, precision, recall = evaluate_thresholds(
        val_personas, optimal_cv, optimal_obl
    )
    
    print(f"Fold {fold}: CV threshold={optimal_cv:.3f}, Obl threshold={optimal_obl:.3f}")
    print(f"  Accuracy={accuracy:.3f}, F1={f1:.3f}")
```

### 4.2 Threshold Aggregation

After cross-validation, aggregate thresholds across folds:

```python
# Method 1: Mean threshold
final_cv_threshold = np.mean(cv_thresholds_per_fold)
final_obl_threshold = np.mean(obligation_thresholds_per_fold)

# Method 2: Median threshold (more robust to outliers)
final_cv_threshold = np.median(cv_thresholds_per_fold)
final_obl_threshold = np.median(obligation_thresholds_per_fold)

# Method 3: Threshold with highest mean F1 across folds
best_fold = np.argmax(f1_scores_per_fold)
final_cv_threshold = cv_thresholds_per_fold[best_fold]
final_obl_threshold = obligation_thresholds_per_fold[best_fold]
```

---

## 5. Decision Boundary Visualization

### 5.1 2D Scatter Plot

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

# Plot each PFP class
colors = {'Stable/Obligated': 'blue', 'Stable/Flexible': 'green',
          'Variable/Obligated': 'red', 'Variable/Flexible': 'orange'}

for persona in training_personas:
    cv = persona.features['income_stability_cv']
    obl = persona.features['obligation_ratio']
    color = colors[persona.label]
    ax.scatter(cv, obl, c=color, alpha=0.5, s=20)

# Draw decision boundaries
ax.axvline(x=final_cv_threshold, color='black', linestyle='--', 
           label=f'CV threshold = {final_cv_threshold:.3f}')
ax.axhline(y=final_obl_threshold, color='black', linestyle=':', 
           label=f'Obligation threshold = {final_obl_threshold:.3f}')

# Label quadrants
ax.text(final_cv_threshold/2, (final_obl_threshold + 1)/2, 'Stable/Obligated', 
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text((final_cv_threshold + 1)/2, (final_obl_threshold + 1)/2, 'Variable/Obligated',
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(final_cv_threshold/2, final_obl_threshold/2, 'Stable/Flexible',
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text((final_cv_threshold + 1)/2, final_obl_threshold/2, 'Variable/Flexible',
        ha='center', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('Income Stability (CV)')
ax.set_ylabel('Obligation Ratio')
ax.set_title('PFP Decision Boundaries')
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('pfp_decision_boundaries.png', dpi=150)
plt.show()
```

### 5.2 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, classification_report

# Predict on validation set
y_true = [p.label for p in val_personas]
y_pred = []
for p in val_personas:
    cv = p.features['income_stability_cv']
    obl = p.features['obligation_ratio']
    
    if cv < final_cv_threshold and obl > final_obl_threshold:
        y_pred.append('Stable/Obligated')
    elif cv < final_cv_threshold and obl <= final_obl_threshold:
        y_pred.append('Stable/Flexible')
    elif cv >= final_cv_threshold and obl > final_obl_threshold:
        y_pred.append('Variable/Obligated')
    else:
        y_pred.append('Variable/Flexible')

# Print results
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

---

## 6. Handling Edge Cases

### 6.1 Borderline Cases

Personas whose scores fall within a defined margin of the threshold are flagged for expert review:

```python
MARGIN = 0.05  # 5% margin

borderline_personas = []
for p in val_personas:
    cv = p.features['income_stability_cv']
    obl = p.features['obligation_ratio']
    
    # Check if near CV threshold
    if abs(cv - final_cv_threshold) < MARGIN:
        borderline_personas.append(p)
    
    # Check if near obligation threshold
    if abs(obl - final_obl_threshold) < MARGIN:
        borderline_personas.append(p)

# These personas are reserved for manual expert review
# (as specified in MDD §2 Labeling Strategy)
```

### 6.2 Class Imbalance Handling

If PFP classes are imbalanced (e.g., Variable/Obligated is rare):

```python
# Option 1: Adjust threshold to balance classes
# Find threshold that produces equal class sizes
target_size = len(personas) / 4
sorted_cvs = sorted(cv_scores)
threshold_idx = int(target_size)
balanced_cv_threshold = sorted_cvs[threshold_idx]

# Option 2: Use class weights in ROC computation
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

# Option 3: Oversample minority classes before threshold computation
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

## 7. Pre-Registered Outcome Framing

As specified in MDD §7, this evaluation is a **hypothesis test**, not a search for a foregone conclusion:

### 7.1 Hypothesis

**H₁:** A learned classifier (Tier 2-4) beats the rule-based classifier (Tier 1) by a pre-registered margin under partial/noisy data.

**H₀:** No learned tier clears that margin, in which case Tier 1 (the deterministic rule) is selected as the module's final design.

### 7.2 Pre-Registered Margin

The margin is defined as **2 points of Macro-F1** (as specified in MDD §6):

```python
PRE_REGISTERED_MARGIN = 0.02  # 2 points of Macro-F1

# If learned model F1 - rule-based F1 > PRE_REGISTERED_MARGIN:
#   → Learned model wins
# Else:
#   → Rule-based model wins (and this is a valid, reportable result)
```

### 7.3 Decision Rule

```python
# After all models are evaluated:
rule_based_f1 = evaluate_tier1(val_personas, final_cv_threshold, final_obl_threshold)
learned_f1s = {
    'logistic_regression': evaluate_tier2_logistic(val_personas),
    'random_forest': evaluate_tier3_rf(val_personas),
    'svm': evaluate_tier3_svm(val_personas),
    'xgboost': evaluate_tier4_xgb(val_personas),
}

# Find best learned model
best_learned_model = max(learned_f1s, key=learned_f1s.get)
best_learned_f1 = learned_f1s[best_learned_model]

# Apply decision rule
if best_learned_f1 - rule_based_f1 > PRE_REGISTERED_MARGIN:
    print(f"Learned model wins: {best_learned_model} (F1={best_learned_f1:.3f})")
    print(f"  beats rule-based (F1={rule_based_f1:.3f}) by {best_learned_f1 - rule_based_f1:.3f}")
    final_model = best_learned_model
else:
    print(f"Rule-based model wins: F1={rule_based_f1:.3f}")
    print(f"  Best learned model {best_learned_model} (F1={best_learned_f1:.3f})")
    print(f"  did not exceed margin of {PRE_REGISTERED_MARGIN}")
    final_model = 'rule_based'
```

---

## 8. Implementation Checklist

| Step | Task | Status |
|------|------|--------|
| 1 | Compute CV for all training personas | |
| 2 | Compute obligation ratio for all training personas | |
| 3 | Run ROC analysis for income stability | |
| 4 | Run ROC analysis for obligation weight | |
| 5 | Visualize ROC curves | |
| 6 | Select optimal thresholds | |
| 7 | Visualize decision boundaries | |
| 8 | Run cross-validation | |
| 9 | Aggregate thresholds across folds | |
| 10 | Identify borderline cases | |
| 11 | Generate confusion matrix | |
| 12 | Document final thresholds | |

---

## 9. Expected Outputs

| Output | Description | Location |
|--------|-------------|----------|
| `optimal_cv_threshold.txt` | Final CV threshold | `1_problem-statement/` |
| `optimal_obligation_threshold.txt` | Final obligation threshold | `1_problem-statement/` |
| `roc_curves.png` | ROC curves for both thresholds | `1_problem-statement/` |
| `decision_boundaries.png` | 2D scatter plot with thresholds | `1_problem-statement/` |
| `confusion_matrix.png` | Confusion matrix for rule-based classifier | `1_problem-statement/` |
| `classification_report.txt` | Precision, recall, F1 per class | `1_problem-statement/` |
| `borderline_personas.csv` | Personas requiring expert review | `1_problem-statement/` |

---

## 10. RRL Justifications

| Concept | RRL Support | Topic |
|---------|-------------|-------|
| ROC-based threshold selection | Standard ML practice for binary classification | 5.C.VII |
| Persona-level split | Prevents data leakage in user-level evaluation | 12.B.I |
| Pre-registered outcomes | MDD §7: hypothesis test, not foregone conclusion | 12.B.I |
| Borderline case expert review | MDD §2: manual expert review for ambiguous cases | 5.C.VIII |
| Class imbalance handling | SMOTE for minority classes | 8.B.V |

---

*Document created: 2026-07-17*
*Next steps: Implement in Python, validate with synthetic data, run on training personas*
