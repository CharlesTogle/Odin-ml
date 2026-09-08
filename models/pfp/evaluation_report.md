# PFP Classifier — Evaluation Report

**Generated:** 2026-09-08T09:13:23.075517
**Folds:** 5
**Pre-registered margin:** 0.02

---

## Winner

**tier3_svm**

> Best learned model (tier3_svm) beats Tier 1 by 0.6050 (> 0.02 margin)

---

## Aggregate Results

| Tier | Macro-F1 (mean ± std) | Accuracy (mean ± std) |
|------|----------------------|----------------------|
| tier0_majority | 0.0363 ± 0.0000 | 0.1700 ± 0.0000 |
| tier1_rule_based | 0.0695 ± 0.0034 | 0.1039 ± 0.0023 |
| tier2_logistic_regression | 0.6656 ± 0.0170 | 0.6454 ± 0.0163 |
| tier2_naive_bayes | 0.4975 ± 0.0028 | 0.4726 ± 0.0024 |
| tier3_random_forest | 0.6585 ± 0.0282 | 0.6548 ± 0.0279 |
| tier3_svm | 0.6745 ± 0.0156 | 0.6770 ± 0.0142 |

---

## Per-Fold Results

### Fold 1

- Train months: [1, 2, 3, 4, 5, 6]
- Test months: [8]
- Train personas: 8400
- Test personas: 1800

| Tier | Macro-F1 | Accuracy |
|------|----------|----------|
| tier0_majority | 0.0363 | 0.1700 |
| tier1_rule_based | 0.0732 | 0.1078 |
| tier2_logistic_regression | 0.6386 | 0.6211 |
| tier2_naive_bayes | 0.4972 | 0.4722 |
| tier3_random_forest | 0.6168 | 0.6139 |
| tier3_svm | 0.6541 | 0.6578 |

### Fold 2

- Train months: [1, 2, 3, 4, 5, 6, 7]
- Test months: [9]
- Train personas: 8400
- Test personas: 1800

| Tier | Macro-F1 | Accuracy |
|------|----------|----------|
| tier0_majority | 0.0363 | 0.1700 |
| tier1_rule_based | 0.0740 | 0.1044 |
| tier2_logistic_regression | 0.6542 | 0.6339 |
| tier2_naive_bayes | 0.4956 | 0.4711 |
| tier3_random_forest | 0.6383 | 0.6356 |
| tier3_svm | 0.6631 | 0.6656 |

### Fold 3

- Train months: [1, 2, 3, 4, 5, 6, 7, 8]
- Test months: [10]
- Train personas: 8400
- Test personas: 1800

| Tier | Macro-F1 | Accuracy |
|------|----------|----------|
| tier0_majority | 0.0363 | 0.1700 |
| tier1_rule_based | 0.0671 | 0.1011 |
| tier2_logistic_regression | 0.6714 | 0.6478 |
| tier2_naive_bayes | 0.4955 | 0.4711 |
| tier3_random_forest | 0.6620 | 0.6578 |
| tier3_svm | 0.6708 | 0.6761 |

### Fold 4

- Train months: [1, 2, 3, 4, 5, 6, 7, 8, 9]
- Test months: [11]
- Train personas: 8400
- Test personas: 1800

| Tier | Macro-F1 | Accuracy |
|------|----------|----------|
| tier0_majority | 0.0363 | 0.1700 |
| tier1_rule_based | 0.0660 | 0.1039 |
| tier2_logistic_regression | 0.6785 | 0.6583 |
| tier2_naive_bayes | 0.4962 | 0.4711 |
| tier3_random_forest | 0.6795 | 0.6733 |
| tier3_svm | 0.6883 | 0.6917 |

### Fold 5

- Train months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
- Test months: [12]
- Train personas: 8400
- Test personas: 1800

| Tier | Macro-F1 | Accuracy |
|------|----------|----------|
| tier0_majority | 0.0363 | 0.1700 |
| tier1_rule_based | 0.0671 | 0.1022 |
| tier2_logistic_regression | 0.6855 | 0.6661 |
| tier2_naive_bayes | 0.5030 | 0.4772 |
| tier3_random_forest | 0.6956 | 0.6933 |
| tier3_svm | 0.6960 | 0.6939 |

---

## Per-Class Accuracy (Last Fold)

| Tier | Stable/Obligated/Tolerant | Stable/Obligated/At-Risk | Stable/Flexible/Tolerant | Stable/Flexible/At-Risk | Variable/Obligated/Tolerant | Variable/Obligated/At-Risk | Variable/Flexible/Tolerant | Variable/Flexible/At-Risk |
|------|------|------|------|------|------|------|------|------|
| tier0_majority | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tier1_rule_based | 0.0305 | 0.0000 | 0.1032 | 0.0000 | 0.5533 | 0.0000 | 0.4810 | 0.0000 |
| tier2_logistic_regression | 0.4915 | 0.9400 | 0.9355 | 0.4477 | 0.6400 | 0.6938 | 0.8291 | 0.6846 |
| tier2_naive_bayes | 0.4305 | 0.4467 | 0.8710 | 0.3366 | 0.8333 | 0.3290 | 0.4684 | 0.4552 |
| tier3_random_forest | 0.8305 | 0.4333 | 0.9484 | 0.3529 | 0.7400 | 0.8664 | 0.7595 | 0.6667 |
| tier3_svm | 0.7220 | 0.5667 | 0.9226 | 0.4869 | 0.8600 | 0.8567 | 0.4937 | 0.6774 |
