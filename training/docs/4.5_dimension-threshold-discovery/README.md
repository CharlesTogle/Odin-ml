# Phase 4.5: PFP Dimension & Threshold Discovery

## What This Phase Is

This phase formally defines the number of classification levels and threshold values for the three **classifying dimensions** (Financial Stability, Financial Obligation, Financial Tolerance), and defines + validates the computation method for the two **overlay indicators** (Financial Trajectory, Financial Margin).

These are **reasoned/derived by the research team**, not statistically discovered from FIES. FIES lacks the panel/temporal structure needed for true empirical threshold discovery (single 6-month cross-sectional record per household, no repeated observation of the same household over time). Any patterns found are bounded by what the synthesis process encoded (circularity caveat) and are provisional pending real user data.

This phase sits between Phase 4 (EDA) and Phase 5 (Feature Engineering) because the dimension definitions and threshold candidates inform which engineered features are produced and how classes are labeled.

## Inputs

| Input | Source | Purpose |
|-------|--------|---------|
| `datasets/engineered/` | Phase 5 output (once available) | Full engineered feature matrices |
| `datasets/processed/` | Phase 3 output | Raw feature matrices (fallback if Phase 5 not yet run) |
| Persona/archetype metadata | Phase 2 | Archetype labels and parameters for validation |

## Process

### 1. Unsupervised Structure Check

Run clustering (k-means / Gaussian Mixture Model) on the continuous Stability, Obligation, and Financial Tolerance scores *independent of the existing PFP label*, with elbow/silhouette analysis, to check whether a strict binary split holds or whether more natural groupings exist.

**Caveat:** Any patterns found are bounded by what the synthesis process encoded (circularity) and are provisional pending real user data.

### 2. New Feature Computation

Implement two new engineered features, computed independently of Stability/Necessity:

- **Financial Trajectory** — trend/slope of income-expense gap across available months (Article IV §5 in Specification.md)
- **Financial Margin** — (Income − Expenses) / Income at a point in time (Article IV §6 in Specification.md)

### 3. Candidate Threshold Documentation

Produce a `dimension-threshold-candidates.md` with proposed cut points and rationale, explicitly scoped as "ready for SME sanity-check on the conceptual framework; numeric values still provisional" (mirrors SME Consultation Log Item 2.2 vs. Backlog Item B.1 split).

### 4. Archetype Table Fix

Flag/fix the Phase 2 archetype table's use of "Income Level: High/Low" as a proxy for Stability. It should instead be derived from actual income variance/CV, since magnitude and consistency are different things (surfaced as a design bug during review).

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| `dimension_discovery.py` | `scripts/` | Clustering analysis + new feature computation script |
| `dimension-threshold-candidates.md` | `docs/4.5_dimension-threshold-discovery/` | Proposed thresholds with rationale |
| Updated `feature-set.md` | `docs/prerequisites/` | Add Financial Trajectory and Financial Margin feature definitions |
| Updated `persona-validation-list.md` | `docs/prerequisites/` | Guidance that personas should vary across Trajectory/Margin combinations within each PFP quadrant |
| Updated `TODO-MDD-Gaps.md` | `docs/prerequisites/` | Log remaining open gap: final threshold values pending SME review + clustering analysis |

## Pipeline Integration

```
datasets/processed/ (Phase 3 output)
    ↓
Phase 4: EDA → figures/
    ↓
Phase 4.5: Dimension & Threshold Discovery (this phase)
    → dimension_discovery.py
    → dimension-threshold-candidates.md
    → Updates to feature-set.md, persona-validation-list.md, TODO-MDD-Gaps.md
    ↓
Phase 5: Feature Engineering → datasets/engineered/
```
