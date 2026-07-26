# Model Design Document - PFP Classifier
**Document Version:** v1.3  
**Module Name:** Personal Financial Profile Classifier  
**Author(s):** Guevarra 
**Date:** 2026-07-26  
**Status:** Draft
**Companion Documents:** `bsp-fies-crosswalk.md`, `synthetic-injection-rules.md`

---

## 0. Module Context & Isolation Boundary (The Invariant Contract)
*This section is non-negotiable and pre-defines the module's API boundaries. The rest of the document serves to realize this contract.*

### 0.1. Module Function
*Classifies a user's Personal Financial Profile (PFP) into one of eight categories (Stable-Flexible-Tolerant, Stable-Flexible-Tight, Stable-Obligated-Tolerant, Stable-Obligated-Tight, Variable-Flexible-Tolerant, Variable-Flexible-Tight, Variable-Obligated-Tolerant, Variable-Obligated-Tight), by selecting and running the best-performing classification algorithm against whatever transaction history currently exists for that user, including partial, short-window, or self-logged data of inconsistent quality.*

### 0.2. Strict Input Contract (What the module receives)
- **Structure:** JSON
- **Required Fields:** 

| Field | Type | Constraints |
| :--- | :--- | :--- |
| `user_id` | String | Non-null, valid UUID format |
| `classification_mode` | String | Must be one of: `STANDARD`, `QUESTIONNAIRE` |
| `payload.historical_transactions` | List[Object] | Required if mode is `STANDARD`; array of transaction objects |
| `payload.questionnaire_answers` | Dict | Required if mode is `QUESTIONNAIRE`; key-value pairs of question IDs and answers |

- **Rejection Criteria:** 
  - Empty `user_id`
  - Invalid `classification_mode` value
  - Missing required payload fields for the specified mode
  - Empty transaction list for `STANDARD` mode
  - Incomplete questionnaire answers for `QUESTIONNAIRE` mode

### 0.3. Strict Output Contract (What the module returns)
- **Structure:** JSON
- **Required Fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `prediction` | String | One of: `STABLE_FLEXIBLE_TOLERANT`, `STABLE_FLEXIBLE_AT_RISK`, `STABLE_OBLIGATED_TOLERANT`, `STABLE_OBLIGATED_AT_RISK`, `VARIABLE_FLEXIBLE_TOLERANT`, `VARIABLE_FLEXIBLE_AT_RISK`, `VARIABLE_OBLIGATED_TOLERANT`, `VARIABLE_OBLIGATED_AT_RISK` |
| `income_stability_score` | Float | Calibrated score between 0 and 1 |
| `obligation_weight_score` | Float | Calibrated score between 0 and 1 |
| `financial_tolerance_score` | Float | Calibrated score between 0 and 1 |
| `confidence` | Float | Calibrated confidence score between 0 and 1 |
| `status` | String | `SUCCESS`, `FAILURE`, or `FALLBACK` |

- **Guaranteed Behavior:** The module must never throw an unhandled exception. It must always return the structured contract (using fallback values if the model fails).

### 0.4. Explicit Non-Responsibilities (Out of Scope)
- **This module does NOT** manage user authentication or session state.
- **This module does NOT** persist classification results to any external database.
- **This module does NOT** handle user consent or privacy policy enforcement.
- **This module does NOT** interact with external microservices during inference.

---

## 1. Problem Statement (Scope, Objectives, and KPIs)
*Define the predictive problem in purely statistical/mathematical terms. Be agnostic about *how* to solve it.*

- **Core Problem:** Multi-class classification with 8 balanced classes (Stable-Flexible-Tolerant, Stable-Flexible-Tight, Stable-Obligated-Tolerant, Stable-Obligated-Tight, Variable-Flexible-Tolerant, Variable-Flexible-Tight, Variable-Obligated-Tolerant, Variable-Obligated-Tight), derived from three binary dimensions (income stability, obligation weight, and financial tolerance). The research question this module answers is: given partial-window, noisy, or inconsistently self-logged transaction data, which classification algorithm — trained to approximate that same label from indirect and partial signals — most reliably reproduces the ground-truth class a full, clean 3-month history would eventually yield?

- **Stakeholders:** The PFP Module consumes the output for profile assignment and drift detection.

---

## 2. Data Collection Plan (Sourcing, Types, and Sizes)
*Define what data is required to train this module, without committing to the final dataset.*

- **Sourcing Strategy:** 
  - **Primary:** Archetype-driven synthetic personas. Archetype segmentation (12 total: 8 canonical PFP octant combinations + up to 4 additional edge-case archetypes) is informed by the **BSP Consumer Finance Report**; each archetype is then parameterized and expanded into granular, time-stamped transaction data using **PSA 2023 FIES microdata** (income/expense totals, family size and weight, population weight, urban/rural, per-capita income, decile ranking), disaggregated to the individual level, via a shared persona-generation methodology. See `bsp-fies-crosswalk.md` for the field-level mapping between the two sources.
  - **Secondary:** Historical anonymized transaction data from thesis prototype users, once available, used to validate synthetic-to-real generalization
  - **Tertiary:** Questionnaire response data from onboarding, used as a secondary/validation label source, not the primary ground truth

- **Required Data Types:** 
  - Time-series transaction data (amount, category, date, type)
  - Income frequency and amount data
  - Expense category distributions
  - Indirect/behavioral features injected during persona augmentation (spending timing relative to payday, category diversity, amount volatility, inter-transaction gaps)
  - Binary labels for income stability, obligation weight, and financial tolerance

- **Why indirect features must be synthetically injected rather than sourced from FIES:** FIES microdata is anonymized per RA 10173 (Data Privacy Act) and PSA disclosure policy, and contains only the aggregate/geographic fields listed above — it was never designed to capture person-level behavioral signals like spending timing or transaction-level volatility. There is no available Filipino dataset that pairs household financial totals with granular behavioral transaction data. Persona-based synthesis is therefore the only feasible route to a labeled dataset with both a defensible ground-truth class and the indirect features the learned candidates are meant to exploit.

- **Minimum Viable Dataset Size:** 
  - At least 500 labeled personas per class (4,000 total for 8 classes), each with a full 3-month "mature" transaction history for ground-truth label derivation
  - Each mature persona additionally sampled at multiple partial-window cuts (e.g., first 2 weeks, first 4 weeks, first 6 weeks) to train and evaluate candidates under the realistic partial-data condition described in Section 1
  - Explicit persona-generation targets set per PFP octant to avoid underrepresenting classes likely to be rarer in the general population (e.g., Variable-Obligated-Tight)
  - **[NOTE]** If BSP segment analysis confirms one or more edge-case archetypes (beyond the 8 octants), this module's ground-truth formula must state whether such personas resolve to their nearest octant or are excluded from Classifier training entirely (they may still be valid Forecaster/Anomaly Detector personas). See `bsp-fies-crosswalk.md` §3.3.

- **Labeling Strategy:** 
  - **Primary (ground truth):** Deterministic label = score/threshold formula applied to each persona's full, mature 3-month window. This is computed automatically, not manually annotated, since income stability, obligation weight, and financial tolerance are themselves defined as computable scores with fixed thresholds.
  - **Secondary (validation only):** Questionnaire-style answers generated per persona, checked for agreement with the deterministic label
  - **Manual expert review:** Reserved only for personas whose computed score sits within a defined margin of the classification threshold (i.e., ambiguous/borderline cases) — a small, tractable review load for a single financial expert, rather than annotating all 4,000 samples from scratch

---

## 3. Data Preprocessing & Cleaning Specifications
*Identify potential data quality issues and the criteria for handling them. Explicitly keep the methods open-ended.*

- **Potential Issues:** 
  - Missing transaction data (gaps in history)
  - Duplicate entries from bulk imports
  - Inconsistent category naming
  - Outlier amounts (extremely high/low values)
  - Class imbalance in income stability labels

- **Handling Criteria (Agnostic):** 
  - *Decision Gate:* We will evaluate the impact of deleting incomplete records vs. imputation
  - *Candidate Techniques to Explore:* 
    - Null imputation: Mean/median imputation for continuous variables, mode for categorical
    - Duplicate deduplication: Exact matching + fuzzy matching for transaction descriptions
    - Outlier capping: IQR method (1.5× IQR threshold) or Z-score (>3σ)
    - Class balancing: SMOTE or class-weighted training

- **Data Versioning:** Raw and cleaned datasets will be stored in versioned S3 buckets or a local data registry with SHA hashes for reproducibility.

---

## 4. Exploratory Data Analysis (EDA) Plan
*State the hypotheses to test and the visualizations/investigations required before modeling.*

- **Hypotheses to Validate:** 
  - Income stability is strongly correlated with employment type
  - Obligation weight is correlated with age and household size
  - The eight PFP classes are separable in the transaction-feature space
  - Stable-Obligated users show consistent patterns in essential spending

- **Required Statistical Summaries:** 
  - Distribution of target variables (class counts)
  - Feature means/variances across classes
  - Skewness of transaction amounts by category
  - Monthly income variance by employment type

- **Correlation Targets:** 
  - Income stability vs. income standard deviation, employment type, income frequency
  - Obligation weight vs. essential spending ratio, household size, age
  - Class membership vs. spending category distributions

- **Outcome of this phase:** The EDA will determine if we need feature scaling, log-transformations, or more aggressive cleaning.

---

## 5. Feature Engineering Strategy
*Define the target feature space dimensions and the strategies to compare.*

- **Feature Encoding Candidates:** 
  - One-Hot Encoding for categorical variables (employment type, category groups)
  - Label Encoding for ordinal categories (education level, age ranges)
  - Cyclical encoding for temporal features (month, day of week)

- **Derived Features:**
  - Income stability: Standard deviation of monthly income, coefficient of variation, frequency consistency score
  - Obligation weight: Ratio of essential+obligatory expenses to total expenses
  - Financial tolerance: Emergency runway (months of expenses covered by savings)
  - Spending diversity: Number of distinct categories used
  - Temporal patterns: Weekend vs. weekday spending ratio, seasonal patterns

- **Scaling Candidates:** 
  - Min-Max scaling (feature range = [0,1])
  - Standardization (Z-score normalization)
  - Robust scaling (using median and IQR, less sensitive to outliers)

- **Dimensionality Reduction Candidates:** 
  - PCA (to reduce to 10-20 components if overfitting occurs)
  - Feature importance selection from Random Forest (top N features)
  - t-SNE for visualization only (not for production)

- **Constraint:** Feature engineering pipeline must fit within memory/compute constraints defined in Section 1.

---

## 6. Data Modeling Strategy (Algorithm Selection & Baseline)
*Define the tiered approach to modeling. **Crucially**, do not select the final model here; define the selection criteria.*

- **Tier 0: Naive Baseline (Sanity Floor):** `[Cost: Low]`
  - Majority-class predictor (always predicts the most frequent PFP class in the training persona set)
  - Purpose: The absolute floor any candidate — rule-based or learned — must clear to be considered informative at all

- **Tier 1: Deterministic Rule-Based Classifier:** `[Cost: Low]`
  - Direct application of the income-stability, obligation-weight, and financial-tolerance score/threshold definitions to whatever transaction window is available (full or partial), producing the 3D octant label
  - Threshold values are calibrated on the training persona set (e.g., via ROC-based cutoff selection) rather than fixed a priori, so the comparison against learned candidates is fair

- **Tier 2: Intermediate Learned Model:** `[Cost: Low]`
  - Logistic Regression (multi-class) with L2 regularization
  - Purpose: Simplest learned candidate; establishes whether any learning signal exists beyond the deterministic rule at all

- **Tier 3: Ensemble/Margin-Based Models:** `[Cost: Medium]`
  - Random Forest Classifier
  - Support Vector Machine (RBF kernel)
  - Purpose: Test whether ensemble/non-linear methods can exploit the injected indirect/behavioral features better than a threshold rule or linear model

- **Tier 4: Advanced Model:** `[Cost: Medium-High]`
  - XGBoost Classifier (gradient boosting, handles class imbalance well)
  - Multi-layer Perceptron (if deep learning proves necessary)
  - Justified only if Tiers 1–3 fail to meet the KPI

- **Selection Criteria:** Final algorithm chosen based on Performance-to-Latency trade-off across **all** tiers (0–4), not just the learned ones:
  - Highest Macro-F1 score within latency budget, evaluated on the same held-out personas and the same partial-window cuts for every candidate
  - Must maintain >75% per-class accuracy across all 8 PFP classes
  - Model size must be deployable in mobile-app context
  - Ties or near-ties (within a pre-registered margin, e.g., 2 points of Macro-F1) are broken in favor of the simpler/more interpretable candidate — i.e., Tier 1 over Tier 3/4 — given the sensitivity of financial classification and the value of an explainable decision to the user

---

## 7. Model Evaluation Plan
*Define the rigorous methodology to compare models fairly, without predefining the outcomes.*

- **Primary Evaluation Metric:** 
  - Macro-F1 Score (primary, prioritizes performance across all eight classes equally)
  - Accuracy (secondary, for overall performance)
  - Justification: Macro-F1 handles class imbalance better and treats each PFP class equally important

- **Secondary Metrics:** 
  - Per-class precision, recall, F1
  - Inference time (P95 latency)
  - Model size (MB)
  - Failure rate on edge cases (e.g., users with very few transactions)

- **Validation Strategy:** 
  - **Persona-level split** (not transaction-level): entire personas are held out for validation/test, so no candidate is evaluated on a persona whose data it was trained on
  - Stratified K-Fold (k=5) at the persona level to maintain class proportions across folds
  - **Every candidate — Tier 0 through Tier 4 — is evaluated on the identical held-out set and identical partial-window cuts** (full history, first 2 weeks, first 4 weeks, first 6 weeks), so the comparison isolates algorithm choice rather than data access
  - Time-series split (3 months train, 1 month validation) to additionally test temporal generalization on mature-history personas

- **Confusion Matrix Analysis:** 
  - Specifically analyze false-positive rates (misclassifying as Stable vs. Variable)
  - Analyze misclassifications between same-income-dimension classes (e.g., Stable-Flexible-Tolerant vs. Stable-Flexible-Tight — same income and obligation, different tolerance)
  - Analyze misclassifications between same-obligation-dimension classes (e.g., Stable-Flexible-Tolerant vs. Stable-Obligated-Tolerant — same income and tolerance, different obligation)

- **Pre-registered outcome framing:** This evaluation is a hypothesis test, not a search for a foregone conclusion. Either result is a valid, reportable contribution: (a) a learned tier beats Tier 1 by a pre-registered margin under partial/noisy data, supporting the case for a learned classifier; or (b) no learned tier clears that margin, in which case Tier 1 (the deterministic rule) is selected as the module's final design, and the finding — that a well-calibrated deterministic rule is sufficient for this problem — is documented as the module's actual result rather than treated as an incomplete experiment. The 8-class PFP structure (3 binary dimensions) increases the difficulty relative to the original 4-class formulation, making the Tier 1 vs. learned comparison even more informative.

---

## 8. Optimization Strategy (Tuning & Ensembling)
*Define the hyperparameter landscape and optimization budget.*

- **Tier 1 (Rule-Based) Calibration:** 
  - Income-stability, obligation-weight, and financial-tolerance thresholds are not fixed a priori; calibrate cut points on the training persona set (e.g., ROC-based threshold selection maximizing Macro-F1) so Tier 1 gets the same "fitting" opportunity as the learned tiers, keeping the comparison fair

- **Tuning Candidates (Tiers 2–4):** 
  - Grid Search (exhaustive, but limited to small search spaces)
  - Random Search (preferred for larger hyperparameter spaces)

- **Hyperparameter Space for Random Forest:**
  - `n_estimators`: [100, 200, 300, 500]
  - `max_depth`: [10, 20, 30, None]
  - `min_samples_split`: [2, 5, 10]
  - `class_weight`: ['balanced', 'balanced_subsample']
  - `max_features`: ['sqrt', 'log2']

- **Hyperparameter Space for XGBoost:**
  - `max_depth`: [3, 5, 7]
  - `learning_rate`: [0.01, 0.1, 0.3]
  - `n_estimators`: [100, 200, 300]
  - `subsample`: [0.8, 1.0]
  - `scale_pos_weight`: [1, 2, 5] (for handling imbalance)

- **Ensemble Candidates:** 
  - Stacking: Logistic Regression meta-model on top of Random Forest + XGBoost predictions
  - Voting: Soft voting (averaging probabilities) between Random Forest and XGBoost
  - Only tested if single model plateaus below KPI

---

## 9. Deployment Strategy & Module Packaging
*Define how the finished model will be delivered as a module, without choosing the specific cloud provider.*

- **Interface Requirement:** 
  - Python function: `predict(input: Dict) -> Dict`
  - Must strictly adhere to Section 0 contract
  - Async support for concurrent requests

- **Artifact Delivery:** 
  - Trained model weights and preprocessing pipeline serialized as `.pkl` (Pickle) or `.joblib`
  - Feature transformation pipeline (Scaler, Encoder) also serialized
  - Model version metadata (training data hash, performance metrics)

- **Integration Point:** 
  - Containerized API service (Flask/FastAPI) deployed on cloud infrastructure (e.g., AWS EC2, GCP Compute Engine)
  - Called by mobile app via REST API
  - Dependency injection for model loading at startup

- **Deployment Candidates:** 
  - Docker container (preferred for reproducibility)
  - Serverless function (if latency requirements can be met)
  - Decision based on required latency (Section 1): Docker on dedicated instance for consistent <500ms

---

## 10. Model Monitoring & Lifecycle Plan
*Define how we will know the module is degrading over time, without specifying the monitoring tool.*

- **Metrics to Track Post-Deployment:** 
  - *Input Drift:* Monitor distribution changes in transaction features (income std. dev., obligation weight ratio, financial tolerance score)
  - *Output Drift:* Monitor changes in prediction confidence and 8-class PFP distribution
  - *Performance:* Weekly accuracy on known labeled samples (if available)

- **Retraining Triggers:** 
  - Weekly average confidence drops below 0.70 for 3 consecutive days
  - Per-class accuracy (on validation set) drops below 0.70
  - Significant input drift detected (PSI > 0.1)

- **Feedback Loop:** 
  - Users can manually correct their PFP via the PFP Screen
  - These corrections serve as ground truth labels for model improvement
  - Quarterly model retraining with accumulated labeled data

---

## 11. Timeline, Dependencies, and Decision Gates
*Break down the 10 phases into a realistic schedule for the thesis project.*

### Shared Data Phase (All Modules)
| Phase | Deliverable | Estimated Duration | Success Gate (Go/No-Go) |
| :--- | :--- | :--- | :--- |
| 1–3 (Problem + Data) | Problem statement, cleaned dataset, synthetic personas | 16 days | Persona set passes expert + distributional validation; minimum 4,000 labeled personas for PFP, 12,000 personas for Forecaster/Anomaly; archetypes validated against BSP CFS (see `bsp-fies-crosswalk.md`) |

### PFP Classifier Module (Picks up from shared Data)
| Phase | Deliverable | Estimated Duration | Success Gate (Go/No-Go) |
| :--- | :--- | :--- | :--- |
| 4 & 5 (EDA & Features) | Feature matrix ready; EDA report | 10 days | EDA confirms class separability on both full and partial windows; feature importance identified |
| 6 & 7 (Modeling & Eval) | Tier 0-4 comparative results | 14 days | Tier 0 (majority) beaten by all other tiers; Tier 1 (rule) and Tier 3 (Random Forest) both evaluated on identical held-out personas |
| 8 (Optimization) | Final selected candidate (rule-based or learned) | 7 days | Selected candidate meets KPI: Macro-F1 > 0.80; selection documented against the pre-registered margin from Section 7 |
| 9 & 10 (Deploy & Monitor) | Packaged module + monitoring setup | 10 days | Integration test passes; monitoring metrics configured |

**PFP Module Duration: 41 days (from feature engineering onward)**

### Forecaster Module (Picks up from shared Data)
| Phase | Deliverable | Estimated Duration | Success Gate (Go/No-Go) |
| :--- | :--- | :--- | :--- |
| 4 & 5 (EDA & Features) | Feature matrix ready; EDA report | 10 days | EDA confirms weekly/semi-monthly/monthly seasonality; ACF/PACF analysis complete; SHAP feature importance identified |
| 6 & 7 (Modeling & Eval) | Baseline + Intermediate results | 14 days | ARIMA baseline established; XGBoost/LightGBM beats ARIMA by >20% MAPE reduction [Bhavana et al., 2025] |
| 8 (Optimization) | Final optimized model | 10 days | Final model meets KPI: MAPE < 15% at total level; researcher-defined targets |
| 9 & 10 (Deploy & Monitor) | Packaged module + monitoring setup | 10 days | Integration test passes; drift detection (ADWIN/CUSUM) configured; monitoring metrics configured |

**Forecaster Module Duration: 44 days (from feature engineering onward)**

### Anomaly Detector Module (Picks up from shared Data)
| Phase | Deliverable | Estimated Duration | Success Gate (Go/No-Go) |
| :--- | :--- | :--- | :--- |
| 4 & 5 (EDA & Features) | Feature matrix ready; EDA report | 7 days | EDA confirms anomaly patterns; feature importance identified; temporal patterns validated |
| 6 & 7 (Modeling & Eval) | Baseline + Intermediate results | 14 days | IQR baseline established; Isolation Forest beats IQR by >50% F1 improvement |
| 8 (Optimization) | Final optimized model | 10 days | Final model meets KPI: F1 ≥ 0.85, Recall ≥ 0.85, FPR ≤ 0.05 [Huang A. et al., 2025; Al Rafi, 2024] |
| 9 & 10 (Deploy & Monitor) | Packaged module + monitoring setup | 10 days | Integration test passes; drift detection (ADWIN/EDDM) configured; monitoring metrics configured |

**Anomaly Detector Module Duration: 41 days (from feature engineering onward)**

### Critical Path
- **Shared Data Phase (16 days)** → Feature engineering triplicates → Modules run in parallel
- **Longest module:** Forecaster (44 days from feature engineering)
- **Total critical path:** 16 + 44 = **60 days (~2 months)**
- PFP (41 days) and Anomaly (41 days) finish before Forecaster; their deploy/monitor phases can overlap with Forecaster optimization

---

## 12. Cold-Start Handling
*Unlike the Forecaster and Anomaly Detector, this module does not require a separate ML cold-start hypothesis.* Day-0 users (no transaction history at all) are served by `QUESTIONNAIRE` mode (Section 0.2, Article VI §2), a deterministic mapping from questionnaire answers to an initial PFP — not a model inference path. This module's Tier 0-4 comparison only applies once `STANDARD` mode is reachable, i.e., once at least some transaction history exists. The boundary between the two is intentional: the questionnaire is the system's cold-start answer for this module, consistent with the Specification's own note (Article VI §3) that a separate ML cold-start fallback is likely unnecessary here.

---

## 13. Concept Drift & Assumptions
*Distinguish model-level drift (this module's learned parameters becoming stale) from user-level drift (Article IV §4, a person's own profile changing), which is a product feature, not a modeling concern.*

- **Model-Level Concept Drift:** The relationship between indirect/behavioral features and the ground-truth PFP label may shift as the target-user economy changes (inflation, minimum wage adjustments, growth of gig work). Monitor for this via the Section 10 drift metrics (PSI on input feature distributions, rolling per-class accuracy) applied specifically to the deployed candidate, whichever tier that turns out to be.
- **Key Assumption (synthetic-to-real gap):** All KPI figures in this document are measured on synthetic personas. Generalization to real users is untested until real prototype-user data is available (Section 2, Secondary source) and is treated as an explicit limitation of this design, not a guarantee.
- **Key Assumption (injected feature realism):** Indirect/behavioral features are manually injected into personas based on RRL citations and financial-expert judgment, not observed in FIES. Any performance gap between Tier 1 and the learned tiers is only as trustworthy as these injection rules; this is documented as a threat to validity rather than assumed away.
- **Key Assumption (BSP-FIES crosswalk):** Archetype segmentation is sourced from the BSP Consumer Finance Report, while granular transaction data is sourced from PSA FIES microdata — two surveys with different units of analysis and sampling frames. Any mismatch between a BSP-defined archetype and the FIES-derived income/expense profile assigned to it is a threat to validity, not a confirmed equivalence; the crosswalk documentation (`bsp-fies-crosswalk.md`) exists to make this mapping explicit and auditable rather than implicit.

---

# Model Design Document - Forecaster
**Document Version:** v2.3  
**Module Name:** Forecaster  
**Author(s):** Guevarra
**Date:** 2026-07-26  
**Status:** Draft
**Companion Documents:** `bsp-fies-crosswalk.md`, `synthetic-injection-rules.md`

---

## 0. Module Context & Isolation Boundary (The Invariant Contract)
*This section is non-negotiable and pre-defines the module's API boundaries. The rest of the document serves to realize this contract.*

### 0.1. Module Function
*Forecasts future expenses at total, category group, and category levels based on historical transaction data using deep learning and ensemble methods.*

### 0.2. Strict Input Contract (What the module receives)
- **Structure:** JSON
- **Required Fields:**

| Field | Type | Constraints |
| :--- | :--- | :--- |
| `user_id` | String | Non-null, valid UUID format |
| `historical_transactions` | List[Object] | Non-empty list; each must contain amount, category, date, type |
| `forecast_horizon` | String | One of: `WEEKLY` (7 days), `SEMI_MONTHLY` (15 days), `MONTHLY` (30 days) |
| `forecast_level` | String | One of: `TOTAL`, `CATEGORY_GROUP`, `CATEGORY` |
| `target_categories` | List[String] | Optional; required if forecast_level is `CATEGORY`; list of category IDs |
| `user_metadata` | Object | Optional; demographic data for cold-start scenarios |

- **Rejection Criteria:**
  - Empty `historical_transactions`
  - Invalid `forecast_horizon` enum value
  - Invalid `forecast_level` value
  - `target_categories` empty when `forecast_level = CATEGORY`

> **Scope Note:** The module is limited to the three fixed horizons above (weekly, semi-monthly, monthly) rather than an arbitrary day count. This is a feasibility constraint, not a modeling preference: Variable-income users (freelancers, gig workers, project-based/contractual workers) report irregular and seasonal pay periods, so training and validating a single model against longer, arbitrary horizons is not achievable within the thesis timeline. Restricting the model to these three horizons keeps evaluation tractable and aligns forecasts to the pay-cycle boundaries most Filipino users actually plan around (weekly, semi-monthly "cutoff" pay, and monthly).

### 0.3. Strict Output Contract (What the module returns)
- **Structure:** JSON
- **Required Fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `forecasts` | List[Object] | Array of forecast objects; each contains date, amount, and metadata |
| `forecast_level` | String | Echo of input forecast_level |
| `forecast_horizon` | String | Echo of input forecast_horizon (`WEEKLY`, `SEMI_MONTHLY`, or `MONTHLY`) |
| `confidence_intervals` | Object | Upper and lower bounds at 80% and 95% confidence |
| `model_version` | String | Version identifier of the model used |
| `status` | String | `SUCCESS`, `FAILURE`, or `FALLBACK` |

- **Guaranteed Behavior:** The module must never throw an unhandled exception. It must always return the structured contract (using fallback values if the model fails).

### 0.4. Explicit Non-Responsibilities (Out of Scope)
- **This module does NOT** manage user authentication or authorization.
- **This module does NOT** store forecast results in any database.
- **This module does NOT** provide budget recommendations (handled by Budget Recommender module).
- **This module does NOT** handle anomaly detection (handled by Anomaly Detector module).

---

## 1. Problem Statement (Scope, Objectives, and KPIs)
*Define the predictive problem in purely statistical/mathematical terms. Be agnostic about *how* to solve it.*

- **Core Problem:** Multi-step time series forecasting of expense amounts at hierarchical levels (total, category group, category), for three fixed horizons (weekly, semi-monthly, monthly), with heteroscedastic noise (variance increases with amount), seasonal patterns (weekly, semi-monthly, monthly), and potential concept drift over time. Longer or arbitrary horizons are explicitly out of scope: Variable-income users' irregular/seasonal pay cycles make longer-horizon targets unreliable to label and validate given the thesis timeline, so the module commits only to the three horizons above.

- **Quantitative Objectives (KPIs):**
  - **Primary:**
    - **MAPE** (Mean Absolute Percentage Error) < 15% at total level
      - **⚠ Domain mismatch note:** The originally cited benchmarks (NNAR 2.67%, CNN-LSTM 2.72% [Krstev et al., 2023; Ullah et al., 2024]) are from electricity load forecasting and generic time-series datasets, not household/personal finance. These benchmarks are not directly comparable to Odin's synthetic Filipino persona data. The 15% target is a researcher-defined fallback based on: (a) household expense forecasting is inherently noisier than electricity load forecasting due to irregular spending patterns, discretionary variability, and income volatility; (b) Odin's data is synthetic with injected behavioral features, adding a layer of approximation; (c) realistic targets for personal finance forecasting in developing-economy contexts should account for higher income volatility and irregular transaction patterns.
    - **MAPE** < 20% at category group level
    - **MAPE** < 25% at category level
  - **Secondary:**
    - **R²** > 0.70
      - **⚠ Domain mismatch note:** The originally cited benchmark (Ensemble R² = 0.9921 [Sonkavde et al., 2023]) is from stock price prediction, a fundamentally different domain. Household expense forecasting has inherently lower R² due to higher variance in discretionary spending. The 0.70 target is a researcher-defined fallback.
    - **RMSE** < 25% of mean daily spending
    - **Inference latency:** P95 < 1s (LSTM inference + post-processing)
    - **Compute budget:** RAM < 1GB, GPU optional [Hall, 2025]
  - **Business Metrics:**
    - Forecast error reduction of at least 22% compared to heuristic baseline (Literature: ML models reduce prediction errors by 22-33% compared to heuristic forecasts [Bhavana et al., 2025])

- **Stakeholders:** The Forecasting Module (Article XXV) and Budgeting Module (Article XXI) consume the output.

---

## 2. Data Collection Plan (Sourcing, Types, and Sizes)
*Define what data is required to train this module, without committing to the final dataset.*

- **Sourcing Strategy:**
  - **Primary:** Archetype-driven synthetic persona/transaction pipeline, shared with the PFP Classifier and Anomaly Detector through preprocessing/EDA and triplicated at feature engineering for this module's specification — 12,000 personas across 12 archetypes (8 PFP octants + up to 4 BSP-informed edge cases; see `bsp-fies-crosswalk.md`), each with daily transaction history. Archetype segmentation is informed by the **BSP Consumer Finance Report**; income/expense deciles, employment type, and other granular transaction parameters are drawn from **PSA 2023 FIES NCR microdata**.
  - **Secondary:** Historical anonymized transaction data from thesis prototype users, once available — used to validate synthetic-to-real generalization
  - **Tertiary:** PSA 2023 FIES microdata (source for persona parameterization, not used directly for training)

- **Required Data Types:**
  - **Time-series:** Daily aggregated expenses by category at total, group, and category levels
  - **Categorical:** Category hierarchies (PSA PCOICOP-based)
  - **Temporal:** Calendar features (day of week, month, holiday indicators, payday proximity)
  - **Transaction metadata:** Amount, category, date, type
  - **Demographic metadata:** Age, employment type, income range, PFP (for cold-start)

- **Minimum Viable Dataset Size:**
  - 12,000 personas (1,000 per archetype × 12 archetypes), each with a full 12-month transaction history
  - ~144,000 monthly summary rows (12,000 personas × 12 months)
  - At least 500 transactions per category for category-level modeling

- **Labeling Strategy:**
  - Supervised (ground truth = future expenses)
  - No manual labeling required (historical data provides natural labels)
  - Time-series split: Training = first 70% of timeline, Validation = next 15%, Test = last 15%

---

## 3. Data Preprocessing & Cleaning Specifications
*Identify potential data quality issues and the criteria for handling them. Explicitly keep the methods open-ended.*

- **Potential Issues:**
  - Missing days in transaction history (days with no transactions)
  - Irregular transaction frequencies (bulk recording of multiple transactions)
  - Outlier amounts (erroneous entries)
  - Category inconsistencies (user-defined vs. standard categories)
  - Seasonal effects (holidays, paydays affecting spending)
  - Zero or negative transaction amounts

- **Handling Criteria (Agnostic):**
  - *Decision Gate:* Evaluate impact of imputation vs. interpolation vs. deletion
  - *Candidate Techniques to Explore:*
    - **Missing values:** Linear interpolation for daily aggregates, forward-fill for categories (Literature: IQR-based detection with Q1 - 1.5×IQR, Q3 + 1.5×IQR [Ullah et al., 2024])
    - **Outlier capping:** Winsorization at 99th percentile or IQR method
    - **Frequency normalization:** Resampling to daily frequency with aggregation
    - **De-seasonalization:** Decompose time series using STL (Seasonal-Trend decomposition using Loess) [Thundiyil et al., 2025]
    - **Zero/negative amounts:** Filter out (invalid transaction)
    - **Seasonal adjustment:** Apply MSTL (Multiple Seasonal-Trend decomposition) for multi-seasonal periods [Williams et al., 2023]

- **Data Versioning:** Raw and cleaned time series stored with version tags; feature sets cached for reproducibility.

---

## 4. Exploratory Data Analysis (EDA) Plan
*State the hypotheses to test and the visualizations/investigations required before modeling.*

- **Hypotheses to Validate:**
  - Daily spending follows a weekly pattern (higher on weekends, paydays) - supported by literature [Ahmed & Dey, 2023]
  - Spending shows semi-monthly seasonality tied to the common 15th/end-of-month Philippine payroll cutoff, distinct from the broader monthly cycle
  - Monthly spending shows seasonality (higher at month start/end) - supported by literature [Williams et al., 2023]
  - Spending patterns differ across PFP classes - supported by literature [Sappa, 2024]
  - Autocorrelation exists at weekly (7-day), semi-monthly (15-day), and monthly (30-day) lags - supported by literature [Thundiyil et al., 2025]

- **Required Statistical Summaries:**
  - Autocorrelation function (ACF) and partial autocorrelation (PACF) - critical for understanding temporal dependencies [Song et al., 2025]
  - Daily spending distribution (mean, variance, skewness)
  - Temporal patterns: Day-of-week effects, month-of-year effects
  - Stationarity tests (ADF test for unit root) - literature notes non-stationarity is the norm in financial data [CITATION NOT FOUND — verify or remove claim]
  - Seasonal decomposition plots (trend, seasonal, residual components)

- **Correlation Targets:**
  - Spending vs. day of week - Literature: retained users show higher merchant diversity [Ahmed & Dey, 2023]
  - Spending vs. payday proximity
  - Category spending inter-correlations
  - Spending vs. user demographic features (age, employment type)

- **Outcome of this phase:** Determines whether to use ARIMA-style autoregressive features, seasonal decomposition, or deep learning approaches [Song et al., 2025].

---

## 5. Feature Engineering Strategy
*Define the target feature space dimensions and the strategies to compare.*

- **Feature Encoding Candidates:**
  - **Temporal encoding:** Day of week (sin/cos), month (sin/cos), day of year, hour (if sub-daily)
  - **Lag features:** Previous 1, 7, 14, 15, 30, 60, 90 days of spending - literature shows temporal order is critical (disrupting order causes up to 1092% MSE increase [Song et al., 2025])
  - **Rolling statistics:** 7-day, 14-day, 15-day, 30-day moving averages and standard deviations
  - **Calendar features:** Payday indicators, holiday indicators, payday proximity
  - **RFM metrics:** Recency, Frequency, Monetary value - identified as most predictive features [Ahmed & Dey, 2023; CITATION NOT FOUND — verify or remove claim]

- **Derived Features:**
  - Day-over-day change (momentum)
  - Spending volatility (rolling standard deviation)
  - Category-level spending ratios (proportion of total)
  - Weekday vs. weekend spending ratio
  - Inter-transaction gaps - literature shows retained users have shorter gaps (11.2 days vs 34.7 days) [Ahmed & Dey, 2023]
  - Merchant diversity score - literature shows retained users have higher diversity (M=6.8 vs 2.9) [Ahmed & Dey, 2023]

- **Decomposition Features:**
  - **STL decomposition:** Trend, seasonal, and residual components [Thundiyil et al., 2025]
  - **MSTL:** Multiple seasonal periods for daily + weekly patterns [Williams et al., 2023]

- **Scaling Candidates:**
  - **Min-Max scaling** (to [0,1] range) - for neural network compatibility
  - **Standardization** (Z-score) - for LSTM compatibility
  - **Robust scaling** (using median and IQR) - less sensitive to outliers

- **Dimensionality Reduction Candidates:**
  - Feature selection: SHAP importance on XGBoost - literature shows SHAP-guided selection reduced features from 86 to 10 (88.5% reduction) while maintaining comparable performance [Schwartz et al., 2024]
  - Genetic Algorithm for optimal feature subset - literature shows GA increased AUC by 21% compared to using all features [Almonteros et al., 2024]
  - Target: 10-15 features (literature shows 7-12 features capture most discriminatory information, peaking at 86% accuracy [Lien & Rajasekharan, 2024])

- **Constraint:** Feature engineering must fit within memory/compute constraints; avoid creating excessive lag features that increase inference time.

---

## 6. Data Modeling Strategy (Algorithm Selection & Baseline)
*Define the tiered approach to modeling. **Crucially**, do not select the final model here; define the selection criteria.*

- **Tier 1: Baseline Model (Must-Have):** `[Cost: Low]`
  - **ARIMA** (Auto-Regressive Integrated Moving Average) - literature: superior for small, linear, or strongly seasonal datasets [Kontopoulou et al., 2023]
  - **ETS** (Exponential Smoothing) - literature: strong baseline for seasonal data (MAPE 3.28%) [Krstev et al., 2023]
  - **Prophet** - literature: achieved lowest weighted error (10.456) vs ARIMA, LSTM [Mariano & Monreal, 2025]
  - Purpose: Establish statistical model baseline

- **Tier 2: Intermediate Model:** `[Cost: Medium]`
  - **XGBoost Regressor** with lag and calendar features - literature: XGBoost outperforms ARIMA and LSTM in some settings (MSE 360.0) [Sonkavde et al., 2023]
  - **LightGBM** - literature: provides best balance between forecasting accuracy and computational efficiency [CITATION NOT FOUND — verify or remove claim]; tree-based models are 126,934% faster to train than deep learning [Hall, 2025]
  - **Random Forest Regressor** - literature: 0.9208 R² for poverty prediction [Onsay & Rabajante, 2024]
  - Purpose: Tree-based ensemble performance; fast training for mobile-first constraints

- **Tier 3: Advanced Model:** `[Cost: High]`
  - **PyTorch LSTM** - literature: LSTM achieved 4.82% MAPE vs 20.97% for ARIMA [Ao & Fayek, 2023]; effective for capturing long-term dependencies [Casolaro et al., 2023]. Implemented via `torch.nn.LSTM` for CPU-efficient training without CUDA dependency.
  - **PyTorch GRU** (Gated Recurrent Unit) - literature: offers comparable accuracy with lower compute than LSTM [Singh U. et al., 2025]. Implemented via `torch.nn.GRU`.
  - **PyTorch BiLSTM** (Bidirectional LSTM) - literature: captures both forward and backward dependencies. Implemented via `torch.nn.LSTM(bidirectional=True)`.
  - **Transformer-based variants** (if needed) - literature: PatchTST achieves 23% lower MSE than Informer with 60% less GPU memory [Song et al., 2025]
  - **Hybrid Models** - literature: ARIMA-NARNN reduced RMSE by 35.3% vs ARIMA alone [Kontopoulou et al., 2023]; LSTM-Logistic Regression achieved AUC-ROC of 0.91 [CITATION NOT FOUND — verify or remove claim]
  - Justification: LSTM is specified in the title; PyTorch chosen over TensorFlow for CPU-efficient training without CUDA dependency; handles sequential dependencies and long-term patterns

- **Selection Criteria:**
  - Lowest MAPE within latency budget
  - Must beat ARIMA baseline by at least 20% reduction in MAPE (Literature benchmark: ML models reduce error by 22-33% [Bhavana et al., 2025])
  - Model size must be deployable (<500MB)
  - For mobile-first constraints, prioritize LightGBM or GRU over full LSTM [Hall, 2025]

---

## 7. Model Evaluation Plan
*Define the rigorous methodology to compare models fairly, without predefining the outcomes.*

- **Primary Evaluation Metric:**
  - **MAPE** (Mean Absolute Percentage Error) at the specified forecast level - literature: recommended for scale-independent comparison [Krstev et al., 2023; Ullah et al., 2024]
  - Justification: Interpretable as percentage error; comparable across different spending levels

- **Secondary Metrics:**
  - **R²** - literature: coefficient of determination for fit quality [Sonkavde et al., 2023]
  - **RMSE** (Root Mean Square Error) - literature: penalizes large errors heavily [Ullah et al., 2024]
  - **MAE** (Mean Absolute Error) - literature: robust, intuitive metric [Bhavana et al., 2025]
  - **Inference time** (P95 latency) - literature: critical for mobile-first applications [Hall, 2025]
  - **Model size** (MB) - literature: 91% of ML models degrade in production, often within days [CITATION NOT FOUND — verify or remove claim]

- **Validation Strategy:**
  - **Time-series split:** Training set = first 70% of timeline, Validation set = next 15%, Test set = last 15%
  - **Walk-forward validation:** Training on increasing windows, evaluating one step ahead - literature: recommended for time series to prevent information leakage [CITATION NOT FOUND — verify or remove claim]
  - **Rolling forecast evaluation:** Forecast N days, compare to actual, roll forward
  - **Purged cross-validation:** To prevent information leakage [CITATION NOT FOUND — verify or remove claim]

- **Confusion Matrix Analysis:**
  - Not applicable for regression; analyze:
    - Error distribution (bias: under/over-forecasting patterns)
    - Error by category (identify categories where model underperforms)
    - Error by user PFP (identify profile-specific performance differences)

---

## 8. Optimization Strategy (Tuning & Ensembling)
*Define the hyperparameter landscape and optimization budget.*

- **Tuning Candidates:**
  - **Random Search** - preferred for LSTM due to many hyperparameters
  - **Bayesian Optimization** - if compute budget allows
  - **Grid Search** - for tree-based models with limited hyperparameters

- **Hyperparameter Space for PyTorch LSTM/GRU:**
  - `n_layers`: [1, 2, 3]
  - `n_units`: [32, 64, 128, 256] - literature: GRU offers comparable accuracy with lower compute [Singh U. et al., 2025]
  - `dropout`: [0.1, 0.2, 0.3]
  - `batch_size`: [16, 32, 64]
  - `learning_rate`: [0.001, 0.01]
  - `sequence_length`: [7, 14, 15, 30, 60] (lookback window) - literature: lookback window of 24-336 time steps [Song et al., 2025; Ullah et al., 2024]
  - `activation`: ['tanh', 'relu']
  - `bidirectional`: [True, False] (for BiLSTM)
  - `optimizer`: ['adam', 'sgd'] (PyTorch optimizers)
  - `weight_decay`: [0, 1e-5, 1e-4] (L2 regularization)

- **Hyperparameter Space for XGBoost/LightGBM:**
  - `n_estimators`: [100, 200, 500]
  - `max_depth`: [3, 5, 7, 10] - literature: XGBoost achieved high F1 (0.680) and AUC (0.872) [CITATION NOT FOUND — verify or remove claim]
  - `learning_rate`: [0.01, 0.1, 0.3]
  - `min_child_weight`: [1, 3, 5]
  - `subsample`: [0.8, 0.9, 1.0]
  - `colsample_bytree`: [0.8, 0.9, 1.0]

- **Ensemble Candidates:**
  - **Weighted averaging** of LSTM and XGBoost predictions - literature: ensemble techniques generally provide superior performance over standalone models [Sonkavde et al., 2023]
  - **Stacking:** Linear regression meta-model on LSTM + XGBoost outputs
  - **Voting ensemble** of multiple models - literature: Voting Ensemble achieved highest R² (78.11%) [Thakur & Jadhav, 2025]
  - **Hybrid ARIMA-LSTM** - literature: hybrid frameworks improve forecasting robustness by combining linear and non-linear modeling [CITATION NOT FOUND — verify or remove claim]
  - Tested only if single model plateaus below KPI

- **Optimization Budget:**
  - Max 50 trials per model for Random Search
  - Max 30 iterations for Bayesian Optimization
  - Ensemble optimization after individual models are tuned

---

## 9. Deployment Strategy & Module Packaging
*Define how the finished model will be delivered as a module, without choosing the specific cloud provider.*

- **Interface Requirement:**
  - Python function: `predict(input: Dict) -> Dict`
  - Must strictly adhere to Section 0 contract
  - Async support for concurrent requests
  - Lightweight inference runtime (<500MB RAM) for mobile-first constraints [Hall, 2025]

- **Artifact Delivery:**
  - **PyTorch LSTM/GRU weights:** `.pth` format (PyTorch state dict)
  - **scikit-learn models:** `.joblib` format (Random Forest, XGBoost, LightGBM)
  - **Preprocessing pipeline:** `.pkl` or `.joblib`
  - **Scaler and feature transformer:** serialized as above
  - **Model version metadata:** training data hash, performance metrics, creation date
  - **Fallback model:** ARIMA/ETS statistical model for cold-start

- **Integration Point:**
  - **Containerized API service** (FastAPI preferred for async support) - literature: inference latency critical for user experience [Hall, 2025]
  - Called by mobile app via REST API
  - Dependency injection for model loading at startup
  - **Lightweight screening at edge** - detect potential concept drift before sending to cloud [CITATION NOT FOUND — verify or remove claim]

- **Deployment Candidates:**
  - **Docker container** (preferred for reproducibility)
  - **Serverless function** (if latency requirements can be met)
  - **Decision:** Docker on dedicated instance with GPU optional; fallback to CPU-only for cost optimization
  - **Mobile-first:** Consider exporting GRU model via PyTorch Mobile for on-device inference [Hall, 2025]

---

## 10. Model Monitoring & Lifecycle Plan
*Define how we will know the module is degrading over time, without specifying the monitoring tool.*

- **Metrics to Track Post-Deployment:**
  - *Input Drift:* Monitor distribution changes in feature values (lagged spending, calendar features) - literature: use Population Stability Index (PSI) with threshold 10% [CITATION NOT FOUND — verify or remove claim]
  - *Output Drift:* Monitor prediction confidence intervals and forecast variance
  - *Performance:* Compare forecasts to actuals (MAPE, RMSE) on weekly basis
  - *Concept Drift:* Detect using ADWIN, DDM, or Page-Hinkley - literature: 91% of ML models degrade in production [CITATION NOT FOUND — verify or remove claim]

- **Drift Detection Methods:**
  - **Hybrid detection:** CUSUM for abrupt drift + moving average for gradual drift - literature: detection delay 31.2 instances for abrupt, 64.8 for gradual [Yashwanth et al., 2023]
  - **SEED and STEPD** - literature: consistently outperform other drift detectors [CITATION NOT FOUND — verify or remove claim]
  - **Adaptive thresholding** - literature: adjusts sensitivity based on stream volatility [Yashwanth et al., 2023]

- **Retraining Triggers:**
  - Rolling MAPE > 20% for 3 consecutive weeks at total level
  - Significant input drift detected (PSI > 0.1)
  - Major changes in user behavior (e.g., new job, relocation)
  - **Severity-aware retraining:** Quantile transformation reduces unnecessary retraining - literature: 93% of features exhibited significant drift [Shakhovska & Pukach, 2025]

- **Adaptation Strategies:**
  - **Sliding-window retraining** - literature: improves AUC by up to 5 percentage points during volatile periods [CITATION NOT FOUND — verify or remove claim]
  - **Parameter updates** - literature: faster than structural updates [Xiang et al., 2023]
  - **Model reuse pool** - literature: Pool ICM reduced retraining events by up to 94% [Eliades & Papadopoulos, 2025]
  - **Online adaptive RNN** - literature: effective for load forecasting under concept drift [Xiang et al., 2023]

- **Feedback Loop:**
  - Actual transaction data used to compute forecast accuracy
  - Errors flagged for model improvement
  - User feedback on forecast accuracy collected via app (optional)
  - **Monthly retraining** with latest data - literature: continuous curation and periodic retraining maintain performance near peak levels (sawtooth pattern) [CITATION NOT FOUND — verify or remove claim]

- **Evaluation Metrics for Drift:**
  - **OldTransfer** (knowledge preservation) - literature: GEM achieved 0.1785 OldTransfer on Forest Cover [Pai et al., 2024]
  - **NewTransfer** (adaptation to new data) - literature: critical for measuring adaptation performance [Pai et al., 2024]
  - **Backward Transfer (BWT)** - measures forgetting [Martin et al., 2023]

---

## 11. Timeline, Dependencies, and Decision Gates
*Break down the 10 phases into a realistic schedule for the thesis project.*

### Shared Data Phase (All Modules)
| Phase | Deliverable | Estimated Duration | Success Gate (Go/No-Go) |
| :--- | :--- | :--- | :--- |
| 1–3 (Problem + Data) | Problem statement, cleaned dataset, synthetic personas | 16 days | Persona set passes expert + distributional validation; minimum 4,000 labeled personas for PFP, 12,000 personas for Forecaster/Anomaly; archetypes validated against BSP CFS (see `bsp-fies-crosswalk.md`) |

### Forecaster Module (Picks up from shared Data)
| Phase | Deliverable | Estimated Duration | Success Gate (Go/No-Go) |
| :--- | :--- | :--- | :--- |
| 4 & 5 (EDA & Features) | Feature matrix ready; EDA report | 10 days | EDA confirms weekly/semi-monthly/monthly seasonality; ACF/PACF analysis complete; SHAP feature importance identified |
| 6 & 7 (Modeling & Eval) | Baseline + Intermediate results | 14 days | ARIMA baseline established; XGBoost/LightGBM beats ARIMA by >20% MAPE reduction [Bhavana et al., 2025] |
| 8 (Optimization) | Final optimized model | 10 days | Final model meets KPI: MAPE < 15% at total level; researcher-defined targets |
| 9 & 10 (Deploy & Monitor) | Packaged module + monitoring setup | 10 days | Integration test passes; drift detection (ADWIN/CUSUM) configured; monitoring metrics configured |

**Forecaster Module Duration: 44 days (from feature engineering onward)**

---

## 12. Cold-Start Hypothesis
*Define how the module handles new users with limited or no historical data.*

- **Core Hypothesis:** New users can be modeled using demographic proxies, synthetic data generation, or zero-shot transfer learning until sufficient transaction history is accumulated.

- **Strategies:**
  1. **Zero-shot forecasting with TEMPO** `[Cost: High]` - literature: pre-trained transformer with decomposition and soft prompts; achieved 6.5% MAE improvement on Weather dataset, 19.1% on ETTm1 [Cao et al., 2024]. Implemented in PyTorch.
  2. **Self-supervised learning (SSL)** `[Cost: High]` - literature: learns representations from unlabeled data; achieved AUC 0.91 for credit risk vs XGBoost's 0.84 [Yachamaneni et al., 2025]
  3. **Synthetic data generation** `[Cost: Medium]` - literature: PAnDA framework using LLMs achieved 37.55% improvement in Recall@10 on sparse datasets [CITATION NOT FOUND — verify or remove claim]
  4. **Knowledge transfer via GCN** `[Cost: High]` - literature: TechCD framework enables zero-shot cognitive diagnosis; achieved 56.73% ACC with out-domain data [Gao et al., 2023]
  5. **Behavioral profiling** `[Cost: Low]` - use user-declared preferences and demographic data (age, employment type, PFP) as proxies
  6. **Default fallback** `[Cost: Low]` - use ARIMA/ETS statistical model until enough data is available - literature: simple models often perform comparably to complex models when data is limited [Krstev et al., 2023]

- **Combined Approach:**
  - **Immediate:** Zero-shot TEMPO or SSL-based representation using only demographic features
  - **First 30 days:** Hybrid ARIMA + demographic proxy features
  - **After 3 months:** Full LSTM or ensemble model with personalized features
  - **Dynamic update:** User profile continuously updated as actual data arrives

---

## 13. Concept Drift & Assumptions
*Define how the module handles changing user behavior and economic conditions over time.*

- **Core Assumption:** Non-stationarity is the norm in financial data, not the exception. Static models degrade over time and require continuous adaptation.

- **Types of Drift to Handle:**
  - **Abrupt drift** - sudden changes (e.g., job change, pandemic) - literature: parameter updates preferred [Hovakimyan & Bravo, 2024]
  - **Gradual drift** - slow changes over time (e.g., lifestyle changes) - literature: detected via moving average [Hovakimyan & Bravo, 2024]
  - **Recurring drift** - past distributions reappear (e.g., seasonal patterns) - literature: least addressed; model reuse pool can handle [Eliades & Papadopoulos, 2025]

- **Detection Methods:**
  - **ADWIN** (ADaptive WINdowing) - literature: with incremental learning dominates on gradual drift [CITATION NOT FOUND — verify or remove claim]
  - **DDM** (Drift Detection Method) - literature: reacts fastest to abrupt shifts [CITATION NOT FOUND — verify or remove claim]
  - **CUSUM** - literature: for abrupt drift detection [Yashwanth et al., 2023]
  - **Hybrid approach** - CUSUM + moving average; detection delay 31.2 instances for abrupt, 64.8 for gradual [Yashwanth et al., 2023]

- **Adaptation Strategies:**
  1. **Sliding-window retraining** - literature: improves AUC by up to 5 percentage points during volatile periods [CITATION NOT FOUND — verify or remove claim]
  2. **Parameter updates** - literature: faster than structural updates [Xiang et al., 2023]
  3. **Model reuse pool** - literature: Pool ICM reduced retraining events by up to 94% [Eliades & Papadopoulos, 2025]
  4. **Severity-aware adaptation** - literature: quantile transformation reduced KS statistic from 0.0559 to 0.0072 [Shakhovska & Pukach, 2025]
  5. **Lifelong learning (GEM)** - literature: outperforms others in preserving old knowledge (0.1785 OldTransfer) [Pai et al., 2024]

- **Monitoring Metrics:**
  - **OldTransfer** - knowledge preservation [Pai et al., 2024]
  - **NewTransfer** - adaptation to new data [Pai et al., 2024]
  - **F1-score on post-drift classification** - literature: 0.89 achieved with hybrid detection [Yashwanth et al., 2023]
  - **False positive rate** - literature: 0.041 vs ADWIN's 0.147 [Yashwanth et al., 2023]

- **Key Recommendation:** Hierarchical architecture combining:
  1. **Lightweight screening at edge** - flag potential shifts [CITATION NOT FOUND — verify or remove claim]
  2. **Rigorous validation in cloud** - with severity-aware thresholds [Shakhovska & Pukach, 2025]
  3. **Feedback loop** - refine model and re-deploy [CITATION NOT FOUND — verify or remove claim]

- **Key Assumption (synthetic-to-real gap):** All KPI figures in this document are measured on synthetic personas. Generalization to real users is untested until real prototype-user data is available (Section 2, Secondary source) and is treated as an explicit limitation of this design, not a guarantee.
- **Key Assumption (injected feature realism):** Indirect/behavioral and temporal features are manually injected into personas based on RRL citations and financial-expert judgment, not observed directly in FIES. Any reported forecast accuracy is only as trustworthy as these injection rules; this is documented as a threat to validity rather than assumed away.
- **Key Assumption (BSP-FIES crosswalk):** Archetype segmentation is sourced from the BSP Consumer Finance Report, while granular transaction data is sourced from PSA FIES microdata — two surveys with different units of analysis and sampling frames. Any mismatch between a BSP-defined archetype and the FIES-derived income/expense profile assigned to it is a threat to validity, not a confirmed equivalence; see `bsp-fies-crosswalk.md`.

---

# Model Design Document - Anomaly Detector
**Document Version:** v2.2  
**Module Name:** Anomaly Detector  
**Author(s):** Guevarra
**Date:** 2026-07-26  
**Status:** Draft
**Companion Documents:** `bsp-fies-crosswalk.md`, `synthetic-injection-rules.md`

---

## 0. Module Context & Isolation Boundary (The Invariant Contract)
*This section is non-negotiable and pre-defines the module's API boundaries. The rest of the document serves to realize this contract.*

### 0.1. Module Function
*Detects anomalous transactions (unusual spending patterns) and overspending transactions (exceeding category budgets) using unsupervised anomaly detection and rule-based thresholding.*

### 0.2. Strict Input Contract (What the module receives)
- **Structure:** JSON
- **Required Fields:**

| Field | Type | Constraints |
| :--- | :--- | :--- |
| `user_id` | String | Non-null, valid UUID format |
| `transactions` | List[Object] | Non-empty list; each must contain amount, category, date, type, category_group |
| `detection_type` | String | One of: `ANOMALOUS`, `OVERSPENDING`, `BOTH` |
| `budget_allocations` | List[Object] | Required if detection_type is `OVERSPENDING` or `BOTH`; contains category budget amounts |
| `whitelist` | List[Object] | Optional; user-defined exclusions |
| `user_metadata` | Object | Optional; demographic data for cold-start scenarios |

- **Rejection Criteria:**
  - Empty `transactions` list
  - Invalid `detection_type` value
  - Missing `budget_allocations` when detection_type = `OVERSPENDING` or `BOTH`
  - Transaction missing required fields (amount, category, date)
  - Zero or negative transaction amounts (filtered out)

### 0.3. Strict Output Contract (What the module returns)
- **Structure:** JSON
- **Required Fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `anomalous_transactions` | List[Object] | List of transaction IDs flagged as anomalous; includes anomaly_score, reason, and feature_contributions |
| `overspending_transactions` | List[Object] | List of transaction IDs flagged as overspending; includes budget_excess, category, and period |
| `model_version` | String | Version identifier of the model used |
| `confidence` | Float | Calibrated confidence score between 0 and 1 |
| `status` | String | `SUCCESS`, `FAILURE`, or `FALLBACK` |

- **Guaranteed Behavior:** The module must never throw an unhandled exception. It must always return the structured contract (using fallback values if the model fails).

### 0.4. Explicit Non-Responsibilities (Out of Scope)
- **This module does NOT** manage user settings or whitelist preferences (handled by Anomaly Detection Module UI).
- **This module does NOT** send notifications (handled by Notifications Module).
- **This module does NOT** take remedial actions on flagged transactions.
- **This module does NOT** interact with external microservices during inference.
- **This module does NOT** perform forecasting or budget recommendations.

---

## 1. Problem Statement (Scope, Objectives, and KPIs)
*Define the predictive problem in purely statistical/mathematical terms. Be agnostic about *how* to solve it.*

- **Core Problem:** Two-layer anomaly detection: (1) identify transactions that are statistically unusual given the user's historical spending patterns, and (2) identify transactions that exceed budget constraints. The system must handle extreme class imbalance (anomalies < 1-5% of transactions), concept drift (changing spending patterns), and cold-start scenarios (new users with limited history).

- **Quantitative Objectives (KPIs):**
  - **Primary (Anomalous Detection):**
    - **F1-Score** ≥ 0.85 (Literature benchmark: TA-IFDC achieved 0.927 [Huang A. et al., 2025]; SSL+Hybrid achieved 0.96 [Al Rafi, 2024])
    - **Recall** ≥ 0.85 (Literature benchmark: CS-DNN achieved 0.82-0.91 recall [Serdan, 2025])
    - **False Positive Rate** ≤ 5% (Literature benchmark: TA-IFDC achieved 0.041 FPR [Huang A. et al., 2025]; DataStream Adapt achieved 0.041 FPR [Yashwanth et al., 2023])
    - **AUC-ROC** ≥ 0.90 (Literature benchmark: Autoencoder achieved 0.971 [Fariha et al., 2025])
  - **Secondary (Overspending Detection):**
    - **Precision** = 100% (Rule-based, deterministic)
    - **Recall** = 100% (No false negatives by design)
  - **Operational:**
    - **Inference latency:** P95 < 300ms (Literature benchmark: Hybrid routing achieves <10-100ms for real-time [Ahmed et al., 2025; Abd-Ellatif et al., 2025])
    - **Compute budget:** RAM < 500MB, no GPU required (Literature: LightGBM achieves 0.58ms latency with low resource usage [Shaha & Gavekar, 2025])
  - **Business Metrics:**
    - **Early Detection Rate** > 70% within first quarter (Literature benchmark: HFSL achieved 0.726 EDR [Zhang & Duan, 2025])
    - **Cost Savings:** Cost-sensitive optimization should prioritize recall over precision, with `C_fn >> C_fp` [Karthikeyan et al., 2026]

- **Stakeholders:** The Anomaly Detection Module (Article XXIX) and Notifications Module (Article XXXV) consume the output.

---

## 2. Data Collection Plan (Sourcing, Types, and Sizes)
*Define what data is required to train this module, without committing to the final dataset.*

- **Sourcing Strategy:**
  - **Primary:** Archetype-driven synthetic persona/transaction pipeline with anomaly injection, shared with the PFP Classifier and Forecaster through preprocessing/EDA and triplicated at feature engineering for this module's specification — 12,000 personas across 12 archetypes (8 PFP octants + up to 4 BSP-informed edge cases; see `bsp-fies-crosswalk.md`), with ~3% of transactions labeled as anomalous via five injection types (monetary spikes, category velocity, temporal deviations, merchant novelty, budget overages). Archetype segmentation is informed by the **BSP Consumer Finance Report**; granular transaction data is generated from **PSA 2023 FIES NCR microdata**. See synthetic-injection-rules.md §5.
  - **Secondary:** Historical anonymized transaction data from thesis prototype users, once available — used to validate synthetic-to-real generalization
  - **Tertiary:** Publicly available fraud detection datasets for pre-training (e.g., credit card fraud datasets)

- **Required Data Types:**
  - **Transaction data:** Amount, category, date, category group, merchant category
  - **Historical context:** Previous transaction amounts by category (minimum 3 months recommended)
  - **Budget data:** Category budgets (allocation amounts and periods)
  - **Behavioral features:** Recency, frequency, monetary value (RFM), inter-transaction gaps
  - **Labels:** Binary (anomalous vs. normal) for supervised training of ensemble models
  - **Demographic metadata:** Age, employment type, PFP (for cold-start)

- **Minimum Viable Dataset Size:**
  - 12,000 personas (1,000 per archetype × 12 archetypes), each with a full 3-month transaction history
  - At least 3% anomalous transactions (synthetic injection target) — within the 1-5% literature range [George et al., 2023]
  - At least 500 transactions per category for category-level modeling

- **Labeling Strategy:**
  - **Semi-supervised:** Normal data assumed to be clean; anomalies injected synthetically
  - **Manual labeling:** Candidate anomalies labeled by domain experts for validation
  - **Self-supervised:** Contrastive and masked objectives for unsupervised learning (Literature: SSL achieves F1=0.96 without labels [Al Rafi, 2024])

---

## 3. Data Preprocessing & Cleaning Specifications
*Identify potential data quality issues and the criteria for handling them. Explicitly keep the methods open-ended.*

- **Potential Issues:**
  - Missing category information
  - Zero or negative transaction amounts
  - Duplicate transactions
  - Whitelist exclusions (user-defined)
  - Concept drift in transaction patterns
  - Seasonal variations (holiday spending, payday patterns)

- **Handling Criteria (Agnostic):**
  - *Decision Gate:* Evaluate impact of removing vs. imputing missing categories
  - *Candidate Techniques to Explore:*
    - **Missing categories:** Impute as "UNKNOWN" category (excluded from detection) - literature: missingness is predictive and should be treated as signal rather than noise [Chen et al., 2024]
    - **Zero/negative amounts:** Filter out (invalid transaction)
    - **Duplicates:** Deduplicate via exact + fuzzy matching
    - **Whitelist:** Pre-filter transactions matching whitelist criteria
    - **Outlier capping:** IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR) [Ullah et al., 2024]
    - **Seasonal adjustment:** Apply MSTL (Multiple Seasonal-Trend decomposition) for multi-seasonal periods [Williams et al., 2023]
    - **Missingness indicators:** Create binary flags for missing values (literature shows missingness can be more predictive than many financial ratios [Chen et al., 2024])

- **Data Versioning:** Preprocessed datasets stored with version tags; whitelist applied at inference time.

---

## 4. Exploratory Data Analysis (EDA) Plan
*State the hypotheses to test and the visualizations/investigations required before modeling.*

- **Hypotheses to Validate:**
  - Transaction amount distributions by category are log-normal - supported by literature [Fariha et al., 2025]
  - Anomalous transactions are rare (<5% of total) - supported by literature [George et al., 2023]
  - Spending patterns differ by PFP - supported by literature [Sappa, 2024]
  - Outliers in amount are often legitimate (e.g., medical emergencies, large purchases)
  - Behavioral features are more predictive than monetary features - literature: behavioral variables significantly improved model fit (χ²=145.5, p<0.001) while monetary variables did not [Hopfgartner et al., 2024]
  - Temporal patterns exist (late-night transactions have highest risk score of 0.72) [Fariha et al., 2025]

- **Required Statistical Summaries:**
  - Amount distribution per category (mean, median, variance, percentiles)
  - Transaction frequency by category
  - Inter-transaction time distribution
  - Category-wise outlier detection using IQR
  - Temporal patterns (hour of day, day of week, seasonal) - literature: fraud peaks late-night and weekends [Luong & Xie, 2026]
  - Feature correlations and multicollinearity

- **Correlation Targets:**
  - Transaction amount vs. category
  - Transaction amount vs. day of week/month
  - Spending vs. budget allocation ratios
  - Behavioral features vs. anomaly labels

- **Outcome of this phase:** Determines if amount distributions warrant log-transformation, which features are most predictive, and how to handle seasonal outliers.

---

## 5. Feature Engineering Strategy
*Define the target feature space dimensions and the strategies to compare.*

- **Feature Encoding Candidates:**
  - **Amount:** Raw, log-transformed, Z-score normalized
  - **Category:** One-Hot Encoding or target encoding
  - **Temporal:** Day of week (sin/cos), hour of day (sin/cos), month, holiday indicators - literature: late-night transactions (0-6 AM) have highest risk score of 0.72 [Fariha et al., 2025]
  - **Behavioral:** Recency, frequency, monetary value (RFM) - literature: RFM features are most predictive [Ahmed & Dey, 2023; CITATION NOT FOUND — verify or remove claim]

- **Derived Features:**
  - **Amount deviation:** `(amount - category_mean) / category_std` - literature: amount deviation from cardholder average is discriminative [Fariha et al., 2025]
  - **Category-specific percentile rank**
  - **Rolling average deviation:** `amount - 7-day moving average`
  - **Category frequency:** how often category appears in user's history
  - **Amount relative to budget:** `amount / budget_allocation`
  - **Inter-transaction gap:** days since last transaction
  - **Hesitation index:** time between transaction start and completion - literature: hesitation index improves churn prediction from 61% to 86% [CITATION NOT FOUND — verify or remove claim]
  - **Behavioral volatility:** standard deviation of transaction amounts over rolling window - literature: behavioral volatility is predictive of risk [Islam et al., 2025]
  - **Merchant diversity score** - literature: retained users have higher diversity (M=6.8 vs 2.9) [Ahmed & Dey, 2023]
  - **Trust-weighted embeddings** combining long-term and short-term behavior - literature: BTVE adapts to behavioral drift with automatic retraining [CITATION NOT FOUND — verify or remove claim]
  - **Wallet balance trend** - literature: factor of 1.74x over other features [Olabintan, 2026]

- **Missingness Indicators:**
  - Binary flags for missing features - literature: Missing(Attr27) was highly predictive—more so than many financial ratios [Chen et al., 2024]

- **Scaling Candidates:**
  - **Standardization** (for Isolation Forest compatibility)
  - **Min-Max scaling** (to [0,1] range)
  - **Robust scaling** (using median and IQR) - less sensitive to outliers

- **Dimensionality Reduction Candidates:**
  - **PCA** (to preserve 90-95% variance) - literature: PCA effectively reduces dimensionality for anomaly detection [Li et al., 2026]
  - **Feature selection:** SHAP importance on XGBoost - literature: SHAP-guided selection reduced features from 86 to 10 (88.5% reduction) while maintaining comparable performance [Schwartz et al., 2024]
  - **Genetic Algorithm** for optimal feature subset - literature: GA increased AUC by 21% compared to using all features [Almonteros et al., 2024]
  - **Target: 7-12 features** - literature shows this captures most discriminatory information (peaking at 86% accuracy) [Lien & Rajasekharan, 2024]

- **Constraint:** Feature engineering must fit within memory/compute constraints; avoid creating too many features that increase inference time.

---

## 6. Data Modeling Strategy (Algorithm Selection & Baseline)
*Define the tiered approach to modeling. **Crucially**, do not select the final model here; define the selection criteria.*

- **Tier 1: Baseline Model (Must-Have):** `[Cost: Low]`
  - **IQR (Inter-Quartile Range) rule-based detection:** median ± 3×IQR
  - **Purpose:** Establish simple statistical baseline
  - **Expected performance:** F1 < 0.50 (limitation of simple rules)

- **Tier 2: Intermediate Model (Unsupervised):** `[Cost: Medium]`
  - **Isolation Forest** - literature: efficient for high-dimensional data; achieves 95.3% detection rate with 4.8% FPR [Zhong, 2025]; isolates anomalies via random partitioning [Fariha et al., 2025]
  - **Autoencoder** - literature: reconstruction error as anomaly score; highest AUC-ROC (0.971) among unsupervised methods [Fariha et al., 2025]
  - **One-Class SVM** - literature: useful for cold-start scenarios [Bader & Haraty, 2025]
  - **Purpose:** Unsupervised anomaly detection; establish unsupervised baselines

- **Tier 3: Advanced Model (Primary):** `[Cost: High]`
  - **Hybrid Ensemble** - literature: ensemble methods consistently outperform single algorithms [Luong & Xie, 2026; Kashif & Naseer, 2025]
    - **LightGBM + Isolation Forest:** Combines supervised and unsupervised detection
    - **XGBoost + DNN ensemble:** Achieved F1-score of 0.74 with 0.35ms inference time [Luong & Xie, 2026]
    - **Voting Ensemble:** Combined multiple classifiers; achieved 99.904% accuracy [Vijayanand & Smrithy, 2025]
  - **Temporal-Attention Isolation Forest with Dynamic Calibration (TA-IFDC):** - literature: achieved F1=0.927, AUC=0.974, FPR=0.041 [Huang A. et al., 2025]
    - Uses dynamic threshold calibration: `θ_k = (1-λ)θ_{k-1} + λ Quantile_β(𝒮_k)`
    - Maintains stable alert rates during seasonal traffic shifts
  - **Cost-Sensitive Deep Neural Network (CS-DNN):** - literature: improved recall from 0.58 to 0.82 with 10:1 cost ratio; recall 0.91 with 50:1 ratio [Serdan, 2025]
  - **Self-Supervised Learning (SSL):** - literature: achieves F1=0.96 with SSL+Hybrid; addresses cold-start by learning from behavior [Al Rafi, 2024; Zhang & Duan, 2025]
  - **Purpose:** State-of-the-art anomaly detection with concept drift handling

- **Tier 4: Sequential Pattern Detection (Optional):** `[Cost: Very High — FLAG: disproportionate to 55-day budget]`
  - **Hybrid CNN-LSTM (ATAD-Net):** - literature: achieved 98.65% accuracy with 8.2ms latency [Abd-Ellatif et al., 2025]
  - **LSTM with Reinforcement Learning (RLFD):** - literature: uniquely detects sequential fraud missed by GBT; achieved 0.549 fraud recall vs 0.226 for GBT [Papanastassiou et al., 2026]
  - **Purpose:** Capture temporal dependencies in spending behavior; essential for detecting sequential anomalies
  - **⚠ Flag:** Tier 4 candidates have implementation complexity disproportionate to the thesis timeline. Retain for literature completeness but prioritize Tier 2-3 for implementation.

- **Selection Criteria:**
  - Highest F1-Score within latency budget (target: F1 ≥ 0.85)
  - Must beat IQR baseline by at least 50% improvement in F1
  - Low false-positive rate (critical for user trust)
  - Model size must be deployable (<200MB for mobile-first constraints) [Hall, 2025]
  - Cost-sensitive optimization: `τ* = argmin C(τ)` where `C(τ) = C_FN·FN(τ) + C_FP·FP(τ)` [Karthikeyan et al., 2026]

---

## 7. Model Evaluation Plan
*Define the rigorous methodology to compare models fairly, without predefining the outcomes.*

- **Primary Evaluation Metric:**
  - **F1-Score** (anomalous class) - literature: balances precision (avoid false alarms) and recall (catch anomalies) [Kalideen, 2025]
  - **Justification:** F1 provides balanced view of performance on minority class

- **Secondary Metrics:**
  - **Precision** - literature: users are sensitive to false positives; 0.98 precision achieved in fraud detection [Al Rafi, 2024]
  - **Recall** - literature: prioritize catching anomalies; 0.886-0.918 recall achieved [Sahraoui & Zari, 2025]
  - **AUC-ROC** - literature: threshold-independent measure; 0.97 achieved by autoencoder [Fariha et al., 2025]
  - **AUC-PR** - literature: more informative for highly imbalanced data [George et al., 2023]
  - **False Positive Rate** - literature: 0.041 achieved by TA-IFDC [Huang A. et al., 2025]
  - **Inference time** (P95 latency) - literature: <10-100ms for real-time [Ahmed et al., 2025]
  - **Cost Savings** - literature: cost-sensitive optimization should minimize `C = c_fp × FP + c_fn × FN` [Karthikeyan et al., 2026]
  - **Early Detection Rate** - literature: 0.726 within first two quarters [Zhang & Duan, 2025]

- **Validation Strategy:**
  - **Stratified K-Fold (k=5)** to maintain anomaly ratio
  - **Time-series split** for temporal generalization (train: first 70%, validate: next 15%, test: last 15%)
  - **Out-of-time validation** - literature: out-of-time validation (OR=2.83) is a strong predictor of reliable performance [Moury, 2026]
  - **Contamination parameter tuning** to adjust sensitivity (contamination rate: 1-5%)

- **Confusion Matrix Analysis:**
  - **False positives** (legitimate transactions flagged as anomalies) - critical for user trust; literature: reduced false positives by 30-40% post-deployment with regularization via complaint rates [CITATION NOT FOUND — verify or remove claim]
  - **False negatives** (missed anomalies) - important for catching issues; cost-sensitive optimization improves recall [Serdan, 2025]

- **Range Metrics for Sequential Anomalies:**
  - **Range-AUC** and **Volume Under the Surface (VUS)** - literature: more robust than point metrics for subsequence anomalies; point-adjust inflates F-measure by 27-31% [Boniol et al., 2024; Zhang et al., 2023]

---

## 8. Optimization Strategy (Tuning & Ensembling)
*Define the hyperparameter landscape and optimization budget.*

- **Tuning Candidates:**
  - **Grid Search** (exhaustive, but limited to small search spaces)
  - **Random Search** (preferred for larger hyperparameter spaces)
  - **Bayesian Optimization** (if compute budget allows)

- **Hyperparameter Space for Isolation Forest:**
  - `n_estimators`: [100, 200, 300] - literature: higher estimators improve performance [Zhong, 2025]
  - `max_samples`: ['auto', 0.5, 0.8]
  - `contamination`: [0.01, 0.05, 0.1, 0.15] (auto or float)
  - `bootstrap`: [True, False]
  - `random_state`: Fixed for reproducibility

- **Hyperparameter Space for Autoencoder:**
  - `n_layers`: [2, 3, 4]
  - `n_units`: [32, 64, 128]
  - `dropout`: [0.1, 0.2, 0.3]
  - `learning_rate`: [0.001, 0.01]
  - `activation`: ['relu', 'tanh']

- **Hyperparameter Space for XGBoost/LightGBM:**
  - `n_estimators`: [100, 200, 300] - literature: LightGBM achieved 0.981 ROC-AUC with 0.58ms latency [Shaha & Gavekar, 2025]
  - `max_depth`: [3, 5, 7, 10]
  - `learning_rate`: [0.01, 0.1, 0.3]
  - `class_weight`: ['balanced', None]
  - `scale_pos_weight`: [1, 2, 5, 10] (for handling imbalance)

- **Threshold Optimization:**
  - **Cost-sensitive threshold:** `τ* = argmin C(τ)` where `C(τ) = C_FN·FN(τ) + C_FP·FP(τ)` [Karthikeyan et al., 2026]
  - **Dynamic calibration:** `θ_k = (1-λ)θ_{k-1} + λ Quantile_β(𝒮_k)` [Huang A. et al., 2025]
  - **Adaptive threshold controller:** `h_t = h_0 · (1 + λ · ν_t)` adjusts sensitivity based on stream volatility [Yashwanth et al., 2023]

- **Ensemble Candidates:**
  - **Weighted averaging** of Isolation Forest and Autoencoder
  - **Voting ensemble** of multiple classifiers - literature: Voting Ensemble achieved highest R² (78.11%) [Thakur & Jadhav, 2025]
  - **Stacking:** Logistic regression meta-model on Isolation Forest + XGBoost + Autoencoder outputs
  - **Hybrid routing:** GRA prefilter reduces heavy model invocations by 40-70% [Sandström, 2024; Adlermann, 2024]
  - **Mixture-of-Experts:** Achieved 96.5% recall with <1.6% false positives [Islam et al., 2025]
  - Tested only if single model plateaus below KPI

- **Cost-Sensitive Optimization:**
  - **Cost ratio exploration:** 10:1, 50:1, 100:1 (FN:FP cost ratio)
  - Literature: CS-DNN with 50:1 ratio achieved recall 0.91 but precision dropped to 0.65 [Serdan, 2025]
  - **Optimal balance:** Determine cost ratio that achieves F1 ≥ 0.85

---

## 9. Deployment Strategy & Module Packaging
*Define how the finished model will be delivered as a module, without choosing the specific cloud provider.*

- **Interface Requirement:**
  - Python function: `predict(input: Dict) -> Dict`
  - Must strictly adhere to Section 0 contract
  - Async support for concurrent requests
  - Lightweight inference runtime (<500MB RAM) for mobile-first constraints [Hall, 2025]

- **Artifact Delivery:**
  - **Isolation Forest weights:** `.pkl` or `.joblib`
  - **Autoencoder weights:** `.h5` or `.pth` format
  - **XGBoost/LightGBM weights:** `.pkl` or `.joblib`
  - **Preprocessing pipeline:** `.pkl` or `.joblib`
  - **Scaler and feature transformer:** serialized as above
  - **Model version metadata:** training data hash, performance metrics, creation date
  - **Fallback model:** IQR statistical model for cold-start

- **Integration Point:**
  - **Containerized API service** (FastAPI preferred for async support) - literature: inference latency critical for user experience [Hall, 2025]
  - Called by mobile app via REST API
  - Dependency injection for model loading at startup
  - **Lightweight screening at edge** - GRA prefilter reduces heavy model invocations [Adlermann, 2024]

- **Deployment Candidates:**
  - **Docker container** (preferred for reproducibility)
  - **Serverless function** (if latency requirements can be met)
  - **Decision:** Docker on dedicated instance for consistent <300ms latency
  - **Mobile-first:** Consider exporting LightGBM model to ONNX or TensorFlow Lite for on-device inference [Hall, 2025]

---

## 10. Model Monitoring & Lifecycle Plan
*Define how we will know the module is degrading over time, without specifying the monitoring tool.*

- **Metrics to Track Post-Deployment:**
  - *Input Drift:* Monitor distribution changes in transaction amounts and categories - literature: use Population Stability Index (PSI) with threshold 10% [CITATION NOT FOUND — verify or remove claim]
  - *Output Drift:* Monitor anomaly flag rate (should remain stable around contamination rate)
  - *Performance:* Compare flagged anomalies to user feedback (whitelist additions)
  - *Precision/Recall:* Track user corrections (whitelist = false positive) and missed anomalies

- **Drift Detection Methods:**
  - **ADWIN** (Adaptive WINdowing) - literature: with incremental learning dominates on gradual drift [CITATION NOT FOUND — verify or remove claim]
  - **DDM** (Drift Detection Method) - literature: reacts fastest to abrupt shifts [CITATION NOT FOUND — verify or remove claim]
  - **EDDM** - literature: captures all critical drifts [CITATION NOT FOUND — verify or remove claim]
  - **CUSUM** - literature: for abrupt drift detection [Yashwanth et al., 2023]
  - **Hybrid detection:** EDDM + ADWIN combined - literature: achieved 100% drift detection rate [CITATION NOT FOUND — verify or remove claim]
  - **Dual detection:** CUSUM for abrupt + moving average for gradual - literature: detection delay 31.2 instances for abrupt, 64.8 for gradual [Yashwanth et al., 2023]
  - **KS Test:** Compares distributions - literature: identified 91.3% of significant changes within 18ms [CITATION NOT FOUND — verify or remove claim]
  - **Embedding cosine distance:** `Drift_rep = (1/k) Σ (1 - (h_t · h_{k-d})/(||h_t|| ||h_{k-d}||))` [Martin et al., 2023]

- **Retraining Triggers:**
  - Anomaly flag rate changes by >2x (significant drift)
  - User whitelist additions spike (high false positives) - literature: false positives reduced by 30-40% post-deployment with regularization [CITATION NOT FOUND — verify or remove claim]
  - Model performance drops below threshold (Precision < 0.80 or Recall < 0.60)
  - Rolling F1 < 0.75 for 3 consecutive weeks
  - Significant input drift detected (PSI > 0.1)
  - **Severity-aware retraining:** Quantile transformation reduces unnecessary retraining - literature: 93% of features exhibited significant drift [Shakhovska & Pukach, 2025]

- **Adaptation Strategies:**
  - **Sliding-window retraining** - literature: improves accuracy by up to 5pp during volatile periods [CITATION NOT FOUND — verify or remove claim]
  - **Parameter updates** - literature: faster than structural updates [Xiang et al., 2023]
  - **Model reuse pool** - literature: Pool ICM reduced retraining events by up to 94% [Eliades & Papadopoulos, 2025]
  - **Adaptive buffering** - literature: retains only drifted batches for retraining; 0.80 accuracy vs FIFO (0.74) and No-Buffer (0.69) under recurring drift [CITATION NOT FOUND — verify or remove claim]
  - **Online adaptive RNN** - literature: effective for load forecasting under concept drift [Xiang et al., 2023]

- **Feedback Loop:**
  - User whitelist additions signal false positives → improve model
  - User marking transactions as normal (if anomaly reported) → feedback for model improvement
  - Monthly retraining with latest transaction data + user feedback
  - **Cost-Benefit Framework:** `S = α·Acc − β·Lat − γ·Cost − δ·FAR` - literature: no single configuration dominates across all regimes [CITATION NOT FOUND — verify or remove claim]

- **Evaluation Metrics for Drift:**
  - **OldTransfer** (knowledge preservation) - literature: GEM achieved 0.1785 OldTransfer [Pai et al., 2024]
  - **NewTransfer** (adaptation to new data) - literature: critical for measuring adaptation performance [Pai et al., 2024]
  - **Backward Transfer (BWT)** - measures forgetting [Martin et al., 2023]

---

## 11. Timeline, Dependencies, and Decision Gates
*Break down the 10 phases into a realistic schedule for the thesis project.*

### Shared Data Phase (All Modules)
| Phase | Deliverable | Estimated Duration | Success Gate (Go/No-Go) |
| :--- | :--- | :--- | :--- |
| 1–3 (Problem + Data) | Problem statement, cleaned dataset, synthetic personas | 16 days | Persona set passes expert + distributional validation; minimum 4,000 labeled personas for PFP, 12,000 personas for Forecaster/Anomaly; archetypes validated against BSP CFS (see `bsp-fies-crosswalk.md`) |

### Anomaly Detector Module (Picks up from shared Data)
| Phase | Deliverable | Estimated Duration | Success Gate (Go/No-Go) |
| :--- | :--- | :--- | :--- |
| 4 & 5 (EDA & Features) | Feature matrix ready; EDA report | 7 days | EDA confirms anomaly patterns; feature importance identified; temporal patterns validated |
| 6 & 7 (Modeling & Eval) | Baseline + Intermediate results | 14 days | IQR baseline established; Isolation Forest beats IQR by >50% F1 improvement |
| 8 (Optimization) | Final optimized model | 10 days | Final model meets KPI: F1 ≥ 0.85, Recall ≥ 0.85, FPR ≤ 0.05 [Huang A. et al., 2025; Al Rafi, 2024] |
| 9 & 10 (Deploy & Monitor) | Packaged module + monitoring setup | 10 days | Integration test passes; drift detection (ADWIN/EDDM) configured; monitoring metrics configured |

**Anomaly Detector Module Duration: 41 days (from feature engineering onward)**

---

## 12. Cold-Start Hypothesis
*Define how the module handles new users with limited or no historical data.*

- **Core Hypothesis:** New users can be detected for anomalies using behavioral biometrics, synthetic data generation, transfer learning, or unsupervised baselines until sufficient transaction history is accumulated.

- **Strategies from Literature:**

| Strategy | Mechanism | Evidence | Cost |
|----------|-----------|----------|------|
| **Self-Supervised Learning** | Pretext tasks on unlabeled data; learns from behavior patterns | SSL achieved F1=0.96; AUC 0.91 for credit risk vs XGBoost 0.84 [Al Rafi, 2024; Yachamaneni et al., 2025] | High |
| **Behavioral Biometrics** | Dynamic trust-weighted embeddings identify users after 14 seconds of interaction | Achieves 97.6% user identification accuracy using interaction patterns [Islam et al., 2025] | Very High — FLAG: disproportionate to thesis scope |
| **Synthetic Data Generation** | GANs, SMOTE, ADASYN, or LLM-based augmentation for cold-start users | PAnDA achieved 15-37% performance gain on cold-start tasks [Du et al., 2025]; TVAE achieved 89% ROC AUC vs 52-59% for alternatives [CITATION NOT FOUND — verify or remove claim] | Medium |
| **Transfer Learning** | Knowledge Concept Graph (KCG) enables zero-shot profiling | TechCD achieved 56.73% ACC with out-domain data [Gao et al., 2023] | High |
| **Unsupervised Baselines** | Isolation Forest, One-Class SVM, autoencoders require only normal data | TA-IFDC achieved F1=0.927 without labels [Huang A. et al., 2025] | Medium |
| **Federated Learning** | Collaborative training across institutions without sharing raw data | SCAFFOLD achieves 84.7% accuracy with robust non-IID performance [CITATION NOT FOUND — verify or remove claim] | Very High — FLAG: infrastructure requirement disproportionate to thesis scope |
| **Active Learning** | Iterative labeling of uncertain cases for exploration | Mainstream approach for cold-start and label scarcity [Zhou & He, 2023] | Medium |

- **Combined Approach:**
  - **Immediate:** Unsupervised Isolation Forest + autoencoder using available data (even minimal)
  - **First 14 seconds:** Behavioral biometrics profile with trust-weighted embeddings [Islam et al., 2025]
  - **First 30 days:** Hybrid ensemble with synthetic data augmentation
  - **After 3 months:** Full ensemble model with personalized features
  - **Dynamic update:** User profile continuously updated as actual data arrives
  - **Fallback:** IQR rule-based detection + SHAP explanations for transparency

- **Expected Performance:**
  - **Day 0:** F1 ≥ 0.60 (unsupervised baseline)
  - **Day 14:** F1 ≥ 0.70 (behavioral biometrics + synthetic data)
  - **Day 30:** F1 ≥ 0.75 (hybrid ensemble + active learning)
  - **Day 90:** F1 ≥ 0.85 (full ensemble with personalized features)

---

## 13. Concept Drift & Assumptions
*Define how the module handles changing user behavior and economic conditions over time.*

- **Core Assumption:** Non-stationarity is the norm in financial data, not the exception [Cabral et al., 2026]. Static models degrade over time and require continuous adaptation. 91% of ML models experience performance decay in production, sometimes within days [CITATION NOT FOUND — verify or remove claim].

- **Types of Drift to Handle:**

| Type | Description | Detection Method | Evidence |
|------|-------------|------------------|----------|
| **Abrupt** | Sudden distribution change (e.g., pandemic, job loss) | CUSUM, DDM, Page-Hinkley | Detection delay: 31.2 instances [Yashwanth et al., 2023]; ARF+HT achieves 0.94 AUC [CITATION NOT FOUND — verify or remove claim] |
| **Gradual** | Slow change over time (e.g., lifestyle changes) | Moving average, ADWIN | Detection delay: 64.8 instances [Yashwanth et al., 2023] |
| **Recurrent** | Past distributions reappear (e.g., seasonal patterns) | Pool ICM, exchangeability testing | Pool ICM reduced retraining events by up to 94% [Eliades & Papadopoulos, 2025] |
| **Incremental** | Slow, continuous shift | EWMA, sliding window | Most challenging; neither adaptation nor lifelong learning performs well [Pai et al., 2024] |

- **Detection Methods:**

| Method | Description | Performance |
|--------|-------------|-------------|
| **ADWIN** | Adaptive windowing for streaming data | Handles gradual drift effectively; with incremental learning dominates on gradual drift [CITATION NOT FOUND — verify or remove claim] |
| **DDM** | Drift Detection Method based on error rates | Reacts fastest to abrupt shifts [CITATION NOT FOUND — verify or remove claim] |
| **EDDM** | Early Drift Detection Method | Captures all critical drifts [CITATION NOT FOUND — verify or remove claim] |
| **Hybrid** | EDDM + ADWIN combined | 100% drift detection rate [CITATION NOT FOUND — verify or remove claim] |
| **KS Test** | Kolmogorov-Smirnov compares distributions | Identified 91.3% of significant changes within 18ms [CITATION NOT FOUND — verify or remove claim] |
| **Embedding cosine** | Distance between hidden representations | `Drift_rep = (1/k) Σ (1 - (h_t · h_{k-d})/(||h_t|| ||h_{k-d}||))` [Martin et al., 2023] |

- **Adaptation Strategies:**

| Strategy | Effectiveness | Evidence |
|----------|---------------|----------|
| **Sliding-window retraining** | Improves AUC by up to 5pp during volatile periods | [CITATION NOT FOUND — verify or remove claim] |
| **Parameter updates** | Faster than structural updates; common approach | [Xiang et al., 2023] |
| **Model reuse pool** | Pool ICM reduced retraining events by 94% with <3% accuracy loss | [Eliades & Papadopoulos, 2025] |
| **Adaptive buffering** | 0.80 accuracy vs FIFO (0.74) and No-Buffer (0.69) under recurring drift | [CITATION NOT FOUND — verify or remove claim] |
| **Severity-aware** | Quantile transformation reduced KS statistic from 0.0559 to 0.0072 | [Shakhovska & Pukach, 2025] |
| **Batch training** | Outperforms ignoring drift; robust performance across diverse streams | [Pereira & Da Silva, 2025] |
| **Online adaptive RNN** | Effective for load forecasting under concept drift | [Xiang et al., 2023] |

- **Key Assumptions Documented:**

| Assumption | Evidence |
|------------|----------|
| **Drift is inevitable** | "Non-stationarity is the norm in financial data, not the exception" [Cabral et al., 2026] |
| **Static models fail** | Traditional models degrade within 12-18 months; retraining alone insufficient [Chikoore et al., 2026] |
| **Drift patterns recur** | "Recurrent concept drift, where past data distributions reappear, is common" [Eliades & Papadopoulos, 2025] |
| **Data drift ≠ Model drift** | Data drift (covariate shift) can be addressed with quantile transformation; concept drift requires model retraining [Shakhovska & Pukach, 2025] |
| **Statistical tests detect drift** | JS Divergence, KS tests, Wasserstein distance widely used, but pooled analysis can mask subgroup-level drift [Shakhovska & Pukach, 2025] |
| **Seasonal patterns stable** | Multi-SARIMA assumes two seasonal periods exist; performance degrades if patterns drift [Williams et al., 2023] |
| **Subgroup-level drift** | Subgroup-level analysis can reveal drift masked by pooled aggregation; 44 of 45 areas showed significant drift despite overall no drift [Shakhovska & Pukach, 2025] |

- **Key Assumption (synthetic-to-real gap):** All KPI figures in this document are measured on synthetic personas with injected anomalies. Generalization to real users and to real (rather than synthetically injected) anomalous behavior is untested until real prototype-user data is available (Section 2, Secondary source) and is treated as an explicit limitation of this design, not a guarantee.
- **Key Assumption (injected feature realism):** Both the behavioral features and the anomalies themselves are synthetically injected per RRL citations and financial-expert judgment, not observed directly in FIES. Any reported detection performance is only as trustworthy as these injection rules; this is documented as a threat to validity rather than assumed away.
- **Key Assumption (BSP-FIES crosswalk):** Archetype segmentation is sourced from the BSP Consumer Finance Report, while granular transaction data is sourced from PSA FIES microdata — two surveys with different units of analysis and sampling frames. Any mismatch between a BSP-defined archetype and the FIES-derived income/expense profile assigned to it is a threat to validity, not a confirmed equivalence; see `bsp-fies-crosswalk.md`.

- **Recommended Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                     Data Ingestion                       │
│  Transactions • User Profiles • Temporal Features        │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Feature Engineering                     │
│  Behavioral (Frequency, Amount Deviations) • Temporal   │
│  Relational (Graph) • Missingness Indicators            │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Hybrid Detection Pipeline                   │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────┐  │
│  │   Unsupervised │  │   Supervised  │  │  Ensemble  │  │
│  │ Autoencoder   │  │  XGBoost/RF   │  │    Fusion  │  │
│  │ Isolation     │  │  Cost-Sens.   │  │   (GRA)    │  │
│  └───────────────┘  └───────────────┘  └────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Threshold Management                   │
│  • Dynamic threshold (sliding window + Bayesian CP)     │
│  • Adaptive threshold controller: h_t = h_0·(1+λ·ν_t)  │
│  • Per-user threshold calibration                       │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Drift Monitoring                       │
│  • KS Test on feature distributions                     │
│  • ADWIN/EDDM for stream-level drift detection          │
│  • Subgroup-level drift analysis                        │
└─────────────────────────────────────────────────────────┘
```

- **Critical Design Principles:**
  1. **Assume concept drift is inevitable** → Build adaptive mechanisms [Cabral et al., 2026]
  2. **Prioritize recall over accuracy** → Cost of missed fraud > false alarms [Karthikeyan et al., 2026]
  3. **Use behavioral indicators** → Transaction frequency, device/location consistency [Shaha & Gavekar, 2025]
  4. **Implement adaptive thresholds** → Dynamic adjustment based on performance feedback [Huang A. et al., 2025]
  5. **Address cold-start early** → Synthetic data via TVAE or LLM augmentation [CITATION NOT FOUND — verify or remove claim; Du et al., 2025]
  6. **Monitor for concept drift continuously** → Dual ADWIN+EDDM strategy [CITATION NOT FOUND — verify or remove claim]
  7. **Design for 30-day data windows** - longer windows don't improve performance [Heirene et al., 2026]