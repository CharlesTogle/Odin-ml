# Synthetic Data Injection Rules

**Document Version:** v1.1
**Author(s):** Guevarra
**Date:** 2026-07-26
**Status:** Draft — Pending SME Validation
**Purpose:** Specify exact rules for generating behavioral features from FIES NCR aggregates
**Data Sources:** FIES 2023 (financial numerical baselines) + BSP CFS 2021 (behavioral/attitudinal patterns)

---

## 1. Overview

This document defines the deterministic rules for transforming PSA 2023 FIES NCR microdata into labeled synthetic personas with transaction histories. The output feeds all three Odin ML models:

- **PFP Classifier:** Questionnaire answers + transaction history → PFP label
- **Forecaster:** Transaction history → expense forecasts
- **Anomaly Detector:** Transaction history (with injected anomalies) → anomaly flags

### 1.1 Input

- **Primary Source:** PSA 2023 FIES NCR subset (41,380 households, 87 columns)
- **Secondary Source:** BSP 2021 Consumer Finance Survey (behavioral/attitudinal patterns)
- **Format:** CSV with ~40 columns
- **Scope:** Metro Manila / National Capital Region only
- **Key columns:** Income totals, expense totals by category, family size, PSU, weights, urban/rural, per capita income, decile rankings

### 1.2 Output

- **12,000 synthetic users** (12 archetypes × 1,000 each)
- **365 days of transaction history** per user
- **8 PFP labels** per user (Stable/Obligated/Tolerant, Stable/Obligated/At-Risk, Stable/Flexible/Tolerant, Stable/Flexible/At-Risk, Variable/Obligated/Tolerant, Variable/Obligated/At-Risk, Variable/Flexible/Tolerant, Variable/Flexible/At-Risk)
- **Questionnaire responses** per user (for QUESTIONNAIRE mode training)

### 1.3 Data Source Calibration

**FIES NCR 2023 Financial Numerical Baselines:**
- Average household income: ₱502,607/year (₱41,884/month); Median: ₱397,866/year (₱33,156/month)
- Average household size: 4.03 members
- Expense ratios: Food 41.1%, Housing/Water 28.9%, Transport 5.9%, Communication 3.9%, Insurance 2.6%, Health 2.4%, Education 1.7%
- Savings rate: Mean 20.4%, Median 19.2%; 93.6% have positive savings
- Income sources: Regular salary 88.2%, Entrepreneurial 32.7%, Remittances 22.1%, Seasonal 19.6%, Pension 17.4%

**BSP CFS 2021 Behavioral/Attitudinal Patterns:**
- Deposit account ownership: 35.3% of households
- Emergency fund (3+ months expenses): Only 2.0% of households
- Debt/loan ownership: 29.3% of households
- Credit card ownership: 1.1% of households
- Residential property ownership: 69.9% of households
- Average net worth: ₱435,346
- Financial risk tolerance: 6.3/10 (self-reported)
- Digital financial services: E-money usage 24.3%, Mobile banking 17.7%
- Past-due bills: 16.4% of households

---

## 2. FIES NCR Field Mapping

### 2.1 Available Fields from FIES Dictionary

| FIES Column | Description | Use in Persona Generation |
|-------------|-------------|---------------------------|
| `W_REGN` | Region | Filter: NCR only |
| `W_PROV` | Province | Geographic sub-grouping |
| `SEQ_NO` | Household ID | Unique identifier |
| `FSIZE` | Average Family Size | Dependency ratio calculation |
| `REG_SAL` | Salaries/Wages from Regular Employment | Primary income stability signal |
| `SEASON_SAL` | Salaries/Wages from Seasonal Employment | Income volatility signal |
| `WAGES` | Total Salaries/Wages | Combined employment income |
| `NETSHARE` | Net Share of Crops/Fruits | Agricultural income component |
| `CASH_ABROAD` | Cash Receipts from Abroad | Remittance component |
| `CASH_DOMESTIC` | Cash Receipts from Domestic Source | Secondary income |
| `RENTALS_REC` | Rentals Received | Passive income |
| `INTEREST` | Interest | Investment income |
| `PENSION` | Pension | Retirement income |
| `DIVIDENDS` | Dividends | Investment income |
| `OTHER_SOURCE` | Other Income NEC | Miscellaneous income |
| `NET_RECEIPT` | Family Sustenance Activities | Informal income |
| `EAINC` | Total Income from Entrepreneurial Activities | Business income |
| `LOSSES` | Losses from Entrepreneurial Activities | Business losses |
| `BREAD` through `OTHER_NON_ALCOHOL` | Food expenditure categories | Essential spending |
| `ALCOHOL` | Alcoholic Beverages | Discretionary spending |
| `CLOTH` | Clothing and Footwear | Discretionary spending |
| `HOUSING_WATER` | Housing, Water, Electricity, Gas | Obligatory spending |
| `HEALTH` | Health | Obligatory spending |
| `TRANSPORT` | Transport | Essential spending |
| `COMMUNICATION` | Communication | Essential spending |
| `RECREATION` | Recreation and Culture | Discretionary spending |
| `EDUCATION` | Education | Obligatory spending |
| `INSURANCE` | Insurance | Obligatory spending |
| `MISCELLANEOUS` | Miscellaneous Goods and Services | Mixed spending |
| `DURABLE` | Durable Furniture and Equipment | Discretionary spending |
| `OCCASION` | Special Family Occasion | Cultural spending |
| `OTHER_EXPENDITURE` | Other Expenditure | Miscellaneous |
| `FOOD` | Total Food Expenditures | Essential spending |
| `NFOOD` | Total Non-Food Expenditure | Mixed spending |
| `TOINC` | Total Income | Primary income metric |
| `TOTEX` | Total Expenditure | Primary expense metric |
| `PERCAPITA` | Per Capita Income | Income normalization |
| `NPCINC` | Per Capita Income Decile (National) | Income grouping |
| `RPCINC` | Per Capita Income Decile (Region) | Regional income grouping |
| `URB` | Urban/Rural | Geographic context |
| `RFACT` | Family/Household Weight | Sampling weight |
| `MEM_RFACT` | Population Weight | Population normalization |

### 2.2 Derived Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Income Stability (CV)** | `σ(monthly_income) / μ(monthly_income)` | Primary PFP dimension |
| **Obligation Ratio** | `(Essential + Obligatory) / Total Expenditure` | Secondary PFP dimension |
| **Savings Rate** | `(TOINC - TOTEX) / TOINC` | Financial health indicator |
| **Engel Coefficient** | `FOOD / TOTEX` | Spending composition |
| **Dependency Ratio** | `(FSIZE - 1) / earners` | Household burden |
| **Income Diversification** | `count(income_sources > 0) / total_sources` | Income stability signal |

---

## 3. Persona Archetype Definitions

### 3.1 PFP-to-FIES NCR Decile Mapping

| PFP Cell | CV Range | Obligated Ratio | Runway | Likely FIES NCR Decile | Income Range (Monthly) | FIES NCR Reference |
|----------|----------|-----------------|--------|------------------------|------------------------|-------------------|
| **Stable/Obligated/Tolerant** | < 0.5 | > 0.6 | ≥ 3 mo | D3–D6 | ₱25,000 – ₱55,000 | Median: ₱33,156/mo |
| **Stable/Obligated/At-Risk** | < 0.5 | > 0.6 | < 3 mo | D3–D5 | ₱20,000 – ₱35,000 | Mean: ₱41,884/mo |
| **Stable/Flexible/Tolerant** | < 0.5 | ≤ 0.6 | ≥ 3 mo | D6–D9 | ₱40,000 – ₱70,000 | P75: ₱49,075/mo |
| **Stable/Flexible/At-Risk** | < 0.5 | ≤ 0.6 | < 3 mo | D5–D7 | ₱25,000 – ₱50,000 | P25: ₱23,313/mo |
| **Variable/Obligated/Tolerant** | ≥ 0.5 | > 0.6 | ≥ 3 mo | D4–D7 | ₱20,000 – ₱60,000 | Mean: ₱41,884/mo |
| **Variable/Obligated/At-Risk** | ≥ 0.5 | > 0.6 | < 3 mo | D1–D4 | ₱5,000 – ₱40,000 | Min: varies |
| **Variable/Flexible/Tolerant** | ≥ 0.5 | ≤ 0.6 | ≥ 3 mo | D5–D8 | ₱15,000 – ₱45,000 | Median: ₱33,156/mo |
| **Variable/Flexible/At-Risk** | ≥ 0.5 | ≤ 0.6 | < 3 mo | D1–D4 | ₱5,000 – ₱20,000 | P25: ₱23,313/mo |

> **Note:** Income ranges calibrated to FIES NCR 2023 decile distribution. The 8-class system adds a Tolerance dimension (runway months) not present in the earlier 4-class version. FIES NCR reference values: Mean ₱41,884/mo, Median ₱33,156/mo, P25 ₱23,313/mo, P75 ₱49,075/mo.

### 3.2 Employment Type Mapping

| Employment Type | FIES Signal | Stability Signal | Income Stability |
|-----------------|-------------|------------------|------------------|
| **Regular Salaried** | `REG_SAL > 0`, `SEASON_SAL = 0` | Stable (CV < 0.5) | High |
| **Seasonal Worker** | `SEASON_SAL > 0` | Variable (CV ≥ 0.5) | Low |
| **Freelance/Gig** | `NET_RECEIPT > 0`, `REG_SAL = 0` | Variable (CV ≥ 0.5) | Low |
| **Entrepreneur** | `EAINC > 0` | Variable (CV ≥ 0.5) | Medium-Low |
| **Mixed Earner** | Multiple income sources > 0 | Depends on composition | Medium |
| **BPO Worker** | Inferred from income level + urban | Variable (CV ≥ 0.5) | Medium-High |
| **OFW Household** | `CASH_ABROAD > 0` | Variable (CV ≥ 0.5) | Medium |

> **Note:** Obligation and Tolerance are computed from expense ratios and savings runway, not from employment type directly. The full 8-class label is computed via `generate_personas.py:_compute_pfp_label()` using 3 independent dimensions.

### 3.3 12 Persona Archetypes (A–L)

The 2×2×2 PFP matrix (Stability × Obligation × Tolerance) produces 8 core cells.
Four additional edge-case archetypes (I–L) cover high-frequency cells with
boundary conditions. Each archetype maps to a specific cell and employment context.
Financial parameters calibrated to FIES NCR 2023; behavioral patterns calibrated to BSP CFS 2021.

| Archetype | PFP Cell | Employment | Income CV | Obligation Ratio | Runway (mo) | Household | Income Range (Monthly) | Key Characteristics |
|-----------|----------|------------|-----------|------------------|-------------|-----------|------------------------|---------------------|
| **A** | Stable/Obligated/Tolerant | Full-time BPO | 0.10 | 0.70 | 5.0 | 1–2 | ₱35,000–₱45,000 | Consistent salary, moderate obligations, healthy emergency fund (CFS: top 2% for runway) |
| **B** | Stable/Obligated/At-Risk | Full-time Manufacturing | 0.15 | 0.85 | 0.5 | 3–4 | ₱25,000–₱35,000 | Heavy family obligations, no savings buffer (CFS: typical debt profile) |
| **C** | Stable/Flexible/Tolerant | Full-time Tech | 0.08 | 0.45 | 9.0 | 1 | ₱55,000–₱70,000 | Low obligations, strong savings, lives with parents (CFS: top 2% for financial health) |
| **D** | Stable/Flexible/At-Risk | Full-time Government | 0.12 | 0.50 | 1.5 | 1–2 | ₱30,000–₱40,000 | Consistent salary, spends discretionary freely (CFS: typical digital services user) |
| **E** | Variable/Obligated/Tolerant | Freelance | 0.70 | 0.75 | 4.0 | 1–2 | ₱25,000–₱60,000 | Irregular income, adequate savings buffer (CFS: above-average financial risk tolerance) |
| **F** | Variable/Obligated/At-Risk | Informal/Contract | 0.65 | 0.80 | 0.3 | 2–3 | ₱18,000–₱40,000 | Paycheck-to-paycheck, income gaps (CFS: below-average deposit ownership) |
| **G** | Variable/Flexible/Tolerant | Freelance Writer/VA | 0.60 | 0.40 | 7.0 | 1–2 | ₱20,000–₱45,000 | Irregular income, built reserves in high months (CFS: top 2% for financial resilience) |
| **H** | Variable/Flexible/At-Risk | Gig Worker | 0.80 | 0.45 | 1.0 | 1–2 | ₱8,000–₱20,000 | Tricycle driver/vendor, no emergency fund (CFS: typical low-income profile) |
| **I** | Variable/Obligated/At-Risk | Full-time (re-employed) | 0.70 | 0.78 | 0.8 | 2–3 | ₱20,000–₱35,000 | Recovering from job loss, depleted savings (CFS: transitional financial shock) |
| **J** | Variable/Flexible/At-Risk | Part-time Sales | 0.55 | 0.55 | 2.5 | 1–2 | ₱15,000–₱30,000 | Low obligations, borderline tolerance (CFS: typical young adult profile) |
| **K** | Stable/Obligated/Tolerant | Full-time Telecom | 0.10 | 0.65 | 4.0 | 3–4 | ₱45,000–₱55,000 | Obligations near threshold, edge case (CFS: mortgage + car loan typical) |
| **L** | Stable/Flexible/At-Risk | Full-time Marketing | 0.10 | 0.50 | 1.0 | 1 | ₱35,000–₱50,000 | No savings habit despite stable income (CFS: typical discretionary spender) |

> **Note:** Income Stability is derived from income *consistency* (coefficient of variation), not income *magnitude*. Thresholds: CV < 0.5 = Stable, ratio > 0.6 = Obligated, runway ≥ 3 months = Tolerant. See `generate_personas.py` for implementation. Archetypes I–L are edge cases that exercise boundary conditions within their cells. Income ranges calibrated to FIES NCR 2023 decile distribution; behavioral patterns calibrated to BSP CFS 2021.

---

## 4. Feature Injection Rules

### 4.1 Income Generation

**Rule 1: Monthly Income Sampling**
```python
# For each persona archetype:
monthly_income = np.random.lognormal(
    mu_ln=log(archetype_mean_income),
    sigma_ln=archetype_income_cv
)

# Clamp to decile range
monthly_income = np.clip(monthly_income, decile_min, decile_max)
```

**Rule 2: Income Stability Injection**
```python
# For Stable profiles (CV < 0.5):
income_noise = np.random.normal(0, 0.05 * monthly_income)  # ±5% noise
monthly_income_stable = monthly_income + income_noise

# For Variable profiles (CV >= 0.5):
income_noise = np.random.normal(0, 0.25 * monthly_income)  # ±25% noise
monthly_income_variable = monthly_income + income_noise

# For Seasonal workers:
seasonal_multiplier = np.random.choice([0.3, 0.5, 0.7, 1.0, 1.2], 
                                        p=[0.1, 0.2, 0.3, 0.3, 0.1])
monthly_income_seasonal = monthly_income * seasonal_multiplier
```

**Rule 3: Payday Timing**
```python
# Regular salaried: 15th and 30th/31st
payday_dates = [15, 30]  # or 31 depending on month

# BPO workers: weekly or bi-weekly (non-standard)
bpo_payday_cycle = np.random.choice([7, 14])  # weekly or bi-weekly

# Freelance: irregular (average 2-3 times per month)
freelance_payday_count = np.random.poisson(lam=2.5)
freelance_payday_dates = sorted(np.random.choice(range(1, 29), 
                                                  size=freelance_payday_count, 
                                                  replace=False))
```

### 4.1.1 Zero-Income Months (Intentional)

Zero-income months are an intentional feature of the synthetic data, modeling real-world income gaps for irregular and project-based employment types. The following income patterns in `generate_transactions.py` produce zero-income months:

| Income Pattern | Frequency | Zero-Income Probability | Archetypes Affected |
|---------------|-----------|------------------------|---------------------|
| `irregular` | Monthly with gaps | 15% per month | Construction Worker, Online Seller, Freelance Developer, College Student |
| `project_based` | Chunk-based | 50% per month | (No current archetype — pattern exists for future use) |
| `unemployed` | None | 100% (all months) | Retired Couple |

This produces approximately 15.4% zero-income rows in the processed dataset, which is consistent with the expected rate from 4 irregular archetypes × 15% probability. These rows are expected behavior, not a sampling bug.

### 4.2 Expense Generation

**Rule 4: Category Allocation**
```python
# For each persona, compute expense ratios from FIES NCR data:
category_ratios = {
    'food': fies_food_ratio * persona_scale_factor,
    'housing': fies_housing_ratio * persona_scale_factor,
    'transport': fies_transport_ratio * persona_scale_factor,
    'health': fies_health_ratio * persona_scale_factor,
    'education': fies_education_ratio * persona_scale_factor,
    'clothing': fies_clothing_ratio * persona_scale_factor,
    'recreation': fies_recreation_ratio * persona_scale_factor,
    'communication': fies_communication_ratio * persona_scale_factor,
    'insurance': fies_insurance_ratio * persona_scale_factor,
    'miscellaneous': fies_misc_ratio * persona_scale_factor,
}

# Normalize to sum to 1.0
total_ratio = sum(category_ratios.values())
category_ratios = {k: v/total_ratio for k, v in category_ratios.items()}
```

**Rule 5: Obligation Ratio Enforcement**
```python
# Compute obligation ratio:
essential_categories = ['food', 'housing', 'transport', 'health', 'education']
obligatory_categories = ['debt_payment']  # Debt/loan repayments

essential_total = sum(category_ratios[c] for c in essential_categories)
obligatory_total = sum(category_ratios[c] for c in obligatory_categories)

obligation_ratio = (essential_total + obligatory_total) / total_ratio

# Enforce PFP obligation dimension:
if 'Obligated' in pfp:
    obligation_ratio = np.random.uniform(0.65, 0.85)  # High obligation
else:
    obligation_ratio = np.random.uniform(0.30, 0.55)  # Low obligation

# Re-scale categories to match target obligation ratio
scale_factor = (obligation_ratio * total_ratio) / (essential_total + obligatory_total)
for cat in essential_categories + obligatory_categories:
    category_ratios[cat] *= scale_factor
```

**Rule 6: Daily Expense Sampling**
```python
# For each day in 365-day history:
for day in range(365):
    for category in categories:
        # Base daily expense
        daily_base = monthly_income * category_ratios[category] / 30
        
        # Add noise (log-normal for right skew)
        daily_expense = np.random.lognormal(
            mu_ln=log(daily_base),
            sigma_ln=0.3  # Moderate variance
        )
        
        # Apply temporal modifiers (see Rule 7)
        daily_expense *= temporal_modifier
        
        # Record transaction
        transactions.append({
            'date': start_date + timedelta(days=day),
            'category': category,
            'amount': round(daily_expense, 2),
            'type': 'expense'
        })
```

### 4.3 Temporal Pattern Injection

**Rule 7: Payday Spike**
```python
# Days 15-16 and 30-31: spending spike
if day in [15, 16, 30, 31]:
    spike_multiplier = np.random.normal(loc=2.1, scale=0.40)
    spike_multiplier = np.clip(spike_multiplier, 1.5, 3.0)
    
    # Discretionary categories spike harder
    if category in ['recreation', 'clothing', 'miscellaneous']:
        spike_multiplier *= 1.3
    
    daily_expense *= spike_multiplier
```

**Rule 8: Petsa de Peligro (Money-Run-Out Days)**
```python
# Days 10-14 and 26-29: spending suppression
if day in [10, 11, 12, 13, 14, 26, 27, 28, 29]:
    suppression_multiplier = np.random.normal(loc=0.50, scale=0.15)
    suppression_multiplier = np.clip(suppression_multiplier, 0.3, 0.7)
    
    # Essential categories suppressed less
    if category in ['food', 'transport']:
        suppression_multiplier = max(suppression_multiplier, 0.6)
    
    daily_expense *= suppression_multiplier
```

**Rule 9: Christmas / Ber Months**
```python
# September-December: gradual spending escalation
month = current_date.month
if month in [9, 10, 11, 12]:
    christmas_multipliers = {9: 1.1, 10: 1.2, 11: 1.4, 12: 2.2}
    christmas_multiplier = christmas_multipliers[month]
    
    # Add noise
    christmas_multiplier *= np.random.normal(1.0, 0.15)
    
    daily_expense *= christmas_multiplier
```

**Rule 10: 13th Month Pay**
```python
# December 15-24: add 13th month bonus
if current_date.month == 12 and 15 <= current_date.day <= 24:
    # Add one month income to spending pool
    bonus_income = monthly_income
    
    # Spike discretionary spending
    if category in ['recreation', 'clothing', 'food']:
        daily_expense *= np.random.normal(2.0, 0.30)
```

**Rule 11: Semana Santa (Holy Week)**
```python
# March/April (moveable feast): travel and leisure spike
if current_date.month in [3, 4] and is_holy_week(current_date):
    if category == 'recreation':
        daily_expense *= np.random.normal(1.4, 0.20)
    elif category == 'food':
        daily_expense *= 0.8  # Closures reduce food spending
```

**Rule 12: School Enrollment Season**
```python
# June (public) + October (some private): education spike
if current_date.month in [6, 10]:
    if category == 'education':
        daily_expense *= np.random.normal(1.4, 0.15)
    # Financial allocation decreases (funds redirected)
    if category == 'miscellaneous':
        daily_expense *= 0.85
```

**Rule 13: Undas / All Saints Day**
```python
# November 1-2: travel and leisure spike
if current_date.month == 11 and current_date.day in [1, 2]:
    if category == 'recreation':
        daily_expense *= np.random.normal(1.45, 0.20)
    if category == 'food':
        daily_expense *= 0.75  # Fewer local purchases
```

**Rule 14: BPO Payday Variation**
```python
# For BPO workers (~15-20% of personas):
if persona_is_bpo:
    # Weekly or bi-weekly payday instead of 15th/30th
    # This smooths out the sharp 15th/30th spike
    if day % bpo_payday_cycle == 0:
        # Mild spike (not as dramatic as regular payday)
        daily_expense *= np.random.normal(1.5, 0.20)
```

### 4.4 Behavioral Feature Injection

**Rule 15: Spending Timing Relative to Payday**
```python
# Compute days since last payday
days_since_payday = (current_date - last_payday_date).days

# Spending timing feature:
# 0 = payday, 1-3 = post-payday surge, 4-10 = normal, 11-14 = pre-payday suppression
if days_since_payday <= 3:
    spending_timing = 'post_payday_surge'
elif days_since_payday <= 10:
    spending_timing = 'normal'
else:
    spending_timing = 'pre_payday_suppression'

# Inject as feature
persona.features['spending_timing'] = spending_timing
```

**Rule 16: Category Diversity**
```python
# Count distinct categories used per month
categories_used = set(t['category'] for t in monthly_transactions)
category_diversity = len(categories_used)

# Normalize to 0-1 scale
max_possible_categories = 10  # Number of expense categories
category_diversity_score = category_diversity / max_possible_categories

# Inject as feature
persona.features['category_diversity'] = category_diversity_score
```

**Rule 17: Amount Volatility**
```python
# Compute coefficient of variation per category
for category in categories:
    category_amounts = [t['amount'] for t in category_transactions]
    cv = np.std(category_amounts) / np.mean(category_amounts)
    
    # Inject as feature
    persona.features[f'{category}_volatility'] = cv
```

**Rule 18: Inter-Transaction Gaps**
```python
# Compute days between transactions
transaction_dates = sorted(set(t['date'] for t in transactions))
gaps = [(transaction_dates[i+1] - transaction_dates[i]).days 
        for i in range(len(transaction_dates)-1)]

# Average gap
avg_gap = np.mean(gaps)
gap_variance = np.var(gaps)

# Inject as features
persona.features['avg_inter_transaction_gap'] = avg_gap
persona.features['gap_variance'] = gap_variance
```

### 4.5 Label Generation

**Rule 19: PFP Label Assignment**
```python
# Compute CV on monthly total spend over 3-month window
monthly_totals = []
for month in range(3):
    month_start = start_date + timedelta(days=30*month)
    month_end = month_start + timedelta(days=30)
    month_total = sum(t['amount'] for t in transactions 
                      if month_start <= t['date'] < month_end)
    monthly_totals.append(month_total)

cv = np.std(monthly_totals) / np.mean(monthly_totals)

# Compute obligation ratio (essential + obligatory expenses / total expenses)
essential_total = sum(t['amount'] for t in transactions 
                      if t['category'] in essential_categories)
obligatory_total = sum(t['amount'] for t in transactions 
                       if t['category'] == 'debt_payment')
obligation_ratio = (essential_total + obligatory_total) / total_spend

# Compute runway (months of expenses covered by current balance)
balance = sum(t['amount'] for t in transactions if t['type'] == 'income') - total_spend
avg_monthly_expense = total_spend / max(1, n_months)
runway_months = balance / avg_monthly_expense if avg_monthly_expense > 0 else 0.0

# Assign 3-dimension PFP label
stability = "Stable" if cv < STABILITY_CV_THRESHOLD else "Variable"
obligation = "Obligated" if obligation_ratio > OBLIGATION_RATIO_THRESHOLD else "Flexible"
tolerance = "Tolerant" if runway_months >= TOLERANCE_RUNWAY_MONTHS else "At-Risk"
pfp_label = f"{stability}/{obligation}/{tolerance}"

# Inject as label
persona.label = pfp_label
```

**Rule 20: Threshold Calibration**
```python
# Thresholds are provisional team-reasoned defaults, not empirically derived.
# They will be validated against real user transaction data in future iterations.
# See dimension_discovery.py for clustering analysis (Phase 4.5).

# Current thresholds (SME draft, to be refined with real data):
STABILITY_CV_THRESHOLD = 0.5    # Stable if CV < 0.5, Variable if CV >= 0.5
OBLIGATION_RATIO_THRESHOLD = 0.6  # Obligated if ratio > 0.6, Flexible if ratio <= 0.6
TOLERANCE_RUNWAY_MONTHS = 3.0   # Tolerant if runway >= 3 months, At-Risk if runway < 3 months
```

---

## 5. Anomaly Injection Rules

### 5.1 Anomaly Types

| Type | Description | Injection Method | Rate |
|------|-------------|------------------|------|
| **Monetary Spike** | Unusually high transaction amount | Multiply by 3-5x | 1-2% |
| **Category Velocity** | Unusual frequency in a category | Insert 3-5 extra transactions | 0.5-1% |
| **Temporal Deviation** | Transaction at unusual time | Shift timestamp to 2-5 AM | 0.5-1% |
| **Merchant Novelty** | Unusual payee/category mismatch | Recategorize transaction | 0.5-1% |
| **Budget Overage** | Exceeds category budget | Compute from budget allocation | 2-3% |

### 5.2 Anomaly Injection Code

```python
def inject_anomalies(transactions, anomaly_rate=0.03):
    """
    Inject anomalies into transaction history.
    
    Target: 3% anomaly rate (within 1-5% literature range)
    """
    anomalies = []
    total_transactions = len(transactions)
    target_anomalies = int(total_transactions * anomaly_rate)
    
    for _ in range(target_anomalies):
        # Select random transaction
        idx = np.random.randint(0, total_transactions)
        transaction = transactions[idx].copy()
        
        # Select anomaly type
        anomaly_type = np.random.choice([
            'monetary_spike', 'category_velocity', 
            'temporal_deviation', 'merchant_novelty', 'budget_overage'
        ], p=[0.3, 0.2, 0.2, 0.15, 0.15])
        
        if anomaly_type == 'monetary_spike':
            # Multiply amount by 3-5x
            transaction['amount'] *= np.random.uniform(3, 5)
            transaction['anomaly_type'] = 'monetary_spike'
            
        elif anomaly_type == 'category_velocity':
            # Insert 3-5 extra transactions in same category
            for _ in range(np.random.randint(3, 6)):
                extra = transaction.copy()
                extra['date'] += timedelta(days=np.random.randint(1, 7))
                extra['amount'] *= np.random.uniform(0.5, 1.5)
                extra['anomaly_type'] = 'category_velocity'
                anomalies.append(extra)
            continue
            
        elif anomaly_type == 'temporal_deviation':
            # Shift to unusual hour (2-5 AM)
            transaction['hour'] = np.random.randint(2, 6)
            transaction['anomaly_type'] = 'temporal_deviation'
            
        elif anomaly_type == 'merchant_novelty':
            # Recategorize to unusual category
            unusual_categories = ['recreation', 'clothing', 'miscellaneous']
            transaction['category'] = np.random.choice(unusual_categories)
            transaction['anomaly_type'] = 'merchant_novelty'
            
        elif anomaly_type == 'budget_overage':
            # Compute budget allocation and exceed it
            budget = compute_category_budget(transaction['category'])
            transaction['amount'] = budget * np.random.uniform(1.5, 2.5)
            transaction['anomaly_type'] = 'budget_overage'
        
        anomalies.append(transaction)
    
    # Add anomalies to transaction list
    all_transactions = transactions + anomalies
    all_transactions.sort(key=lambda x: x['date'])
    
    return all_transactions
```

---

## 6. Validation Rules

### 6.1 Persona Validation Checklist

Before synthetic data is finalized, each archetype must be validated against:

| Check | Criterion | Pass/Fail |
|-------|-----------|-----------|
| **Income Range** | Monthly income within FIES NCR decile range | |
| **Expense Ratio** | Category ratios match FIES NCR averages | |
| **Obligation Ratio** | Within PFP-specified range | |
| **CV** | Within PFP-specified range | |
| **Savings Rate** | Realistic for income level (0-30%) | |
| **Engel Coefficient** | Within Philippine range (0.3-0.6) | |
| **Temporal Patterns** | Payday spikes visible in transaction history | |
| **Anomaly Rate** | 1-5% of total transactions | |

### 6.2 SME Validation Form

```
Persona Archetype: _______________
FIES NCR Decile: _______________
Employment Type: _______________

Please validate the following:

1. Income Range: ₱_________ to ₱_________ per month
   [ ] Realistic  [ ] Too high  [ ] Too low

2. Expense Allocation:
   - Food: _______%
   - Housing: _______%
   - Transport: _______%
   - Health: _______%
   - Education: _______%
   - Other: _______%
   [ ] Realistic  [ ] Needs adjustment

3. Obligation Ratio: _______%
   [ ] Realistic  [ ] Too high  [ ] Too low

4. Savings Rate: _______%
   [ ] Realistic  [ ] Too high  [ ] Too low

5. Behavioral Patterns:
   - Payday spending spike: [ ] Realistic  [ ] Exaggerated  [ ] Understated
   - Petsa de peligro: [ ] Realistic  [ ] Exaggerated  [ ] Understated
   - Christmas escalation: [ ] Realistic  [ ] Exaggerated  [ ] Understated

6. Overall Assessment:
   [ ] Approved  [ ] Needs revision (specify: _______________)

Expert Signature: _______________
Date: _______________
```

---

## 7. Implementation Notes

### 7.1 Pipeline Architecture

```
FIES NCR CSV
    │
    ▼
┌─────────────────────────┐
│  1. Data Loader          │  Load FIES NCR subset
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  2. Archetype Mapper     │  Map FIES records to 12 archetypes (A–L)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  3. Income Generator     │  Generate monthly income streams
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  4. Expense Generator    │  Generate daily expenses by category
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  5. Temporal Injector    │  Apply payday, petsa, Christmas patterns
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  6. Behavioral Injector  │  Inject timing, diversity, volatility features
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  7. Anomaly Injector     │  Inject 1-5% anomalous transactions
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  8. Label Generator      │  Compute PFP labels from CV + obligation ratio
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  9. Validator            │  Run validation checks
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  10. Export              │  Save to CSV/Parquet
└─────────────────────────┘
```

### 7.2 Output Schema

```python
# Transaction-level output:
{
    'user_id': str,           # UUID
    'date': date,             # Transaction date
    'category': str,          # Expense category
    'amount': float,          # Transaction amount (PHP)
    'type': str,              # 'expense' or 'income'
    'pfp_label': str,         # PFP classification
    'income_stability_cv': float,  # CV for income stability
    'obligation_ratio': float,     # Obligation weight score
    'anomaly_type': str|None,      # 'monetary_spike', etc. or None
    'features': dict          # Injected behavioral features
}

# Persona-level output:
{
    'user_id': str,           # UUID
    'archetype': str,         # Archetype name
    'pfp_label': str,         # PFP classification
    'income_stability_cv': float,
    'obligation_ratio': float,
    'monthly_income': float,  # Average monthly income
    'family_size': int,
    'employment_type': str,
    'fies_decile': int,       # FIES NCR decile
    'transaction_count': int,
    'anomaly_count': int,
    'anomaly_rate': float
}
```

### 7.3 Storage Requirements

| Component | Format | Size Estimate |
|-----------|--------|---------------|
| 12,000 personas | Parquet | ~4 MB |
| 12,000 × 365 days × 10 categories | Parquet | ~1.3 GB |
| Transaction-level (50M records) | Parquet | ~5 GB |
| Feature matrices | NumPy | ~250 MB |
| **Total** | | **~6.75 GB** |

---

## 8. RRL Justifications

| Rule | RRL Support | Topic |
|------|-------------|-------|
| Log-normal income distribution | FIES distributions confirm right-skewed behavior | 1.B.I |
| Payday spikes (15th/30th) | BSP Payment Systems Report: semi-monthly payroll convention | 2.B.I |
| Petsa de peligro | Cacnio & Lubangco (2026): scarcity and consumption priorities | 2.D.I |
| Christmas escalation | BSP CES Q4 2025: seasonal spending outlook | 2.D.I |
| 13th month pay | Republic Act No. 6686: 13th Month Pay Law | 2.D.I |
| Semana Santa | PhilStar/Inquirer: Holy Week travel volume | 2.D.I |
| Undas | PhilStar/GMA: All Saints Day travel patterns | 2.D.I |
| BPO payday variation | BSP BOP Report Q2 2024: BPO export revenues | 2.D.I |
| Paluwagan/Ambag | Flores (2025): Financial freedom of Filipinos | 2.A.I |
| Isolation Forest for anomalies | D'Souza et al.: effective for unsupervised anomaly detection | 8.B.II |
| 1-5% anomaly rate | George et al.: fraud prevalence typically <1-5% | 8.A.I |
| Semi-supervised approach | D'Souza et al.: unsupervised preferred due to absence of labeled data | 8.B.I |

---

## 9. Open Questions for SME

1. **Obligation ratio thresholds:** Is 0.5 the correct cutoff for Obligated vs Flexible in NCR context?
2. **CV threshold:** Is 0.2 the correct cutoff for Stable vs Variable in NCR context?
3. **Category ratios:** Do FIES NCR averages accurately reflect young professional spending?
4. **Anomaly rate:** Is 3% realistic for personal finance, or should it be lower (1-2%)?
5. **Temporal patterns:** Are the payday/petsa/Christmas multipliers realistic for NCR?

---

*Document created: 2026-07-17*
*Next steps: Review with SME, finalize thresholds, begin pipeline implementation*
