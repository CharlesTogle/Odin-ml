# BSP–FIES Crosswalk

**Document Version:** v1.0
**Author(s):** Guevarra
**Date:** 2026-07-26
**Status:** Draft
**Purpose:** Map BSP Consumer Finance Survey (CFS) fields to PSA 2023 FIES NCR microdata, document the unit-of-analysis equivalence assumptions, and formally justify the 12 archetype roster against BSP segmentation.

---

## 1. Unit-of-Analysis Mapping

### 1.1 Survey Units

| Source | Survey Unit | Unit Definition | Implication |
|--------|-------------|-----------------|-------------|
| **BSP CFS 2021** | Individual respondent (18+) | One respondent per household; questions about personal financial behavior, attitudes, and inclusion | Behavioral/attitudinal data is individual-level but reported per household |
| **PSA FIES 2023** | Household aggregate | One row per household; income and expenditure totals summed across all household members | Financial numerical data is household-level, not individual |
| **Odin Target** | Individual user (18+) | One PFP classification per user; transaction history is individual-level | Target user is a single individual within a household |

### 1.2 Disaggregation Assumption

FIES data is household-aggregate, but Odin targets individual users. The persona-generation pipeline applies the following disaggregation:

1. **Income disaggregation:** Household income (`TOINC`) is divided by household size (`FSIZE`) to produce a per-capita income proxy. For archetypes with known employment status, income is further adjusted by employment-type multipliers (e.g., regular salaried = 1.0× per-capita, freelance = 0.7× per-capita reflecting irregular receipts).

2. **Expense disaggregation:** Household expense categories (`FOOD`, `HOUSING_WATER`, `TRANSPORT`, etc.) are divided by household size to produce per-capita expense proxies. Archetype-specific adjustments are applied (e.g., single-person households absorb 100% of food costs; multi-person households split costs).

3. **Savings rate:** Computed as `(TOINC - TOTEX) / TOINC` at the household level, then applied directly to the individual persona as the savings rate parameter.

4. **Emergency runway:** Computed as `savings_balance / monthly_expenses` using the archetype's savings rate and expense level. Not directly available in FIES; derived from savings rate × months of accumulation.

### 1.3 Threat to Validity

This disaggregation assumes that per-capita income and expense shares are reasonable proxies for individual financial profiles. In reality, household spending is not evenly distributed (e.g., primary earners bear disproportionate obligation weight). This mismatch is a known limitation, documented in each MDD's Section 13 under "Key Assumption (BSP-FIES crosswalk)."

---

## 2. Field-Level Mapping Table

### 2.1 Income Source Fields

| BSP CFS Field | Description | FIES Equivalent | FIES Variable ID | Mapping Quality | Notes |
|---------------|-------------|-----------------|-------------------|-----------------|-------|
| Primary income source | Main source of household income | Salaries/wages from regular employment | `REG_SAL` | Direct | 88.2% of FIES NCR households report regular salary |
| Secondary income source | Additional income sources | Entrepreneurial activities | `EAINC` | Direct | 32.7% of FIES NCR households have entrepreneurial income |
| Remittance receipt | Receives remittances from abroad/domestic | Cash receipts from abroad + domestic | `CASH_ABROAD` + `CASH_DOMESTIC` | Direct | 22.1% of FIES NCR households receive remittances |
| Pension income | Receives pension | Pension | `PENSION` | Direct | 17.4% of FIES NCR households receive pension |
| Seasonal income | Income from seasonal employment | Seasonal salaries/wages | `SEASON_SAL` | Direct | 19.6% of FIES NCR households have seasonal income |
| Investment income | Interest, dividends, rentals | Interest + dividends + rentals | `INTEREST` + `DIVIDENDS` + `RENTALS_REC` | Direct | Passive income components |
| Informal income | Family sustenance activities | Net receipt from family sustenance | `NET_RECEIPT` | Direct | Informal sector income |

### 2.2 Financial Behavior Fields

| BSP CFS Field | Description | FIES Equivalent | FIES Variable ID | Mapping Quality | Notes |
|---------------|-------------|-----------------|-------------------|-----------------|-------|
| Deposit account ownership | Has savings/checking account | No direct equivalent | — | **Injected** | BSP: 35.3% of households; assumed based on archetype |
| Emergency fund adequacy | Months of expenses covered by savings | No direct equivalent | — | **Injected** | BSP: only 2.0% have 3+ months; computed from savings rate |
| Debt/loan ownership | Has outstanding debt | No direct equivalent | — | **Injected** | BSP: 29.3% of households; assumed based on archetype |
| Credit card ownership | Has credit card | No direct equivalent | — | **Injected** | BSP: 1.1% of households; assumed based on archetype |
| Digital financial services | Uses e-money or mobile banking | No direct equivalent | — | **Injected** | BSP: 24.3% e-money, 17.7% mobile banking; assumed based on archetype |
| Financial risk tolerance | Self-reported risk tolerance (1-10) | No direct equivalent | — | **Injected** | BSP: average 6.3/10; assigned based on archetype |

### 2.3 Financial Numerical Fields

| BSP CFS Field | Description | FIES Equivalent | FIES Variable ID | Mapping Quality | Notes |
|---------------|-------------|-----------------|-------------------|-----------------|-------|
| Household income | Total annual household income | Total household income | `TOINC` | Direct | Primary income variable |
| Food expenditure | Annual food spending | Total food expenditure | `FOOD` | Direct | 41.1% of FIES NCR total expenditure |
| Housing expenditure | Annual housing/water spending | Housing and water expenditure | `HOUSING_WATER` | Direct | 28.9% of FIES NCR total expenditure |
| Transportation expenditure | Annual transport spending | Transportation expenditure | `TRANSPORT` | Direct | 5.9% of FIES NCR total expenditure |
| Health expenditure | Annual health spending | Medical care expenditure | `HEALTH` | Direct | 2.4% of FIES NCR total expenditure |
| Education expenditure | Annual education spending | Education expenditure | `EDUCATION` | Direct | 1.7% of FIES NCR total expenditure |
| Communication expenditure | Annual communication spending | Communication expenditure | `COMMUNICATION` | Direct | 3.9% of FIES NCR total expenditure |
| Insurance expenditure | Annual insurance spending | Insurance | `INSURANCE` | Direct | 2.6% of FIES NCR total expenditure |
| Net worth | Total household assets minus liabilities | No direct equivalent | — | **Injected** | BSP: average ₱435,346; estimated from savings rate × accumulation |
| Past-due bills | Has past-due bills | No direct equivalent | — | **Injected** | BSP: 16.4% of households; assumed based on archetype |

### 2.4 Demographic Fields

| BSP CFS Field | Description | FIES Equivalent | FIES Variable ID | Mapping Quality | Notes |
|---------------|-------------|-----------------|-------------------|-----------------|-------|
| Household size | Number of household members | Total number of family members | `HH_SIZE` / `FSIZE` | Direct | BSP: 4.1 avg; FIES: 4.03 avg (NCR) |
| Urban/rural | Area classification | Urban/rural indicator | `URB` | Direct | All FIES NCR = urban |
| Region | Geographic region | Region | `W_REGN` | Direct | Filtered to NCR (Region NCR) |
| Province | Province within region | Province | `W_PROV` | Direct | NCR provinces (Manila, Quezon, etc.) |
| Household head age | Age of household head | Household head age | `HEAD_AGE` | Direct | Available in FIES |
| Household head sex | Sex of household head | Household head sex | `HEAD_SEX` | Direct | Available in FIES |
| Household head education | Education level | Household head highest grade completed | `HEAD_EDUC` | Direct | Available in FIES |
| Household head occupation | Occupation type | Household head occupation | `HEAD_OCCUP` | Direct | Available in FIES |
| Household head class of worker | Employment class | Household head class of worker | `HEAD_CLASS` | Direct | Available in FIES (regular, casual, self-employed, etc.) |

### 2.5 Housing & Asset Fields

| BSP CFS Field | Description | FIES Equivalent | FIES Variable ID | Mapping Quality | Notes |
|---------------|-------------|-----------------|-------------------|-----------------|-------|
| Residential property ownership | Owns residential property | Tenure status | `TENURE` | Partial | FIES: owner, renter, rent-free; BSP: 69.9% own |
| Dwelling type | Type of building/house | Type of building/house | `BUILDING` | Direct | Available in FIES |
| Floor area | House floor area | House floor area | `FLOOR_AREA` | Direct | Available in FIES |
| Durable goods ownership | Number of appliances/vehicles | Multiple columns | `FRIDGE`, `WASHER`, `AC`, `CAR`, `MOTORCYCLE`, etc. | Direct | Proxy for asset accumulation |

---

## 3. Archetype-to-Octant Mapping

### 3.1 PFP Dimension Definitions

| Dimension | Label: Low | Label: High | Threshold |
|-----------|-----------|-------------|-----------|
| **Financial Stability** | Variable | Stable | CV < 0.5 = Stable; CV ≥ 0.5 = Variable |
| **Financial Obligation** | Flexible | Obligated | Ratio ≤ 0.6 = Flexible; Ratio > 0.6 = Obligated |
| **Financial Tolerance** | At-Risk | Tolerant | Runway ≥ 3 months = Tolerant; Runway < 3 months = At-Risk |

### 3.2 Full Archetype Roster

| Archetype | Name | PFP Octant | BSP Segment(s) | BSP Justification | Classifier Inclusion |
|-----------|------|------------|-----------------|-------------------|---------------------|
| **A** | Regular employee, moderate obligations, healthy fund | Stable·Obligated·Tolerant | Regular salaried + moderate debt + adequate savings | BSP: 88.2% have regular salary; 29.3% have debt; 2.0% have 3+ months emergency fund (top tier) | Yes — canonical octant |
| **B** | Regular employee, heavy obligations, no savings | Stable·Obligated·At-Risk | Regular salaried + high debt + no emergency fund | BSP: 88.2% regular salary; 29.3% have debt; 16.4% past-due bills; 2.0% have emergency fund | Yes — canonical octant |
| **C** | Regular employee, low obligations, strong savings | Stable·Flexible·Tolerant | Regular salaried + no debt + strong savings | BSP: 35.3% deposit account owners; 2.0% have 3+ months emergency fund (top tier); low debt | Yes — canonical octant |
| **D** | Regular employee, low obligations, minimal savings | Stable·Flexible·At-Risk | Regular salaried + low debt + minimal savings | BSP: 35.3% deposit account; low debt; 2.0% emergency fund (below threshold) | Yes — canonical octant |
| **E** | Freelancer, high obligations, adequate savings buffer | Variable·Obligated·Tolerant | Irregular income + debt + adequate savings | BSP: 32.7% entrepreneurial; 22.1% remittances; 29.3% debt; variable income | Yes — canonical octant |
| **F** | Contract worker, high obligations, paycheck-to-paycheck | Variable·Obligated·At-Risk | Contractual/casual + high debt + no savings | BSP: 19.6% seasonal income; 29.3% debt; 16.4% past-due bills; no emergency fund | Yes — canonical octant |
| **G** | Freelancer, low obligations, healthy emergency fund | Variable·Flexible·Tolerant | Irregular income + no debt + strong savings | BSP: 32.7% entrepreneurial; low debt; 2.0% have 3+ months (top tier); high risk tolerance | Yes — canonical octant |
| **H** | Irregular earner, variable income, no emergency fund | Variable·Flexible·At-Risk | Informal income + no debt + no savings | BSP: 22.1% remittances; informal sector; no deposit account; minimal financial services | Yes — canonical octant |
| **I** | Recovering from financial shock | Variable·Obligated·At-Risk | Recently re-employed + depleted savings + debt from unemployment | BSP: 16.4% past-due bills; 29.3% debt; emergency credit during job loss; transitional state | Yes — edge case (shares octant with F; distinguished by temporal trajectory) |
| **J** | Variable income, low obligations, borderline tolerance | Variable·Flexible·At-Risk | Part-time + occasional income + borderline savings | BSP: 19.6% seasonal; borderline emergency fund (2.5 months); low debt | Yes — edge case (borderline Tolerant/At-Risk threshold) |
| **K** | Stable income, high obligations approaching threshold | Stable·Obligated·Tolerant | Regular salaried + mortgage/car loan + moderate savings | BSP: 69.9% residential property ownership; 29.3% debt; obligation ratio near 0.6 threshold | Yes — edge case (obligation ratio near threshold boundary) |
| **L** | Stable income, no savings habit | Stable·Flexible·At-Risk | Regular salaried + low debt + no savings despite stable income | BSP: 35.3% deposit account; discretionary spending > savings; no emergency fund | Yes — edge case (challenges "stable = healthy" assumption) |

### 3.3 Edge-Case vs Canonical Distinction

The 8 canonical archetypes (A–H) map one-to-one to the 8 PFP octants. The 4 edge-case archetypes (I–L) each share an octant with a canonical archetype but are distinguished by:

- **I** (shares Variable·Obligated·At-Risk with F): Temporal trajectory — recovering from financial shock rather than chronically paycheck-to-paycheck
- **J** (shares Variable·Flexible·ight with H): Borderline threshold — financial tolerance near the 3-month boundary, not chronically zero
- **K** (shares Stable·Obligated·Tolerant with A): Threshold proximity — obligation ratio near the 0.6 boundary, not deep into Obligated territory
- **L** (shares Stable·Flexible·At-Risk with D): Behavioral distinction — stable income but deliberately low savings, not low-income with minimal savings

For Classifier training, all 12 archetypes resolve to one of the 8 PFP octants. Edge cases are included in training (not excluded) because they represent realistic within-octant variation that the Classifier must handle. The deterministic ground-truth formula labels them by their octant, regardless of edge-case status.

### 3.4 Out-of-Scope Archetypes (Not Included)

| Archetype | Reason Excluded | BSP Overlap |
|-----------|----------------|-------------|
| **OFW Dependent** | Primarily a household composition type, not a financial behavior archetype; remittance income is captured by archetypes E, F, G, H | BSP: 22.1% receive remittances — captured by existing archetypes |
| **Senior Citizen** | Age-based, not behavior-based; pension income captured by existing archetypes | BSP: 17.4% receive pension — captured by existing archetypes |
| **Retired Couple** | Household composition, not financial behavior; pension + low obligations captured by archetype C or G | BSP: pension income + low obligations — captured by existing archetypes |

---

## 4. Assumptions & Limitations

### 4.1 Explicit Assumptions

1. **BSP≈FIES≈Odin-user equivalence:** Archetype segmentation from BSP CFS (behavioral/attitudinal) is mapped to FIES NCR (financial numerical) as if both describe the same population. In reality, BSP CFS is nationwide (not NCR-specific) and FIES NCR is NCR-specific. Any mismatch between BSP-defined archetypes and FIES-derived income/expense profiles is a threat to validity, not a confirmed equivalence.

2. **Household-to-individual disaggregation:** FIES data is household-aggregate; Odin targets individual users. Per-capita disaggregation assumes roughly even income/expense distribution within households, which is known to be imperfect.

3. **Temporal mismatch:** BSP CFS is 2021; FIES NCR is 2023. Income and expense levels may have shifted due to inflation and economic changes. The 2-year gap is accepted as unavoidable given data availability.

4. **Behavioral features are injected, not observed:** Emergency fund adequacy, deposit account ownership, debt levels, credit card ownership, financial risk tolerance, and digital financial services usage are assigned to archetypes based on BSP aggregate statistics, not observed in FIES microdata. Any performance gap between Tier 1 and learned classifiers is only as trustworthy as these injection rules.

5. **Archetype count is fixed at 12:** The 8 canonical + 4 edge-case roster is a design decision, not an empirical finding. Additional archetypes may be needed if BSP segmentation reveals groups not captured by the current 12.

### 4.2 Limitations

1. **BSP CFS is nationwide; FIES data is NCR-only.** Archetype behavioral patterns are drawn from nationwide data but applied to NCR-specific financial profiles. Urban/rural differences in financial behavior may not transfer cleanly.

2. **FIES does not capture behavioral/attitudinal data.** Fields like "emergency fund adequacy" or "financial risk tolerance" are not available in FIES and must be injected from BSP statistics or expert judgment.

3. **Archetype assignment is deterministic, not probabilistic.** Each synthetic persona is assigned to exactly one archetype. In reality, individuals may exhibit mixed behaviors across archetypes.

4. **Income disaggregation is simplified.** Per-capita income is a crude proxy; household income distribution is typically skewed (primary earners receive more). The pipeline does not model intra-household income distribution.

5. **Savings rate is computed from FIES aggregates.** FIES reports total income and total expenditure; savings is derived as the residual. This does not distinguish between voluntary savings and forced savings (e.g., mandatory contributions).

---

## 5. Data Source Summary

| Source | Year | Scope | Key Statistics Used |
|--------|------|-------|---------------------|
| **PSA 2023 FIES NCR** | 2023 | 41,380 households, 87 columns | Income: avg ₱502,607/yr, median ₱397,866/yr; Expense ratios: Food 41.1%, Housing 28.9%, Transport 5.9%; Savings rate: mean 20.4%; Household size: 4.03 |
| **BSP 2021 Consumer Finance Survey** | 2021 | Nationwide, 267 pages | Deposit accounts: 35.3%; Emergency fund 3+ months: 2.0%; Debt: 29.3%; Credit cards: 1.1%; Residential property: 69.9%; Net worth: ₱435,346; Risk tolerance: 6.3/10; E-money: 24.3%; Mobile banking: 17.7%; Past-due bills: 16.4% |

---

*This document is referenced by all three MDDs (PFP Classifier v1.3, Forecaster v2.3, Anomaly Detector v2.2) in Section 2 (Data Collection Plan) and Section 13 (Concept Drift & Assumptions).*
