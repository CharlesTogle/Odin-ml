---
document: "persona-validation-list-SME-draft.md"
version: "0.1"
status: Draft (SME Review)
scope: Training — Prerequisites
depends_on:
  - "../1_problem-statement/README.md"
  - "feature-set.md"
---

> **Review Target:** SME (Dr. Go)  
> **Purpose:** Validate archetype roster, parameter ranges, and coverage before synthetic data generation.  
> **Date:** July 2026

---

## 1. Framework Overview

The Financial Profile (PFP) framework identifies **five features** of a user's financial state, categorized by how the system uses them:

| Category | Feature | Role in System | Data Source |
|----------|---------|----------------|-------------|
| **Classifying** | Financial Stability | Determines PFP class | Transaction history (income CV) |
| **Classifying** | Financial Obligation | Determines PFP class | Transaction data (expense ratios) |
| **Classifying** | Financial Tolerance | Determines PFP class | Balance history (emergency runway) |
| **Non-Classifying** | Financial Behavior | Personalizes communication/nudges | Questionnaire |
| **Non-Classifying** | Surplus/Deficit Rate | Internal tracking (forecasting, projections) | Transaction data (inflow/outflow) |

The three classifying dimensions produce a **2×2×2 = 8 cell matrix**. Archetypes are assigned to cells based on their quantitative parameters. Financial Behavior and Surplus/Deficit Rate do **not** determine the PFP class.

### Classifying Dimension Definitions

| Dimension | Label: Low | Label: High | Measurement |
|-----------|-----------|-------------|-------------|
| **Financial Stability** | Variable (CV ≥ 0.5) | Stable (CV < 0.5) | Coefficient of variation of income amounts across observed periods |
| **Financial Obligation** | Flexible (ratio ≤ 0.6) | Obligated (ratio > 0.6) | (Essential + Obligatory expenses) / Total expenses |
| **Financial Tolerance** | At-Risk (runway < 3 months) | Tolerant (runway ≥ 3 months) | Months of current expenses that existing savings/balance can cover if inflows cease |

> **Note:** Thresholds above are working assumptions. Numeric thresholds are intentionally not finalized — they will be validated in Phase 4.5 (Dimension Threshold Discovery).

---

## 2. Persona Matrix (2×2×2)

All 8 cells are represented. Four additional archetypes cover edge cases within high-frequency cells.

| | **Stable** | **Variable** |
|---|---|---|
| **Obligated + Tolerant** | A — Regular employee, moderate obligations, healthy emergency fund | E — Freelancer, high obligations, adequate savings buffer |
| **Obligated + At-Risk** | B — Regular employee, heavy obligations, no savings | F — Contract worker, high obligations, living paycheck-to-paycheck |
| **Flexible + Tolerant** | C — Regular employee, low obligations, strong savings | G — Freelancer, low obligations, healthy emergency fund |
| **Flexible + At-Risk** | D — Regular employee, low obligations, minimal savings | H — Irregular earner, variable income, no emergency fund |

**Additional Edge-Case Archetypes (4):**

| Archetype | Cell | Edge Case |
|-----------|------|-----------|
| I | Variable-Obligated-At-Risk | Recovering from financial shock (recent job loss → re-employed, runway depleted) |
| J | Variable-Flexible-At-Risk | Variable income but deliberately low obligations; borderline financial tolerance |
| K | Stable-Obligated-Tolerant | Stable income, high obligations, healthy savings — but obligations approaching threshold |
| L | Stable-Flexible-At-Risk | Stable income, genuinely low obligations, but no savings habit (spends discretionary freely) |

---

## 3. Archetype Roster

### Archetype A — Regular Employee, Moderate Obligations, Healthy Emergency Fund

**Dimension Values:**
- Financial Stability: **Stable** (CV ≈ 0.1)
- Financial Obligation: **Obligated** (ratio ≈ 0.70)
- Financial Tolerance: **Tolerant** (runway ≈ 5 months)

**Employment Context:** Regular employee at a BPO company, 2 years tenure. Consistent monthly salary with minor overtime variation. Single, no dependents.

**Obligation Breakdown:**
- Rent (studio apartment, Makati): ~25% of income
- Groceries and utilities: ~15%
- Transportation (commute): ~8%
- Consumer debt repayment (credit card): ~12%
- Insurance (HMO + life): ~5%
- Discretionary (dining out, entertainment): ~15%
- Savings/investment: ~10%

**Emergency Runway:** Savings balance covers ~5 months of current monthly expenses. No dependents to support in emergency.

**Expected PFP Label:** Stable · Obligated · Tolerant

---

### Archetype B — Regular Employee, Heavy Obligations, No Savings

**Dimension Values:**
- Financial Stability: **Stable** (CV ≈ 0.15)
- Financial Obligation: **Obligated** (ratio ≈ 0.85)
- Financial Tolerance: **At-Risk** (runway ≈ 0.5 months)

**Employment Context:** Regular employee at a manufacturing firm, 5 years tenure. Consistent salary. Married, 1 child (toddler).

**Obligation Breakdown:**
- Rent (2-bedroom apartment, Pasig): ~20%
- Groceries and utilities: ~20%
- Transportation (commute + child transport): ~10%
- Childcare: ~8%
- Consumer debt repayment (2 credit cards + personal loan): ~15%
- Insurance (HMO): ~5%
- Remittance to provincial family: ~7%
- Discretionary: ~5%
- Savings: ~0%

**Emergency Runway:** Savings balance covers ~0.5 months of expenses. Entire paycheck goes to obligations.

**Expected PFP Label:** Stable · Obligated · At-Risk

---

### Archetype C — Regular Employee, Low Obligations, Strong Savings

**Dimension Values:**
- Financial Stability: **Stable** (CV ≈ 0.08)
- Financial Obligation: **Flexible** (ratio ≈ 0.45)
- Financial Tolerance: **Tolerant** (runway ≈ 9 months)

**Employment Context:** Regular employee at a tech company, 3 years tenure. Consistent salary with quarterly bonus. Single, lives with parents (no rent).

**Obligation Breakdown:**
- Groceries and utilities (contributes to household): ~15%
- Transportation (motorcycle fuel): ~5%
- Insurance (HMO + life + personal accident): ~8%
- Consumer debt: ~0%
- Discretionary (dining, hobbies, travel fund): ~17%
- Savings/investment: ~30%
- Personal spending: ~25%

**Emergency Runway:** Savings + investment balance covers ~9 months of current expenses. No dependents.

**Expected PFP Label:** Stable · Flexible · Tolerant

---

### Archetype D — Regular Employee, Low Obligations, Minimal Savings

**Dimension Values:**
- Financial Stability: **Stable** (CV ≈ 0.12)
- Financial Obligation: **Flexible** (ratio ≈ 0.50)
- Financial Tolerance: **At-Risk** (runway ≈ 1.5 months)

**Employment Context:** Regular employee at a government agency, 4 years tenure. Very consistent salary. Single, no dependents. Lives with parents (no rent).

**Obligation Breakdown:**
- Groceries and utilities (contributes to household): ~12%
- Transportation: ~5%
- Consumer debt (gadget installment plans): ~8%
- Insurance (GSIS): ~5%
- Discretionary (online shopping, dining, entertainment): ~30%
- Savings: ~5% (inconsistent — some months zero)
- Personal spending: ~35%

**Emergency Runway:** Savings balance covers ~1.5 months. Large discretionary allocation could theoretically be redirected.

**Expected PFP Label:** Stable · Flexible · At-Risk

---

### Archetype E — Freelancer, High Obligations, Adequate Savings Buffer

**Dimension Values:**
- Financial Stability: **Variable** (CV ≈ 0.7)
- Financial Obligation: **Obligated** (ratio ≈ 0.75)
- Financial Tolerance: **Tolerant** (runway ≈ 4 months)

**Employment Context:** Freelance graphic designer, 3 years self-employed. Income varies monthly (some months 2× others). Single, no dependents. Has 3 active retainer clients.

**Obligation Breakdown:**
- Rent (1-bedroom, Quezon City): ~22%
- Groceries and utilities: ~15%
- Transportation: ~7%
- Health insurance (PhilHealth + private): ~5%
- Debt repayment (previous business loan): ~12%
- Discretionary: ~10%
- Savings/investment: ~14% (varies — more in good months)
- Business expenses (software, equipment): ~15%

**Emergency Runway:** Savings balance covers ~4 months of current expenses. Retainer income provides partial floor in low months.

**Expected PFP Label:** Variable · Obligated · Tolerant

---

### Archetype F — Contract Worker, High Obligations, Paycheck-to-Paycheck

**Dimension Values:**
- Financial Stability: **Variable** (CV ≈ 0.65)
- Financial Obligation: **Obligated** (ratio ≈ 0.80)
- Financial Tolerance: **At-Risk** (runway ≈ 0.3 months)

**Employment Context:** Fixed-term project-based employee (6-month contracts) at a construction firm. Income gaps between contracts (1–3 weeks). Married, spouse is a part-time vendor.

**Obligation Breakdown:**
- Rent (room rental, Malabon): ~18%
- Groceries and utilities: ~20%
- Transportation: ~8%
- Debt repayment (2MP loan + credit card): ~15%
- Child's school expenses: ~10%
- Remittance to in-laws: ~5%
- Discretionary: ~4%
- Savings: ~0%

**Emergency Runway:** Savings balance covers ~0.3 months. Spouse's vendor income is inconsistent and not captured in system.

**Expected PFP Label:** Variable · Obligated · At-Risk

---

### Archetype G — Freelancer, Low Obligations, Healthy Emergency Fund

**Dimension Values:**
- Financial Stability: **Variable** (CV ≈ 0.6)
- Financial Obligation: **Flexible** (ratio ≈ 0.40)
- Financial Tolerance: **Tolerant** (runway ≈ 7 months)

**Employment Context:** Freelance content writer and part-time virtual assistant, 2 years self-employed. Income variable but has built reserves during high-earning months. Single, no dependents. Lives with sibling (contributes to utilities only).

**Obligation Breakdown:**
- Utilities contribution: ~5%
- Transportation: ~5%
- Health insurance (private): ~4%
- Discretionary (dining, subscriptions, hobbies): ~26%
- Savings/investment: ~25% (variable — some months higher)
- Business expenses: ~5%
- Personal spending: ~30%

**Emergency Runway:** Savings balance covers ~7 months of current expenses. Low obligations mean even low-income months are manageable.

**Expected PFP Label:** Variable · Flexible · Tolerant

---

### Archetype H — Irregular Earner, Variable Income, No Emergency Fund

**Dimension Values:**
- Financial Stability: **Variable** (CV ≈ 0.8)
- Financial Obligation: **Flexible** (ratio ≈ 0.45)
- Financial Tolerance: **At-Risk** (runway ≈ 1 month)

**Employment Context:** Part-time tricycle driver and occasional market vendor. No formal employment contract. Income is daily and highly variable. Single, no dependents. Lives with elderly parent (contributes to household expenses).

**Obligation Breakdown:**
- Household contribution: ~15%
- Transportation/fuel: ~10%
- Food (personal + shared household): ~15%
- Health (out-of-pocket, no insurance): ~5%
- Discretionary: ~10%
- Savings: ~0% (inconsistent)
- Remaining income: spent on miscellaneous/untracked

**Emergency Runway:** Cash-on-hand + minimal savings covers ~1 month of expenses. No formal savings mechanism.

**Expected PFP Label:** Variable · Flexible · At-Risk

---

### Archetype I — Variable-Obligated-Low, Recovering from Financial Shock

**Dimension Values:**
- Financial Stability: **Variable** (CV ≈ 0.7)
- Financial Obligation: **Obligated** (ratio ≈ 0.78)
- Financial Tolerance: **At-Risk** (runway ≈ 0.8 months)

**Employment Context:** Recently re-employed regular employee at a retail company (3 months tenure). Was laid off from previous employer 4 months ago — depleted savings during unemployment. Married, spouse also recently re-employed.

**Obligation Breakdown:**
- Rent (1-bedroom, Mandaluyong): ~20%
- Groceries and utilities: ~18%
- Transportation: ~8%
- Debt repayment (emergency credit card debt from unemployment): ~15%
- Spouse's transportation: ~5%
- Insurance (HMO, newly enrolled): ~4%
- Discretionary: ~8%
- Savings: ~2% (just restarted)

**Emergency Runway:** Savings recovering — covers ~0.8 months. Debt from unemployment period still outstanding.

**Expected PFP Label:** Variable · Obligated · At-Risk  
**Note:** Archetype is transitional. Financial Tolerance may shift to Tolerant as savings rebuild.

---

### Archetype J — Variable-Flexible-At-Risk, Borderline Financial Tolerance

**Dimension Values:**
- Financial Stability: **Variable** (CV ≈ 0.55)
- Financial Obligation: **Flexible** (ratio ≈ 0.55)
- Financial Tolerance: **At-Risk** (runway ≈ 2.5 months)

**Employment Context:** Part-time sales associate at a mall + occasional online selling. Income variable but has established a base from the retail job. Single, no dependents. Lives with parents (no rent).

**Obligation Breakdown:**
- Groceries and utilities (contributes to household): ~12%
- Transportation: ~7%
- Health expenses (PhilHealth + occasional checkups): ~4%
- Debt repayment (small gadget loan): ~5%
- Discretionary (dining, social, shopping): ~27%
- Savings: ~5% (inconsistent)
- Personal spending: ~40%

**Emergency Runway:** Savings balance covers ~2.5 months — just below the Tolerant threshold. Low obligations provide flexibility, but savings habit is inconsistent.

**Expected PFP Label:** Variable · Flexible · At-Risk  
**Note:** Borderline Financial Tolerance. Small behavioral change (consistent savings) could shift this archetype to Tolerant. Obligation ratio also near the Obligated threshold — a rent obligation would push it over.

---

### Archetype K — Stable-Obligated-Tolerant, Obligations Approaching Threshold

**Dimension Values:**
- Financial Stability: **Stable** (CV ≈ 0.1)
- Financial Obligation: **Obligated** (ratio ≈ 0.65 — near threshold)
- Financial Tolerance: **Tolerant** (runway ≈ 4 months)

**Employment Context:** Regular employee at a telecom company, 4 years tenure. Consistent salary. Married, spouse is a freelance tutor (variable income, not in system). 1 child (school-age).

**Obligation Breakdown:**
- Mortgage (condo unit, Taguig): ~25%
- Groceries and utilities: ~18%
- Transportation (family car amortization + fuel): ~12%
- Child's school tuition: ~8%
- Insurance (HMO + life): ~6%
- Debt repayment (car loan): (included in transportation)
- Discretionary: ~12%
- Savings: ~5%

**Emergency Runway:** Savings + time deposit covers ~4 months of family expenses. Spouse's income is variable and unreliable.

**Expected PFP Label:** Stable · Obligated · Tolerant
**Note:** Obligation ratio is near the 0.6 threshold. A new obligation (e.g., appliance installment, increased school fees) could push this archetype into a higher obligation state, reducing discretionary space. System should monitor obligation drift.

---

### Archetype L — Stable-Flexible-At-Risk, No Savings Habit Despite Stable Income

**Dimension Values:**
- Financial Stability: **Stable** (CV ≈ 0.1)
- Financial Obligation: **Flexible** (ratio ≈ 0.50)
- Financial Tolerance: **At-Risk** (runway ≈ 1 month)

**Employment Context:** Regular employee at a marketing agency, 2 years tenure. Consistent salary with minor variation. Single, no dependents. Lives alone (rents a studio).

**Obligation Breakdown:**
- Rent (studio, BGC): ~20%
- Groceries and utilities: ~12%
- Transportation: ~6%
- Insurance (HMO only): ~3%
- Discretionary (dining out, bar, online shopping, gadgets): ~35%
- Savings: ~2% (inconsistent — some months zero)
- Subscriptions (streaming, gym, apps): ~12%
- Personal spending: ~10%

**Emergency Runway:** Savings balance covers ~1 month. Income is stable, so cash flow is never negative — but no meaningful buffer exists.

**Expected PFP Label:** Stable · Flexible · At-Risk  
**Note:** This archetype challenges the assumption that stable income implies financial health. System should prioritize building savings habit through automated micro-transfers.

---

## 4. Non-Classifying Features (Informational)

Financial Behavior and Surplus/Deficit Rate do **not** determine the PFP class. They are tracked by the system for personalization and internal calculations. The tables below show the possible values and how they influence system behavior for each classifying dimension combination.

### 4.1 Financial Behavior (User Preference — Captured via Questionnaire)

| Trait | Values | Captured Via | System Impact |
|-------|--------|-------------|---------------|
| **Earning Beta** | Procyclical / Acyclical / Countercyclical | Questionnaire: "How does your income typically change during economic downturns?" | Countyclical earners receive more aggressive savings prompts during stable periods; procyclical earners receive liquidity warnings during downturns |
| **Financial Literacy Gradient** | Low / Moderate / High | Questionnaire: 3–5 short financial math questions | Low literacy → simpler language, visual explanations, no jargon; High literacy → detailed breakdowns, amortization tables, rate comparisons |
| **Behavioral Signature** | Present-biased / Loss-averse / Overconfident / Balanced | Questionnaire: 5–7 behavioral finance scenario questions | Present-biased → commitment devices (auto-transfers, locked goals); Loss-averse → loss-prevention framing; Overconfident → risk-awareness nudges |

### 4.2 Surplus/Deficit Rate (Internal Tracking — Derived from Transaction Data)

Surplus/Deficit Rate = (Inflows − Outflows) / Inflows

| Rate Status | Definition | Internal System Use |
|-------------|------------|---------------------|
| **Healthy Surplus** | Rate > 15% | Savings projection confidence is high; system may suggest goal acceleration |
| **Thin Surplus** | Rate 0–15% | System monitors for drift; may suggest micro-optimizations |
| **Near-Zero** | Rate ≈ 0% | System flags as fragile; recommends smallest possible buffer build |
| **Deficit** | Rate < 0% | System triggers debt-hardship pre-check; prioritizes income-expense rebalancing |

### 4.3 System Impact by PFP Class

The tables below show how Financial Behavior and Surplus/Deficit Rate modify system behavior **within** each classifying dimension combination. The PFP class itself is determined solely by Financial Stability, Financial Obligation, and Financial Tolerance.

#### Stable · Obligated

| Financial Tolerance | Surplus/Deficit Rate | System Response |
|--------------------|----------------------|-----------------|
| Tolerant | Healthy Surplus | Standard obligation tracking; suggest surplus allocation to savings goals |
| Tolerant | Thin Surplus | Monitor obligation drift; flag if any new obligation approaches |
| At-Risk | Deficit | Priority: debt-hardship pre-check → hardship threshold assessment → hardship relief options |
| At-Risk | Near-Zero | Priority: emergency fund micro-transfer setup → obligation-priority scheduling |

#### Stable · Flexible

| Financial Tolerance | Surplus/Deficit Rate | System Response |
|--------------------|----------------------|-----------------|
| Tolerant | Healthy Surplus | Standard tracking; suggest goal acceleration or investment exploration |
| Tolerant | Thin Surplus | Flag inconsistency — stable income + low obligations should yield surplus; investigate discretionary |
| At-Risk | Near-Zero | Priority: automated micro-transfer to build buffer → spending category alerts |
| At-Risk | Deficit | Priority: expense categorization audit → identify reducible discretionary |

#### Variable · Obligated

| Financial Tolerance | Surplus/Deficit Rate | System Response |
|--------------------|----------------------|-----------------|
| Tolerant | Healthy Surplus | High months: suggest top-up to emergency fund; low months: draw from buffer |
| Tolerant | Thin Surplus | Monitor income volatility; suggest income-stabilization strategies (retainer pursuit, diversification) |
| At-Risk | Deficit | Priority: emergency fund rebuild → income smoothing education → debt-hardship pre-check |
| At-Risk | Near-Zero | Priority: runway monitoring with alerts → obligation-priority scheduling → micro-transfer in high months |

#### Variable · Flexible

| Financial Tolerance | Surplus/Deficit Rate | System Response |
|--------------------|----------------------|-----------------|
| Tolerant | Healthy Surplus | High months: top-up emergency fund; low months: flexible obligations allow natural drawdown |
| Tolerant | Thin Surplus | Flag volatility impact; suggest building buffer during high-income periods |
| At-Risk | Near-Zero | Priority: income volatility education → automated micro-transfer on high-income detection |
| At-Risk | Deficit | Priority: expense categorization audit → identify reducible non-essential → emergency fund first |

### 4.4 Financial Behavior Modifier (Applied on Top of PFP Class Response)

| Behavioral Trait | Modifier to System Response |
|------------------|----------------------------|
| **Present-biased** | Add commitment devices (auto-transfer locks, savings goal progression bars, delayed discretionary approval) |
| **Loss-averse** | Frame all recommendations as loss-prevention ("Without a buffer, you risk…") rather than gain-promotion ("You could save…") |
| **Overconfident** | Add risk-awareness nudges ("Your runway is X months — here's what happens if…") |
| **Low Financial Literacy** | Simplify language, use visual charts instead of tables, avoid jargon, add explanatory tooltips |
| **High Financial Literacy** | Show detailed breakdowns, amortization schedules, rate comparisons, projections |
| **Countercyclical earner** | During economic downturns: increase savings prompts, reduce investment suggestions; during stability: aggressive buffer building |
| **Procyclical earner** | During economic stability: standard guidance; during downturns: liquidity-first messaging, expense reduction priority |

---

## 6. Validation Form

Please review the following for each archetype:

| Criterion | What to Check |
|-----------|--------------|
| **Parameter realism** | Are the obligation ratios, CV values, and runway months realistic for the described context? |
| **Internal consistency** | Do the dimension values (Stable/Variable, Obligated/Flexible, Tolerant/At-Risk) match the persona description? |
| **Stereotyping** | Does any archetype rely on demographic assumptions (age, gender, location) rather than behavioral parameters? |
| **Missing archetypes** | Are there common Filipino financial profiles not represented in the 12 archetypes? |
| **Threshold sensitivity** | For borderline archetypes (J, K): are the parameter values realistic near the threshold boundaries? |
| **Recovery trajectory** | For Archetype I: is the "recovering from shock" scenario realistic given the parameter values? |

---

## 7. Missing Archetypes

Are there common Filipino financial profiles not represented above?

| Archetype | Stability | Obligation | Financial Tolerance | Description | Why Needed |
|-----------|-----------|------------|-----------------|-------------|------------|
| *(TBD)* | | | | | |
| *(TBD)* | | | | | |
| *(TBD)* | | | | | |

---

## 8. Open Questions for SME

1. **Obligation threshold (0.6):** Is this boundary between Obligated and Flexible realistic for Filipino young adults? Too high? Too low?
2. **Financial Tolerance threshold (3 months):** Is 3 months of emergency runway a meaningful dividing line between Tolerant and At-Risk in the Philippine context?
3. **Income CV threshold (0.5):** Is a coefficient of variation of 0.5 a reasonable boundary between Stable and Variable income for someone transitioning between regular employment and freelancing?
4. **Archetype count:** Are 12 archetypes sufficient, or should additional edge cases be added for the thesis scope?
5. **Archetype I (recovery):** Is the "recovering from financial shock" scenario realistic? Should recovery trajectory be a tracked state?
6. **Archetype L (no savings habit):** Does this archetype adequately represent the "stable income but poor savings behavior" segment, or is it too narrow?

---

*Awaiting SME review. All dimension labels and thresholds are working assumptions subject to validation.*
