# OpenCode Agent Task List — Odin MDD Follow-Ups
**Date:** 2026-07-26 (status refreshed 2026-08-10)
**Relates to:** `module-design-document.md` (PFP Classifier v1.4, Forecaster v2.4, Anomaly Detector v2.3, Budget Optimizer v1.0)
**Purpose:** Tasks flagged in the MDD with `[PROP — pending]` or `[NOTE]` that require external context, source documents, or research the researchers/Claude do not have loaded in-session. Each task lists what's needed, why it matters, and what "done" looks like so the agent can close the loop without a researcher re-explaining context.

## Status Summary (2026-08-10)

| Task | Status |
|------|--------|
| 1 — Build `bsp-fies-crosswalk.md` | **Done** — file exists at `1_problem-statement/bsp-fies-crosswalk.md` (v1.0, Draft) |
| 2 — BSP-informed edge-case archetypes | Pending — needs SME review of `persona-validation-list-SME-draft.md` |
| 3 — Author+year citations | Pending |
| 4 — Ground candidate models in literature + cost tags | Pending |
| 5 — Forecaster target metrics from literature | Pending |
| 6 — Forecaster persona duration consistency | Pending |
| 7 — Unify shared Data phase across timelines | Pending |

---

## Task 1 — Build `bsp-fies-crosswalk.md`
**Priority:** High (blocks Data Collection Plan sign-off for all three modules)

**Context:** Odin's synthetic persona pipeline now draws on two separate PSA/BSP sources: the **BSP Consumer Finance Report** (archetype/persona segmentation) and **PSA 2023 FIES microdata** (granular transaction-level data). The MDD currently references a crosswalk file that doesn't exist yet.

**What to produce:**
1. **Unit-of-analysis mapping** — state explicitly how BSP's survey unit (household vs. individual respondent) and FIES's household-aggregate unit each map onto Odin's individual-user target. Reuse the disaggregation-assumption pattern already established for FIES-only use.
2. **Field-level mapping table** — for every BSP field used to define an archetype (e.g., income source type, savings behavior, debt indicators — pull actual field names from the report), state which FIES field(s), if any, it's cross-referenced against, and flag any BSP field with no FIES counterpart (these become injected/assumed features, same treatment as existing indirect features).
3. **Archetype-to-octant mapping** — formally confirm or correct the working assumption that the 12 archetypes = 8 PFP octants (Stable/Variable × Flexible/Obligated × Tolerant/At-Risk) + up to 4 edge cases. Table format: archetype name → BSP segment(s) it derives from → PFP octant it maps to (or "edge case — no direct octant mapping").
4. **Assumptions & limitations section** — name the BSP≈FIES≈Odin-user equivalence as an explicit threat to validity, matching the tone of the existing "Key Assumption" blocks in each MDD's Section 13.

**Done when:** the file exists, is referenced correctly from all three MDDs' Section 2 and Section 13 (already updated to point to it), and the `[PROP — pending]` tags in `module-design-document.md` can be removed.

---

## Task 2 — Identify and justify BSP-informed edge-case archetypes
**Priority:** High (blocks Task 1's archetype table and the Classifier's ground-truth formula)

**Context:** Guevarra's working hypothesis is that the 12 archetypes are the 8 PFP octant combinations plus up to 4 edge cases the BSP Consumer Finance Report may surface (e.g., informal/irregular income sources, remittance-dependent households, unbanked or heavily cash-based segments — confirm against the actual report rather than assuming these examples).

**What to do:**
1. Review the BSP Consumer Finance Report's segmentation and flag any respondent group that doesn't cleanly resolve to one of the 8 octants (e.g., income too irregular to score as Stable *or* Variable under the existing formula, or obligation structure that isn't captured by the binary Flexible/Obligated split).
2. Cross-check candidates against the three archetypes already ruled **out of scope** in an earlier audit (OFW Dependent, Senior Citizen, Retired Couple) — determine whether any BSP edge case duplicates, subsumes, or is genuinely distinct from these. Out-of-scope archetypes stay out of scope; don't reintroduce them under a new name.
3. For each confirmed edge case, answer the open question flagged in the Classifier's Data Collection Plan: **does this persona type get folded into its nearest octant for Classifier training, or excluded from Classifier training entirely** (while still remaining valid for Forecaster/Anomaly Detector personas, which don't require an octant label)? This can't be left implicit — the Classifier's deterministic ground-truth formula depends on every training persona having a resolvable label.

**Done when:** a final, named list of 0–4 edge-case archetypes exists with BSP justification, out-of-scope cross-check documented, and the Classifier-inclusion question answered per archetype.

---

## Task 3 — Enforce author+year citations; remove anonymous "Paper N" placeholders
**Priority:** Medium (research-integrity risk at defense)

**Context:** The current MDD has **45 citation instances across 18 distinct anonymous placeholders** (e.g., "Paper 97, 2025," "Paper 30, 2025," "Paper 9, 2025" — concentrated in the Anomaly Detector's drift-detection and monitoring sections). These aren't verifiable or defensible against panelist questioning.

**What to do:**
1. For each "Paper N, 2025" placeholder, locate the actual source in the team's literature review and replace it with `[Author(s), Year]` format, consistent with the properly-cited entries already in the document (e.g., `[Shakhovska & Pukach, 2025]`).
2. If a placeholder's underlying source can't be found or re-verified, do not guess an author — flag it explicitly as `[CITATION NOT FOUND — verify or remove claim]` rather than leaving it anonymous or inventing an attribution.
3. Apply the same author+year convention to all *future* literature additions across `Specification.md` and the `Odin-ML` repo, not just this MDD.

**Done when:** zero "Paper N" placeholders remain in `module-design-document.md`, and every remaining citation resolves to a real, checkable source.

---

## Task 4 — Ground candidate models in literature actually used; tag implementation cost
**Priority:** Medium (scope discipline, especially for Anomaly Detector)

**Context:** Candidate models per module should be models actually used in the reviewed literature (not literature-adjacent or aspirational). The Anomaly Detector's Tier 3–4 candidate list in particular (TA-IFDC, CS-DNN, SSL, Mixture-of-Experts, ATAD-Net CNN-LSTM, LSTM+RL) and its cold-start table (federated learning, behavioral biometrics, TEMPO zero-shot transformers) read as more of a literature survey than a scoped shortlist for a single thesis module with an ~55-day budget.

**What to do:**
1. For each candidate model across all three MDDs, confirm it's cited from a paper that actually implemented and evaluated that model (not just mentioned it), per Task 3's citation cleanup.
2. Add a lightweight "implementation cost" tag to each candidate (e.g., Low/Medium/High relative to the module's timeline budget) so pruning decisions are visible in the document rather than made silently later.
3. Flag any candidate whose complexity is clearly disproportionate to the module's timeline for researcher review — don't unilaterally delete, but surface it.

**Done when:** every candidate model in all three MDDs has a real literature source and a cost tag, and researchers have an explicit prune/keep list to review.

---

## Task 5 — Derive Forecaster target metrics from literature; flag domain mismatches
**Priority:** Medium

**Context:** Forecaster metrics should be derived from literature first, and only redefined by researchers/Claude if found inappropriate. Current targets (e.g., MAPE < 5% total-level, citing NNAR at 2.67% and CNN-LSTM at 2.72%) cite benchmarks that are likely from different datasets/domains than Odin's synthetic Filipino persona data.

**What to do:**
1. Re-derive each Forecaster KPI target from literature that used comparable data (household/personal finance forecasting, not generic time-series benchmarks from unrelated domains).
2. Where the closest available literature benchmark is domain-mismatched, flag it explicitly rather than presenting it as a like-for-like target, and propose a researcher-defined fallback target with reasoning.

**Done when:** each Forecaster KPI cites a domain-appropriate benchmark, or is explicitly marked as researcher-defined with justification.

---

## Task 6 — Reconcile Forecaster persona duration inconsistency
**Priority:** Low (internal consistency, quick fix)

**Context:** Flagged in-document (`module-design-document.md`, Forecaster Section 2): the Sourcing Strategy and row-count math describe personas with **12 months** of daily transaction history, while the Minimum Viable Dataset Size bullet describes a **3-month** "mature" history — these can't both be right.

**What to do:** Confirm the intended persona duration with the team and correct whichever figure is wrong (likely the "3-month" language, which may have been copied from the Classifier's MDD without updating for the Forecaster's longer horizon needs).

**Done when:** all persona-duration figures in the Forecaster section agree with each other and with the row-count math.

---

## Task 7 — Unify the shared Data phase across the three timeline tables
**Priority:** Low (planning clarity, not a technical blocker)

**Context:** Each MDD's Timeline table currently budgets its own 14–16 day "Data" phase independently, but the pipeline is shared through preprocessing/EDA and only triplicates at feature engineering. As written, the three timelines could be misread as requiring the data step three separate times.

**What to do:** Restructure the timeline tables (or add a shared cross-module timeline) so the Data phase appears once as a shared effort, with each module's timeline picking up from the point of triplication (feature engineering) onward.

**Done when:** the three MDDs' timelines no longer imply three independent data-collection efforts.

---

## Known Data & Version Gaps (flagged, not fixed)

These were surfaced during the serving-API work (2026-08-10) and left as **flags** per the scoped agreement — none were silently "fixed" in the training data or retrained models.

1. **Persona volume vs. SME-reviewed archetypes.** The persona validation list documents **12 archetypes (A–L)** for SME review, but the pipeline generates **12,000 personas** (`generate_personas.py` default `personas_per_archetype=1000`, verified in `training/synth/personas.parquet`). Confirm 12,000 is intended (per FIES 2023 sample size) or scale down.
2. **PFP per-class support is zero for some labels.** `train_pfp.py` produced per-class metrics of 0% for several `pfp_label` classes (e.g., `Stable/Obligated/At-Risk`). This is a labeling artifact of the synth data, not a serving bug — needs a labeling-quality pass before trusting PFP class confidence.
3. **Label vocabulary mismatch.** `system-spec.md` v0.3.0 uses `At-Risk` while training data and `train_pfp.py` use `Tight` (e.g., `Stable/Obligated/Tight`). Either the spec or the labels must change; the spec is the ground truth. **Status: resolved in code and docs (2026-08-13)** — serving code, training scripts, and training docs now use `At-Risk` per the spec. The only remaining `Tight` labels live in the gitignored trained artifacts (`training/models/`); they will resolve on the next retrain.
4. **Python version drift.** The thesis system spec targets Python 3.14; `odin-ml` runtime is pinned to 3.13.14 (`.python-version`). Revisit before deployment. **Status: resolved (2026-08-13)** — `.python-version` bumped to 3.14.4 and all references updated; install the venv under 3.14.4 on next setup.
5. **scikit-learn version pin.** Training artifacts were produced with scikit-learn 1.9.0, but `requirements.txt` pins 1.8.0 — produces `InconsistentVersionWarning` app-wide (filtered, non-fatal). Pin to 1.9.0 or retrain. **Status: resolved (2026-08-13)** — `requirements.txt` pins `scikit-learn==1.9.0`.
6. **Anomaly artifact metadata.** `evaluation.json` has no `feature_columns` key; the serving layer hardcodes `ANOMALY_FEATURE_COLS` (24) in `app/services/anomaly_service.py`. Move the column list into `metadata.json` on the next retrain.
7. **Forecaster CI method.** Prediction intervals come from RF forest percentiles (bagging variance) rather than a true quantile LSTM. Adequate for v1, documented in `app/services/forecast_service.py`.