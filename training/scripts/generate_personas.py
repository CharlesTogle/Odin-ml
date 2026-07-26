"""
Synthetic Persona Generator for Odin ML Models

Reads FIES NCR data and generates 12 archetypes with synthetic personas.
Archetypes follow the 2x2x2 PFP matrix (Stability x Obligation x Tolerance)
as defined in persona-validation-list-SME-draft.md.

Thresholds (provisional, pending SME validation):
  - Stability: CV < 0.5 = Stable, CV >= 0.5 = Variable
  - Obligation: ratio > 0.6 = Obligated, ratio <= 0.6 = Flexible
  - Tolerance: runway >= 3 months = Tolerant, runway < 3 months = Tight

Uses FIES variable IDs (TOINC, FOOD, etc.) for column access.
See: Odin-Paper/Model/2_Data Collection/FIES Dictionary & Valueset.csv

Data Sources:
  - PSA 2023 FIES NCR (41,380 households) — financial numerical baselines
  - BSP 2021 Consumer Finance Survey — behavioral/attitudinal patterns

Usage:
    python scripts/generate_personas.py --input <fies_csv> --output <output_dir>
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fies_columns import (
    FIESColumns,
    FIESColumnError,
    FIESColumnWarning,
    validate_columns,
)


class PersonaGenerationError(Exception):
    """Raised when persona generation fails."""
    pass


# ---------------------------------------------------------------------------
# PFP dimension thresholds (provisional — pending SME validation in Phase 4.5)
# ---------------------------------------------------------------------------

STABILITY_CV_THRESHOLD = 0.5    # CV < 0.5 = Stable, >= 0.5 = Variable
OBLIGATION_RATIO_THRESHOLD = 0.6  # ratio > 0.6 = Obligated, <= 0.6 = Flexible
TOLERANCE_RUNWAY_MONTHS = 3.0   # runway >= 3 months = Tolerant, < 3 = Tight

# 8 PFP classes (2 x 2 x 2)
PFP_CLASSES = [
    "Stable/Flexible/Tolerant",
    "Stable/Flexible/Tight",
    "Stable/Obligated/Tolerant",
    "Stable/Obligated/Tight",
    "Variable/Flexible/Tolerant",
    "Variable/Flexible/Tight",
    "Variable/Obligated/Tolerant",
    "Variable/Obligated/Tight",
]


@dataclass
class PersonaArchetype:
    """Defines a financial behavior archetype from the SME draft."""
    archetype_id: str
    name: str
    income_range: tuple[float, float]
    income_cv: float
    obligation_ratio: float
    savings_rate: float
    runway_months: float
    household_size: tuple[int, int]
    employment_type: str
    description: str
    expected_pfp: str
    required_fies_ids: tuple[str, ...] = ("TOINC", "FOOD", "HOUSING_WATER", "TRANSPORT", "HEALTH", "EDUCATION")


# ---------------------------------------------------------------------------
# 12 archetypes from persona-validation-list-SME-draft.md
# Each maps to a specific 2x2x2 cell in the PFP matrix.
# Income ranges calibrated to FIES NCR 2023 decile distribution.
# Behavioral patterns calibrated to BSP CFS 2021.
# ---------------------------------------------------------------------------

ARCHETYPES = [
    # A: Stable / Obligated / Tolerant
    PersonaArchetype(
        archetype_id="A",
        name="BPO Employee, Moderate Obligations, Healthy Fund",
        income_range=(35000, 45000),  # FIES NCR D5-D6
        income_cv=0.10,
        obligation_ratio=0.70,
        savings_rate=0.10,
        runway_months=5.0,
        household_size=(1, 2),
        employment_type="full_time",
        description="Regular BPO employee, 2yr tenure, consistent salary with minor overtime variation. Single, no dependents. CFS: top 2% for emergency runway.",
        expected_pfp="Stable/Obligated/Tolerant",
    ),
    # B: Stable / Obligated / Tight
    PersonaArchetype(
        archetype_id="B",
        name="Manufacturing Worker, Heavy Obligations, No Savings",
        income_range=(25000, 35000),  # FIES NCR D4-D5
        income_cv=0.15,
        obligation_ratio=0.85,
        savings_rate=0.00,
        runway_months=0.5,
        household_size=(3, 4),
        employment_type="full_time",
        description="Regular manufacturing employee, 5yr tenure. Married, 1 child. Entire paycheck goes to obligations. CFS: typical debt profile.",
        expected_pfp="Stable/Obligated/Tight",
    ),
    # C: Stable / Flexible / Tolerant
    PersonaArchetype(
        archetype_id="C",
        name="Tech Employee, Low Obligations, Strong Savings",
        income_range=(55000, 70000),  # FIES NCR D7-D9
        income_cv=0.08,
        obligation_ratio=0.45,
        savings_rate=0.30,
        runway_months=9.0,
        household_size=(1, 1),
        employment_type="full_time",
        description="Regular tech company employee, 3yr tenure. Single, lives with parents (no rent). High savings rate. CFS: top 2% for financial health.",
        expected_pfp="Stable/Flexible/Tolerant",
    ),
    # D: Stable / Flexible / Tight
    PersonaArchetype(
        archetype_id="D",
        name="Government Employee, Low Obligations, Minimal Savings",
        income_range=(30000, 40000),  # FIES NCR D5-D6
        income_cv=0.12,
        obligation_ratio=0.50,
        savings_rate=0.05,
        runway_months=1.5,
        household_size=(1, 2),
        employment_type="full_time",
        description="Government agency employee, 4yr tenure. Very consistent salary. Single. Spends discretionary freely. CFS: typical digital services user.",
        expected_pfp="Stable/Flexible/Tight",
    ),
    # E: Variable / Obligated / Tolerant
    PersonaArchetype(
        archetype_id="E",
        name="Freelancer, High Obligations, Adequate Buffer",
        income_range=(25000, 60000),  # FIES NCR D4-D7
        income_cv=0.70,
        obligation_ratio=0.75,
        savings_rate=0.14,
        runway_months=4.0,
        household_size=(1, 2),
        employment_type="freelancer",
        description="Freelance graphic designer, 3yr self-employed. 3 retainer clients. Income varies monthly. CFS: above-average financial risk tolerance.",
        expected_pfp="Variable/Obligated/Tolerant",
    ),
    # F: Variable / Obligated / Tight
    PersonaArchetype(
        archetype_id="F",
        name="Contract Worker, High Obligations, Paycheck-to-Paycheck",
        income_range=(18000, 40000),  # FIES NCR D3-D5
        income_cv=0.65,
        obligation_ratio=0.80,
        savings_rate=0.00,
        runway_months=0.3,
        household_size=(2, 3),
        employment_type="informal",
        description="Fixed-term construction contracts. Income gaps between contracts. Married, spouse part-time vendor. CFS: below-average deposit ownership.",
        expected_pfp="Variable/Obligated/Tight",
    ),
    # G: Variable / Flexible / Tolerant
    PersonaArchetype(
        archetype_id="G",
        name="Freelance Writer/VA, Low Obligations, Healthy Fund",
        income_range=(20000, 45000),  # FIES NCR D3-D6
        income_cv=0.60,
        obligation_ratio=0.40,
        savings_rate=0.25,
        runway_months=7.0,
        household_size=(1, 2),
        employment_type="freelancer",
        description="Freelance content writer + part-time VA, 2yr self-employed. Built reserves in high-earning months. CFS: top 2% for financial resilience.",
        expected_pfp="Variable/Flexible/Tolerant",
    ),
    # H: Variable / Flexible / Tight
    PersonaArchetype(
        archetype_id="H",
        name="Tricycle Driver/Vendor, No Emergency Fund",
        income_range=(8000, 20000),  # FIES NCR D1-D3
        income_cv=0.80,
        obligation_ratio=0.45,
        savings_rate=0.00,
        runway_months=1.0,
        household_size=(1, 2),
        employment_type="gig_worker",
        description="Part-time tricycle driver and occasional market vendor. No formal contract. Highly variable daily income. CFS: typical low-income profile.",
        expected_pfp="Variable/Flexible/Tight",
    ),
    # I: Variable / Obligated / Tight (edge case — recovering)
    PersonaArchetype(
        archetype_id="I",
        name="Recovering from Financial Shock, Depleted Savings",
        income_range=(20000, 35000),  # FIES NCR D3-D5
        income_cv=0.70,
        obligation_ratio=0.78,
        savings_rate=0.02,
        runway_months=0.8,
        household_size=(2, 3),
        employment_type="full_time",
        description="Recently re-employed (3mo tenure). Laid off 4mo ago, depleted savings. Married, spouse also re-employed. CFS: transitional financial shock.",
        expected_pfp="Variable/Obligated/Tight",
    ),
    # J: Variable / Flexible / Tight (edge case — borderline tolerance)
    PersonaArchetype(
        archetype_id="J",
        name="Part-time Sales + Online Selling, Borderline Tolerance",
        income_range=(15000, 30000),  # FIES NCR D2-D4
        income_cv=0.55,
        obligation_ratio=0.55,
        savings_rate=0.05,
        runway_months=2.5,
        household_size=(1, 2),
        employment_type="sales",
        description="Part-time mall sales associate + occasional online selling. Low obligations but inconsistent savings. CFS: typical young adult profile.",
        expected_pfp="Variable/Flexible/Tight",
    ),
    # K: Stable / Obligated / Tolerant (edge case — near threshold)
    PersonaArchetype(
        archetype_id="K",
        name="Telecom Employee, High Obligations Near Threshold",
        income_range=(45000, 55000),  # FIES NCR D6-D7
        income_cv=0.10,
        obligation_ratio=0.65,
        savings_rate=0.05,
        runway_months=4.0,
        household_size=(3, 4),
        employment_type="full_time",
        description="Telecom employee, 4yr tenure. Married (spouse freelance tutor), 1 child. Obligations near threshold. CFS: mortgage + car loan typical.",
        expected_pfp="Stable/Obligated/Tolerant",
    ),
    # L: Stable / Flexible / Tight (edge case — no savings habit)
    PersonaArchetype(
        archetype_id="L",
        name="Marketing Agency, No Savings Habit Despite Stable Income",
        income_range=(35000, 50000),  # FIES NCR D5-D7
        income_cv=0.10,
        obligation_ratio=0.50,
        savings_rate=0.02,
        runway_months=1.0,
        household_size=(1, 1),
        employment_type="full_time",
        description="Marketing agency employee, 2yr tenure. Single, rents studio. Spends discretionary freely, no buffer. CFS: typical discretionary spender.",
        expected_pfp="Stable/Flexible/Tight",
    ),
]


@dataclass
class SyntheticPersona:
    """A generated synthetic persona."""
    persona_id: str
    archetype_id: str
    archetype_name: str
    monthly_income: float
    income_cv: float
    obligation_ratio: float
    savings_rate: float
    runway_months: float
    household_size: int
    employment_type: str
    age: int
    gender: str
    education: str
    fbp_label: str
    # Expense breakdown
    food_expense: float
    housing_expense: float
    transport_expense: float
    health_expense: float
    education_expense: float
    other_expense: float
    # Financial metrics
    debt_amount: float
    savings_amount: float
    emergency_fund: float
    # Metadata
    fies_income_used: Optional[str] = None


def load_fies_data(file_path: str) -> pd.DataFrame:
    """
    Load FIES data with error handling. Auto-detects CSV or Parquet format.

    Args:
        file_path: Path to FIES CSV or Parquet file

    Returns:
        pandas DataFrame with FIES data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or malformed
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"FIES data file not found: {file_path}\n"
            f"Please download from PSA website or provide correct path."
        )

    if path.stat().st_size == 0:
        raise ValueError(f"FIES data file is empty: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise ValueError(f"Failed to read Parquet file: {e}")
    elif suffix == ".csv":
        try:
            df = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            raise ValueError(f"FIES data file is empty or malformed: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse FIES CSV: {e}")
    else:
        raise ValueError(f"Unsupported file format: {suffix} (expected .csv or .parquet)")

    if len(df) == 0:
        raise ValueError(f"FIES data file contains no rows: {file_path}")

    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


def filter_ncr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data for NCR (National Capital Region) using FIES IDs.

    Args:
        df: Full FIES DataFrame

    Returns:
        Filtered DataFrame with NCR data only

    Raises:
        FIESColumnError: If region column not found
    """
    fies = FIESColumns(df)

    region_col = fies.get("W_REGN")
    if region_col is None:
        raise FIESColumnError(
            "Cannot filter NCR: Region column (W_REGN) not found.\n"
            f"Available columns: {list(df.columns[:10])}..."
        )

    ncr_patterns = ["NCR", "National Capital Region", "13"]
    ncr_mask = pd.Series([False] * len(df), index=df.index)

    for pattern in ncr_patterns:
        if region_col.dtype == object:
            ncr_mask |= region_col.str.contains(pattern, case=False, na=False)
        else:
            try:
                ncr_mask |= region_col.astype(str).str.startswith(pattern)
            except (ValueError, TypeError):
                continue

    ncr_df = df[ncr_mask].copy()

    if len(ncr_df) == 0:
        warnings.warn(
            "No NCR data found. Using all regions.\n"
            f"Region values found: {region_col.unique()[:10]}",
            FIESColumnWarning,
        )
        return df.copy()

    print(f"Filtered to {len(ncr_df):,} NCR rows")
    return ncr_df


def compute_fies_statistics(df: pd.DataFrame) -> dict:
    """Compute aggregate statistics from FIES data using FIES IDs."""
    fies = FIESColumns(df)

    income = fies.get("TOINC")
    food = fies.get("FOOD")
    housing = fies.get("HOUSING_WATER")
    transport = fies.get("TRANSPORT")
    health = fies.get("HEALTH")
    education = fies.get("EDUCATION")
    household_size = fies.get("HH_SIZE")

    stats = {
        "total_households": len(df),
        "income": _compute_column_stats(income, "income"),
        "food_expense": _compute_column_stats(food, "food"),
        "housing_expense": _compute_column_stats(housing, "housing"),
        "transport_expense": _compute_column_stats(transport, "transport"),
        "health_expense": _compute_column_stats(health, "health"),
        "education_expense": _compute_column_stats(education, "education"),
        "household_size": _compute_column_stats(household_size, "household_size"),
    }

    if income is not None and food is not None:
        total_expenses = pd.Series([0.0] * len(df), index=df.index)
        for col in [food, housing, transport, health, education]:
            if col is not None:
                total_expenses = total_expenses.add(col, fill_value=0)

        valid_mask = (income > 0) & (total_expenses > 0)
        if valid_mask.any():
            savings_rate = (income[valid_mask] - total_expenses[valid_mask]) / income[valid_mask]
            stats["savings_rate"] = {
                "mean": float(savings_rate.mean()),
                "median": float(savings_rate.median()),
            }
        else:
            stats["savings_rate"] = {"mean": 0.0, "median": 0.0}
    else:
        stats["savings_rate"] = {"mean": 0.0, "median": 0.0}

    return stats


def _compute_column_stats(series: Optional[pd.Series], name: str) -> dict:
    """Compute statistics for a single column with error handling."""
    if series is None:
        warnings.warn(f"Column '{name}' not found, using default statistics")
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}

    try:
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_series) == 0:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}

        return {
            "mean": float(numeric_series.mean()),
            "median": float(numeric_series.median()),
            "std": float(numeric_series.std()),
            "min": float(numeric_series.min()),
            "max": float(numeric_series.max()),
            "q25": float(numeric_series.quantile(0.25)),
            "q75": float(numeric_series.quantile(0.75)),
        }
    except Exception as e:
        warnings.warn(f"Error computing statistics for '{name}': {e}")
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}


def compute_expense_ratios(df: pd.DataFrame) -> dict:
    """Compute average expense category ratios from FIES data using FIES IDs."""
    fies = FIESColumns(df)

    income = fies.get("TOINC")
    food = fies.get("FOOD")
    housing = fies.get("HOUSING_WATER")
    transport = fies.get("TRANSPORT")
    health = fies.get("HEALTH")
    education = fies.get("EDUCATION")

    if income is None or income.sum() == 0:
        warnings.warn("Cannot compute expense ratios: income data missing or zero")
        return {"food": 0.35, "housing": 0.20, "transport": 0.10, "health": 0.05, "education": 0.08, "other": 0.22}

    total_income = income.sum()

    def safe_sum(series):
        return series.sum() if series is not None else 0

    ratios = {
        "food": safe_sum(food) / total_income,
        "housing": safe_sum(housing) / total_income,
        "transport": safe_sum(transport) / total_income,
        "health": safe_sum(health) / total_income,
        "education": safe_sum(education) / total_income,
        "other": 0.22,
    }

    return ratios


def _compute_pfp_label(
    income_cv: float,
    obligation_ratio: float,
    runway_months: float,
) -> str:
    """Compute 8-class PFP label from three binary dimensions."""
    stability = "Stable" if income_cv < STABILITY_CV_THRESHOLD else "Variable"
    obligation = "Obligated" if obligation_ratio > OBLIGATION_RATIO_THRESHOLD else "Flexible"
    tolerance = "Tolerant" if runway_months >= TOLERANCE_RUNWAY_MONTHS else "Tight"
    return f"{stability}/{obligation}/{tolerance}"


def generate_persona(
    archetype: PersonaArchetype,
    persona_index: int,
    fies_stats: dict,
    expense_ratios: dict,
    rng: np.random.Generator,
) -> SyntheticPersona:
    """Generate a single synthetic persona from an archetype."""
    # Income with noise
    base_income = rng.uniform(archetype.income_range[0], archetype.income_range[1])
    monthly_income = round(base_income / 1000) * 1000

    # Income CV with noise (clamped to valid range)
    income_cv = float(np.clip(
        archetype.income_cv + rng.normal(0, 0.03), 0.01, 1.5
    ))

    # Obligation ratio with noise
    obligation_ratio = float(np.clip(
        archetype.obligation_ratio + rng.normal(0, 0.03), 0.05, 0.95
    ))

    # Emergency runway with noise (months of expenses covered by savings)
    runway_months = float(np.clip(
        archetype.runway_months + rng.normal(0, 0.3), 0.0, 24.0
    ))

    # Household size
    household_size = int(rng.integers(
        archetype.household_size[0], archetype.household_size[1] + 1
    ))

    # Age based on employment type
    age_ranges = {
        "full_time": (25, 55),
        "informal": (18, 50),
        "gig_worker": (20, 40),
        "freelancer": (25, 45),
        "sales": (22, 40),
    }
    age_range = age_ranges.get(archetype.employment_type, (25, 50))
    age = int(rng.integers(age_range[0], age_range[1] + 1))

    # Gender
    gender = rng.choice(["Male", "Female"], p=[0.48, 0.52])

    # Education level
    education_levels = [
        "High School Graduate",
        "Some College",
        "Bachelor's Degree",
        "Vocational/Technical",
    ]
    education = rng.choice(education_levels)

    # Compute expense breakdown based on FIES ratios and archetype
    total_income = monthly_income

    food_ratio = expense_ratios["food"] * (0.8 + 0.4 * (1 - obligation_ratio))
    housing_ratio = expense_ratios["housing"] * (0.7 + 0.6 * obligation_ratio)
    transport_ratio = expense_ratios["transport"]
    health_ratio = expense_ratios["health"]
    education_ratio = expense_ratios["education"]
    other_ratio = expense_ratios["other"]

    total_expense_ratio = food_ratio + housing_ratio + transport_ratio + health_ratio + education_ratio + other_ratio
    food_ratio /= total_expense_ratio
    housing_ratio /= total_expense_ratio
    transport_ratio /= total_expense_ratio
    health_ratio /= total_expense_ratio
    education_ratio /= total_expense_ratio
    other_ratio /= total_expense_ratio

    # Actual expenses with noise
    base_expense = total_income * (1 - archetype.savings_rate)
    food_expense = round(base_expense * food_ratio * rng.uniform(0.85, 1.15) / 100) * 100
    housing_expense = round(base_expense * housing_ratio * rng.uniform(0.85, 1.15) / 100) * 100
    transport_expense = round(base_expense * transport_ratio * rng.uniform(0.8, 1.2) / 100) * 100
    health_expense = round(base_expense * health_ratio * rng.uniform(0.7, 1.3) / 100) * 100
    education_expense = round(base_expense * education_ratio * rng.uniform(0.6, 1.4) / 100) * 100
    other_expense = round(base_expense * other_ratio * rng.uniform(0.8, 1.2) / 100) * 100

    # Financial metrics
    total_expenses = food_expense + housing_expense + transport_expense + health_expense + education_expense + other_expense
    savings_amount = round(total_income * archetype.savings_rate * rng.uniform(0.5, 1.5) / 100) * 100
    debt_amount = round(total_expenses * obligation_ratio * rng.uniform(0.5, 1.5) / 100) * 100
    emergency_fund = round(savings_amount * rng.uniform(0.2, 0.8) / 100) * 100

    # 3-dimension PFP label
    fbp_label = _compute_pfp_label(income_cv, obligation_ratio, runway_months)

    persona_id = f"persona_{archetype.archetype_id}_{persona_index:04d}"

    return SyntheticPersona(
        persona_id=persona_id,
        archetype_id=archetype.archetype_id,
        archetype_name=archetype.name,
        monthly_income=monthly_income,
        income_cv=round(income_cv, 3),
        obligation_ratio=round(obligation_ratio, 3),
        savings_rate=round(archetype.savings_rate, 3),
        runway_months=round(runway_months, 2),
        household_size=household_size,
        employment_type=archetype.employment_type,
        age=age,
        gender=gender,
        education=education,
        fbp_label=fbp_label,
        food_expense=food_expense,
        housing_expense=housing_expense,
        transport_expense=transport_expense,
        health_expense=health_expense,
        education_expense=education_expense,
        other_expense=other_expense,
        debt_amount=debt_amount,
        savings_amount=savings_amount,
        emergency_fund=emergency_fund,
        fies_income_used=fies_stats.get("income", {}).get("mean", None),
    )


def generate_all_personas(
    personas_per_archetype: int = 1000,
    fies_stats: Optional[dict] = None,
    expense_ratios: Optional[dict] = None,
    seed: int = 42,
) -> list[SyntheticPersona]:
    """
    Generate all personas across all archetypes.

    Args:
        personas_per_archetype: Number of personas per archetype
        fies_stats: FIES statistics (uses defaults if None)
        expense_ratios: Expense ratios (uses defaults if None)
        seed: Random seed

    Returns:
        List of SyntheticPersona objects
    """
    rng = np.random.default_rng(seed)

    if fies_stats is None:
        warnings.warn("Using default FIES statistics")
        fies_stats = {
            "income": {"mean": 250000, "median": 200000},
        }
    if expense_ratios is None:
        warnings.warn("Using default expense ratios")
        expense_ratios = {
            "food": 0.35,
            "housing": 0.20,
            "transport": 0.10,
            "health": 0.05,
            "education": 0.08,
            "other": 0.22,
        }

    all_personas = []

    try:
        for archetype in ARCHETYPES:
            print(f"Generating {personas_per_archetype} personas for archetype {archetype.archetype_id} ({archetype.name})...")
            for i in range(personas_per_archetype):
                persona = generate_persona(
                    archetype, i, fies_stats, expense_ratios, rng
                )
                all_personas.append(persona)
    except Exception as e:
        raise PersonaGenerationError(f"Failed to generate personas: {e}")

    print(f"Generated {len(all_personas):,} total personas ({len(ARCHETYPES)} archetypes x {personas_per_archetype})")
    return all_personas


def validate_personas(personas: list[SyntheticPersona]) -> dict:
    """Validate generated personas against expected distributions."""
    df = pd.DataFrame([asdict(p) for p in personas])

    validation = {
        "total_personas": len(personas),
        "archetype_counts": df["archetype_id"].value_counts().to_dict(),
        "pfp_label_counts": df["fbp_label"].value_counts().to_dict(),
        "income_stats": {
            "mean": float(df["monthly_income"].mean()),
            "median": float(df["monthly_income"].median()),
            "std": float(df["monthly_income"].std()),
        },
        "obligation_ratio_stats": {
            "mean": float(df["obligation_ratio"].mean()),
            "median": float(df["obligation_ratio"].median()),
        },
        "income_cv_stats": {
            "mean": float(df["income_cv"].mean()),
            "median": float(df["income_cv"].median()),
        },
        "runway_months_stats": {
            "mean": float(df["runway_months"].mean()),
            "median": float(df["runway_months"].median()),
        },
    }

    return validation


def export_personas(personas: list[SyntheticPersona], output_dir: str) -> None:
    """Export personas to JSON and Parquet."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "personas.json"
    with open(json_path, "w") as f:
        json.dump([asdict(p) for p in personas], f, indent=2)
    print(f"Exported {len(personas)} personas to {json_path}")

    parquet_path = output_path / "personas.parquet"
    df = pd.DataFrame([asdict(p) for p in personas])
    df.to_parquet(parquet_path, index=False)
    print(f"Exported {len(personas)} personas to {parquet_path}")

    summary_path = output_path / "archetype_summary.json"
    summary = {}
    for archetype in ARCHETYPES:
        archetype_personas = [p for p in personas if p.archetype_id == archetype.archetype_id]
        summary[archetype.archetype_id] = {
            "name": archetype.name,
            "expected_pfp": archetype.expected_pfp,
            "count": len(archetype_personas),
            "avg_income": float(np.mean([p.monthly_income for p in archetype_personas])),
            "avg_obligation_ratio": float(np.mean([p.obligation_ratio for p in archetype_personas])),
            "avg_income_cv": float(np.mean([p.income_cv for p in archetype_personas])),
            "avg_runway_months": float(np.mean([p.runway_months for p in archetype_personas])),
            "pfp_labels": {
                label: sum(1 for p in archetype_personas if p.fbp_label == label)
                for label in set(p.fbp_label for p in archetype_personas)
            },
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Exported archetype summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic personas from FIES data")
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/raw/family_income_and_expenditure.csv",
        help="Path to FIES CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synth/",
        help="Output directory for generated personas",
    )
    parser.add_argument(
        "--personas-per-archetype",
        type=int,
        default=1000,
        help="Number of personas to generate per archetype",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-fies",
        action="store_true",
        help="Skip loading FIES data and use default statistics",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error on missing critical columns",
    )

    args = parser.parse_args()

    fies_stats = None
    expense_ratios = None

    if not args.skip_fies:
        try:
            df = load_fies_data(args.input)

            report = validate_columns(df, strict=args.strict)
            if not report["is_valid"]:
                print(f"Warning: Missing critical columns: {report['critical_missing']}")
                if args.strict:
                    sys.exit(1)

            ncr_df = filter_ncr(df)
            fies_stats = compute_fies_statistics(ncr_df)
            expense_ratios = compute_expense_ratios(ncr_df)

            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            stats_path = output_path / "fies_statistics.json"
            with open(stats_path, "w") as f:
                json.dump(fies_stats, f, indent=2)
            print(f"Saved FIES statistics to {stats_path}")

            ratios_path = output_path / "expense_ratios.json"
            with open(ratios_path, "w") as f:
                json.dump(expense_ratios, f, indent=2)
            print(f"Saved expense ratios to {ratios_path}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Falling back to default statistics")
        except FIESColumnError as e:
            print(f"FIES Column Error: {e}")
            print("Falling back to default statistics")
        except Exception as e:
            print(f"Unexpected error loading FIES data: {e}")
            print("Falling back to default statistics")

    try:
        personas = generate_all_personas(
            personas_per_archetype=args.personas_per_archetype,
            fies_stats=fies_stats,
            expense_ratios=expense_ratios,
            seed=args.seed,
        )
    except PersonaGenerationError as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

    validation = validate_personas(personas)
    print("\nValidation Results:")
    print(json.dumps(validation, indent=2))

    export_personas(personas, args.output)

    print("\nPersona generation complete!")


if __name__ == "__main__":
    main()
