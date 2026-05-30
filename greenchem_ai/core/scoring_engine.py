from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


WEIGHTS = {
    "solvent_green": 0.22,
    "toxicity": 0.18,
    "biodegradability": 0.12,
    "regulatory": 0.12,
    "yield": 0.14,
    "efactor": 0.12,
    "atom_economy": 0.10,
}


@dataclass(frozen=True)
class ProcessMetrics:
    green_score: float
    toxicity_score: float
    e_factor: float
    atom_economy: float
    yield_percent: float
    contributions: dict[str, float]


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, value)))


def calculate_e_factor(waste_kg: float, yield_percent: float) -> float:
    """Approximate E-Factor as waste mass per useful product mass."""
    yield_fraction = max(float(yield_percent), 1.0) / 100.0
    estimated_product_kg = max(1.0, 100.0 * yield_fraction)
    return round(float(waste_kg) / estimated_product_kg, 3)


def efactor_to_score(e_factor: float) -> float:
    # 0 is ideal, 8+ is poor for this small process-screening MVP.
    return clamp(100.0 - (float(e_factor) / 8.0) * 100.0)


def calculate_process_score(
    solvent: pd.Series,
    reaction: pd.Series,
    yield_percent: float,
    waste_kg: float,
    atom_economy: float | None = None,
) -> ProcessMetrics:
    e_factor = calculate_e_factor(waste_kg=waste_kg, yield_percent=yield_percent)
    atom = float(atom_economy if atom_economy is not None else reaction["atom_economy"])

    contribution_inputs = {
        "solvent_green": clamp(solvent["green_score"]),
        "toxicity": 100.0 - clamp(solvent["toxicity_score"]),
        "biodegradability": clamp(solvent["biodegradability_score"]),
        "regulatory": 100.0 - clamp(solvent["regulatory_risk"]),
        "yield": clamp(yield_percent),
        "efactor": efactor_to_score(e_factor),
        "atom_economy": clamp(atom),
    }
    contributions = {
        key: round(value * WEIGHTS[key], 2) for key, value in contribution_inputs.items()
    }
    score = round(float(np.sum(list(contributions.values()))), 2)
    return ProcessMetrics(
        green_score=clamp(score),
        toxicity_score=round(float(solvent["toxicity_score"]), 2),
        e_factor=e_factor,
        atom_economy=round(atom, 2),
        yield_percent=round(float(yield_percent), 2),
        contributions=contributions,
    )


def estimate_optimized_yield(current_yield: float, solvent: pd.Series, reaction: pd.Series) -> float:
    base = max(float(current_yield), float(reaction["average_yield"]) - 4.0)
    green_bonus = (float(solvent["green_score"]) - 50.0) / 100.0 * 4.0
    risk_penalty = float(solvent["regulatory_risk"]) / 100.0 * 2.0
    return round(clamp(base + green_bonus - risk_penalty, 1.0, 98.0), 2)


def estimate_optimized_waste(current_waste_kg: float, solvent: pd.Series, reaction: pd.Series) -> float:
    solvent_factor = 1.0 - ((float(solvent["green_score"]) - 50.0) / 100.0 * 0.18)
    reaction_factor = max(0.65, float(reaction["waste_factor"]) / 3.5)
    estimated = float(current_waste_kg) * solvent_factor * reaction_factor
    return round(max(0.1, estimated), 2)
