from __future__ import annotations

import pandas as pd

from core.feedback_engine import expert_memory_matches, feedback_adjustment, load_feedback
from core.reaction_database import compatible_solvents
from core.scoring_engine import (
    calculate_process_score,
    estimate_optimized_waste,
    estimate_optimized_yield,
)
from core.toxicity_model import predict_toxicity
from core.xai_engine import build_decision_trace


def analyze_current_process(
    reaction: pd.Series,
    solvent: pd.Series,
    yield_percent: float,
    waste_kg: float,
    solvents: pd.DataFrame,
) -> dict:
    metrics = calculate_process_score(solvent, reaction, yield_percent, waste_kg)
    qsar_toxicity = predict_toxicity(solvent["smiles"], solvents)
    return {
        "reaction_type": reaction["reaction_type"],
        "solvent": solvent["name"],
        "green_score": metrics.green_score,
        "toxicity_score": metrics.toxicity_score,
        "qsar_toxicity_estimate": qsar_toxicity,
        "e_factor": metrics.e_factor,
        "atom_economy": metrics.atom_economy,
        "yield_percent": metrics.yield_percent,
        "waste_kg": float(waste_kg),
        "contributions": metrics.contributions,
    }


def recommend_solvents(
    reaction: pd.Series,
    current_solvent: pd.Series,
    current_yield: float,
    current_waste_kg: float,
    solvents: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    compatible = set(compatible_solvents(reaction))
    feedback = load_feedback()
    current = analyze_current_process(reaction, current_solvent, current_yield, current_waste_kg, solvents)
    candidates: list[dict] = []

    for _, solvent in solvents.iterrows():
        if solvent["name"] == current_solvent["name"]:
            continue
        if compatible and solvent["name"] not in compatible:
            continue

        opt_yield = estimate_optimized_yield(current_yield, solvent, reaction)
        opt_waste = estimate_optimized_waste(current_waste_kg, solvent, reaction)
        metrics = calculate_process_score(solvent, reaction, opt_yield, opt_waste)
        toxicity_estimate = predict_toxicity(solvent["smiles"], solvents)
        adjustment = feedback_adjustment(
            solvent["name"],
            reaction["reaction_type"],
            feedback,
            current_solvent=current_solvent["name"],
        )
        memories = expert_memory_matches(
            reaction["reaction_type"],
            current_solvent["name"],
            recommended_solvent=solvent["name"],
            feedback=feedback,
            limit=3,
        )
        toxicity_improvement_score = max(
            0.0,
            current["toxicity_score"] - float(solvent["toxicity_score"]),
        )
        efactor_improvement_score = 0.0
        if current["e_factor"] > 0:
            efactor_improvement_score = (
                max(0.0, current["e_factor"] - metrics.e_factor)
                / current["e_factor"]
                * 100.0
            )
        improvement_score = (
            0.5 * toxicity_improvement_score
            + 0.5 * min(100.0, efactor_improvement_score)
        )
        rank_score = (
            0.5 * metrics.green_score
            + 0.5 * improvement_score
            + adjustment
        )
        candidate = {
            "reaction_type": reaction["reaction_type"],
            "solvent": solvent["name"],
            "chem21_classification": solvent["chem21_classification"],
            "gsk_score": float(solvent["gsk_score"]),
            "green_score": metrics.green_score,
            "toxicity_score": metrics.toxicity_score,
            "qsar_toxicity_estimate": toxicity_estimate,
            "e_factor": metrics.e_factor,
            "atom_economy": metrics.atom_economy,
            "yield_percent": metrics.yield_percent,
            "waste_kg": opt_waste,
            "contributions": metrics.contributions,
            "feedback_adjustment": adjustment,
            "expert_memory_matches": memories,
            "toxicity_improvement_score": round(toxicity_improvement_score, 2),
            "efactor_improvement_score": round(min(100.0, efactor_improvement_score), 2),
            "improvement_score": round(improvement_score, 2),
            "rank_score": round(rank_score, 2),
        }
        candidate["decision_trace"] = build_decision_trace(current, candidate, rank_score, adjustment)
        candidates.append(candidate)

    return sorted(candidates, key=lambda item: item["rank_score"], reverse=True)[:top_n]
