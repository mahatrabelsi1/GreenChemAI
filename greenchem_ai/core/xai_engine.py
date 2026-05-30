from __future__ import annotations


def build_decision_trace(current: dict, candidate: dict, rank_score: float, feedback_adjustment: float) -> list[str]:
    return [
        f"Filter: candidate is compatible with {candidate['reaction_type']}.",
        f"Toxicity change: {current['toxicity_score']:.1f} -> {candidate['toxicity_score']:.1f}.",
        f"GreenScore change: {current['green_score']:.1f} -> {candidate['green_score']:.1f}.",
        f"E-Factor change: {current['e_factor']:.3f} -> {candidate['e_factor']:.3f}.",
        f"Rank score after deterministic scoring and feedback adjustment: {rank_score:.2f}.",
        f"Feedback adjustment applied: {feedback_adjustment:+.2f}.",
    ]


def recommendation_xai(best: dict, current: dict) -> dict:
    return {
        "selected_solvent": best["solvent"],
        "old_solvent": current["solvent"],
        "new_solvent": best["solvent"],
        "toxicity_improvement": round(current["toxicity_score"] - best["toxicity_score"], 2),
        "green_score_improvement": round(best["green_score"] - current["green_score"], 2),
        "e_factor_improvement": round(current["e_factor"] - best["e_factor"], 3),
        "yield_assumption": best["yield_percent"],
        "atom_economy": best["atom_economy"],
        "decision_trace": best["decision_trace"],
        "limitations": [
            "Compatibility is screened from a small curated dataset and must be validated experimentally.",
            "Yield and waste changes are transparent approximations, not reaction-specific kinetic simulations.",
            "QSAR toxicity estimate is trained on the bundled solvent table for MVP demonstration.",
        ],
    }
