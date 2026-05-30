from __future__ import annotations

import requests

from ai.prompts import EXPLANATION_SYSTEM_PROMPT, build_explanation_prompt


def fallback_explanation(
    current: dict,
    best: dict,
    xai: dict,
    rag_context: list[str] | None = None,
    expert_context: list[str] | None = None,
) -> str:
    tox_delta = xai["toxicity_improvement"]
    score_delta = xai["green_score_improvement"]
    ef_delta = xai["e_factor_improvement"]
    waste_delta = current["waste_kg"] - best["waste_kg"]
    rag_context = rag_context or []
    expert_context = expert_context or []
    science_line = (
        f"Scientific context retrieved for this explanation emphasizes that {rag_context[0].lower()}"
        if rag_context
        else "Scientific context retrieval was unavailable, so the explanation uses the calculated process metrics only."
    )
    expert_line = (
        f"Expert memory was also retrieved: {expert_context[0]}"
        if expert_context
        else "No matching expert-memory record was retrieved for this exact recommendation yet."
    )
    return (
        f"{best['solvent']} was selected by the deterministic scoring engine, not by the LLM.\n\n"
        f"Decision basis: the platform first filtered solvents for compatibility with "
        f"{current['reaction_type']}, then ranked the remaining candidates using GreenScore, toxicity, "
        f"biodegradability, regulatory risk, yield, E-Factor, atom economy, and expert-memory adjustment.\n\n"
        f"Main improvement: compared with {current['solvent']}, the recommended process improves the "
        f"GreenScore by {score_delta:.1f} points, reduces the toxicity score by {tox_delta:.1f}, "
        f"reduces the estimated E-Factor by {ef_delta:.3f}, and lowers estimated waste by "
        f"{waste_delta:.1f} kg.\n\n"
        f"Scientific rationale: {science_line} This supports the recommendation because lower hazard, "
        f"lower waste intensity, and better solvent sustainability are aligned with green chemistry screening.\n\n"
        f"Adaptive context: {expert_line}\n\n"
        "Human validation remains required. A chemist should confirm solubility, conversion, selectivity, "
        "workup behavior, safety, cost, and scale-up constraints before adopting the substitution."
    )


def explain_with_ollama(
    current: dict,
    best: dict,
    xai: dict,
    rag_context: list[str],
    expert_context: list[str] | None = None,
    model_name: str = "llama3",
    host: str = "http://localhost:11434",
    timeout: int = 20,
) -> tuple[str, bool]:
    prompt = build_explanation_prompt(current, best, xai, rag_context)
    try:
        response = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={
                "model": model_name,
                "prompt": f"{EXPLANATION_SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 220},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        if not text:
            raise ValueError("Ollama returned an empty explanation.")
        return text, True
    except Exception:
        return fallback_explanation(current, best, xai, rag_context, expert_context), False
