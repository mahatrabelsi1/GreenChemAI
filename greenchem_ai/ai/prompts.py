EXPLANATION_SYSTEM_PROMPT = """
You explain already calculated green chemistry results.
Never invent calculations, rankings, solvent choices, or missing values.
State that the recommendation came from deterministic scoring.
Keep the explanation short, practical, and suitable for a hackathon demo.
Use clear section-like paragraphs: decision basis, main improvement, scientific rationale, human validation.
"""


def build_explanation_prompt(current: dict, best: dict, xai: dict, rag_context: list[str]) -> str:
    context = "\n".join(f"- {chunk}" for chunk in rag_context[:3])
    trace = "\n".join(f"- {item}" for item in xai["decision_trace"])
    return f"""
Computed current process:
- Reaction: {current['reaction_type']}
- Current solvent: {current['solvent']}
- Current GreenScore: {current['green_score']}
- Current toxicity score: {current['toxicity_score']}
- Current E-Factor: {current['e_factor']}
- Atom economy: {current['atom_economy']}

Computed best recommendation:
- Recommended solvent: {best['solvent']}
- Recommended GreenScore: {best['green_score']}
- Recommended toxicity score: {best['toxicity_score']}
- Recommended E-Factor: {best['e_factor']}
- Assumed yield: {best['yield_percent']}
- Waste estimate: {best['waste_kg']} kg

Deterministic decision trace:
{trace}

Retrieved reference context:
{context}

Write a concise explanation of why the selected solvent is preferred. Do not recalculate anything.
"""
