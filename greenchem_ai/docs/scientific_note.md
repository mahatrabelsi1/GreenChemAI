# Scientific Note

GreenChem AI is an Explainable Green Chemistry Decision Support System for early green chemistry screening. It focuses on prevention by identifying potentially safer solvent substitutions before experimentation, and process optimization by estimating improved waste and performance metrics.

GreenChem AI combines deterministic scientific scoring, solvent similarity analysis, scientific literature retrieval, expert-memory learning, and human validation to help chemists identify safer and more sustainable solvent alternatives before experimentation.

## Deterministic Metrics

The platform calculates:

- GreenScore from weighted solvent and process factors
- Toxicity score from curated local solvent data
- QSAR-style toxicity estimate from a Random Forest trained on local molecular descriptors
- E-Factor approximation from waste and estimated product mass
- Atom economy approximation from the reaction class database

## Solvent Ranking

Candidate solvents are first filtered by compatibility with the selected reaction class. They are then scored using transparent numerical terms:

- Optimized GreenScore
- Toxicity improvement versus current solvent
- E-Factor improvement versus current process
- Human feedback adjustment from previous validations

The final rank score uses a 50/50 multi-criteria structure:

```text
Rank Score =
0.5 x Candidate GreenScore
+ 0.5 x Improvement Score
+ Expert Memory Adjustment

Improvement Score =
0.5 x Toxicity Improvement Score
+ 0.5 x Normalized E-Factor Improvement Score
```

This balances absolute candidate quality with direct substitution benefit. E-Factor improvement is normalized before weighting because E-Factor and toxicity use different numerical scales.

The LLM is not part of the ranking function.

## Adaptive Expert Memory

Human validation is stored in `data/feedback.csv` and retrieved in later predictions as expert memory. Accepted solvent recommendations receive a positive deterministic adjustment in similar future reaction contexts. Rejected recommendations and requests for alternatives receive negative adjustments. This lets the platform adapt to expert judgment without allowing the LLM to make scientific decisions.

## First-Part Backend Bridge

The project imports the first application layer as a supporting evidence bridge. The bridge does not replace the GreenScore engine. It adds extra explainability from an enriched solvent table:

- Hansen solubility parameter distance between the current and recommended solvent
- Density-based backend E-Factor estimate
- Solvent family metadata
- Simple structural alerts from SMILES patterns

These values appear in the XAI workspace, scientific sources, Llama explanation context, and PDF report. They help a chemist inspect whether a substitution is chemically plausible before laboratory validation.

## Limitations

This MVP is not a substitute for laboratory validation. The compatibility table is curated for demonstration, not exhaustive. Yield and waste optimization are approximate. Toxicity modeling is intentionally lightweight and trained only on the bundled solvent table. A production version should use larger toxicology datasets, reaction-specific experimental data, uncertainty estimates, and safety review workflows.

## Green Chemistry Relevance

The workflow supports:

- Prevention: screen hazards before experiments
- Atom economy: expose reaction-class material efficiency
- Safer solvents: replace high-risk solvents with better alternatives
- Waste reduction: use E-Factor as a visible optimization target
- Human oversight: capture chemist validation and rejections

## AI4Green Solvent Flashcards

The MVP includes a local flashcard-style solvent comparison page inspired by the AI4Green Solvent Guide. It uses the bundled solvent table rather than copying AI4Green source code, keeping the feature lightweight while still giving users a visual solvent hazard and sustainability view.
