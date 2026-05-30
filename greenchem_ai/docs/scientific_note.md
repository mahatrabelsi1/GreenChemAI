# Scientific Note

GreenChem AI is a local decision-support MVP for early green chemistry screening. It focuses on prevention by identifying potentially safer solvent substitutions before experimentation, and process optimization by estimating improved waste and performance metrics.

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

The LLM is not part of the ranking function.

## Adaptive Expert Memory

Human validation is stored in `data/feedback.csv` and retrieved in later predictions as expert memory. Accepted solvent recommendations receive a positive deterministic adjustment in similar future reaction contexts. Rejected recommendations and requests for alternatives receive negative adjustments. This lets the platform adapt to expert judgment without allowing the LLM to make scientific decisions.

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
