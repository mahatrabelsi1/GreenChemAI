# Pitch Outline

## Problem

Many labs still select solvents from habit, availability, or legacy procedures. Hazardous solvents like DMF and dichloromethane can enter workflows before greener alternatives are evaluated.

## Solution

GreenChem AI is an Explainable Green Chemistry Decision Support System that screens solvent substitutions and process metrics before experimentation.

Memorized sentence:

GreenChem AI combines deterministic scientific scoring, solvent similarity analysis, scientific literature retrieval, expert-memory learning, and human validation to help chemists identify safer and more sustainable solvent alternatives before experimentation.

## Why It Is Not Just an LLM Wrapper

The LLM never ranks or calculates. Deterministic code filters compatible solvents, computes E-Factor, estimates atom economy, calculates GreenScore, and ranks alternatives. Llama only explains the computed outcome.

## Demo

Input:

- Esterification
- DMF
- 65% yield
- 40 kg waste

Output:

- Current GreenScore and risk profile
- Top 5 greener solvent alternatives
- Before/after comparison
- XAI decision trace
- Solvent flashcards for visual comparison of hazard and sustainability properties
- Human validation feedback
- Expert-memory retrieval that changes future deterministic rankings
- Downloadable scientific report

## Impact

GreenChem AI helps teams prevent hazardous solvent choices, reduce process waste, and build a feedback loop between scientific scoring and chemist judgment.

## Next Steps

- Add richer toxicology datasets
- Add reaction-specific solvent outcome datasets
- Add uncertainty scoring
- Add PDF ingestion for site-specific solvent guides
- Add SQLite feedback store and per-user preferences
