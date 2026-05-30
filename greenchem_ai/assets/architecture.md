# Architecture

```text
User input
  |
  v
Streamlit dashboard
  |
  v
Local deterministic science pipeline
  |
  +--> solvent_database.py loads solvent properties
  +--> reaction_database.py loads reaction assumptions
  +--> toxicity_model.py extracts RDKit features and trains/loads Random Forest
  +--> scoring_engine.py calculates E-Factor, atom economy, GreenScore contributions
  +--> optimization_engine.py ranks compatible alternatives
  +--> xai_engine.py builds decision trace and limitations
  |
  v
Top 5 solvent recommendations
  |
  +--> rag_engine.py retrieves green chemistry context from local ChromaDB
  +--> rag_engine.py retrieves expert feedback memory from local validation history
  +--> llama_explainer.py asks local Ollama for explanation only
  |
  v
Human validation
  |
  v
feedback_engine.py stores Accept / Reject / Request Alternative in data/feedback.csv
  |
  v
Future deterministic rankings receive expert-memory adjustments
```

## AI Boundary

The LLM never chooses a solvent. It receives:

- Current calculated metrics
- Recommended calculated metrics
- Deterministic decision trace
- Retrieved reference chunks

It returns concise explanatory prose. If Ollama is unavailable, the app uses a deterministic fallback explanation.

## Adaptive Memory

Human validation is not just stored. It is converted into deterministic ranking adjustments:

- Accept gives the recommended solvent a future bonus for similar reaction contexts.
- Reject gives the recommended solvent a future penalty for similar reaction contexts.
- Request Alternative gives a smaller penalty.

The expert-memory RAG panel shows which prior decisions were retrieved for the current process.
