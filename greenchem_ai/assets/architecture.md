# Architecture

GreenChem AI is an Explainable Green Chemistry Decision Support System.

It combines deterministic scientific scoring, solvent similarity analysis, scientific literature retrieval, expert-memory learning, and human validation to help chemists identify safer and more sustainable solvent alternatives before experimentation.

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
  +--> science_backend_bridge.py adds first-part HSP, density E-Factor, and structural-alert evidence
  +--> xai_engine.py builds decision trace and limitations
  |
  v
Top 5 solvent recommendations
  |
  +--> rag_engine.py retrieves green chemistry context from local ChromaDB
  +--> rag_engine.py retrieves expert feedback memory from local validation history
  +--> science_backend_bridge.py contributes supporting evidence to XAI and reports
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

## First-Part Bridge

The first application layer is integrated as supporting scientific evidence, not as a duplicate UI. Its enriched solvent data is stored in `data/solvents_backend.csv`, and `core/science_backend_bridge.py` exposes:

- Hansen solubility parameter distance between the current and recommended solvents
- Density-based backend E-Factor estimate
- Solvent family and density metadata
- Simple structural alerts

This bridge feeds the XAI panel, Scientific Sources panel, Llama context, and PDF report. It does not choose or override the recommended solvent.
