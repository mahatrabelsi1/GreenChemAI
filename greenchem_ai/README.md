# GreenChem AI

AI-powered green solvent substitution and process optimization platform for a local AI x Green Chemistry hackathon MVP.

The LLM does not decide, rank, calculate, or recommend. The recommendation is produced by deterministic scientific scoring and local ML support. Ollama only explains values that have already been calculated.

## Features

- Toxic solvent substitution for Axis 2.1
- Chemical process optimization for Axis 2.3
- RDKit molecular feature extraction with fallback parsing
- Random Forest QSAR-style toxicity estimator trained from the local solvent table
- Transparent 0-100 GreenScore with weighted contributions
- E-Factor and atom economy approximations
- ChromaDB RAG with built-in green chemistry chunks
- Expert-memory RAG from accepted/rejected human validation
- Ollama/Llama explanation with fallback if Ollama is offline
- Human validation feedback stored locally in CSV
- Downloadable PDF scientific report
- Local solvent flashcards inspired by the AI4Green Solvent Guide
- AI Process Assistant that routes a natural-language chemistry problem into the right workflow
- Browser-local assistant voice for routed recommendations with a mute toggle

## Install

Recommended Python version: 3.10, 3.11, or 3.12. RDKit and some Streamlit dependencies may not publish wheels for very new Python versions.

```bash
cd greenchem_ai
pip install -r requirements.txt
streamlit run app.py
```

Optional local model:

```bash
ollama pull llama3
ollama serve
```

You can also use `llama3.1` or `llama3.2` by changing the model name in the app.

## Demo Input

- Reaction type: Esterification
- Current solvent: DMF
- Yield: 65
- Waste: 40 kg

You can also start from the **AI Assistant** page and type:

```text
I am running an esterification in DMF. Yield is 65% and waste is 40 kg. I want a safer solvent and lower waste.
```

The assistant extracts the reaction, solvent, yield, and waste, recommends the Process Analysis workflow, and pre-fills the analysis form.

## Assistant Voice

The chemist enters text only. After the AI Process Assistant routes the problem, the app attempts to read the recommended workflow aloud automatically using the browser's built-in `speechSynthesis` API. It does not call a hosted Hugging Face endpoint or any external TTS API. Use the sidebar **Mute assistant voice** toggle to silence narration.

Future local-TTS upgrade path: download and run an open-source TTS model locally, then connect it through your own backend. Calling a hosted Hugging Face Endpoint would count as an external API and is not used in this MVP.

## How Scoring Works

GreenScore is calculated from transparent weighted components:

- Solvent green score: 22%
- Low toxicity: 18%
- Biodegradability: 12%
- Low regulatory risk: 12%
- Yield: 14%
- Low E-Factor: 12%
- Atom economy: 10%

E-Factor is approximated as:

```text
waste kg / estimated product kg
```

The optimized process uses deterministic assumptions for yield and waste based on solvent greenness, regulatory risk, and reaction class waste factor. These assumptions are visible in the XAI trace and should be validated experimentally.

## RAG and Feedback Workflow

GreenChem AI uses two separate local retrieval paths:

- Scientific RAG retrieves solvent substitution, E-Factor, atom economy, CHEM21/GSK-style guidance, and green chemistry context for the Llama explanation.
- Expert-memory RAG retrieves prior human validation from `data/feedback.csv` and makes it visible during later predictions.

Accepted recommendations add a positive deterministic ranking adjustment. Rejected or alternative-requested recommendations add a negative adjustment. This means expert decisions influence future rankings while the LLM still never decides the solvent.

## Project Structure

```text
greenchem_ai/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── solvents.csv
│   ├── reactions.csv
│   └── feedback.csv
├── core/
│   ├── solvent_database.py
│   ├── reaction_database.py
│   ├── scoring_engine.py
│   ├── optimization_engine.py
│   ├── toxicity_model.py
│   ├── xai_engine.py
│   └── feedback_engine.py
├── ai/
│   ├── llama_explainer.py
│   ├── rag_engine.py
│   └── prompts.py
├── assets/
│   └── architecture.md
└── docs/
    ├── scientific_note.md
    ├── pitch_outline.md
    └── demo_script.md
```

## Local-Only Design

No external APIs are required. ChromaDB persists locally under `data/chroma_store`, model artifacts are saved under `data/toxicity_model.joblib`, and user validation is stored in `data/feedback.csv`.

## Optional AI4Green Solvent Guide

The original AI4Green solvent flashcards project is available at:

https://github.com/AI4Green/solvent_flashcards

It provides a standalone visual interface for comparing common laboratory solvents using CHEM21 solvent guide data. Its package can be installed separately:

```bash
pip install solvent-guide
python -m solvent_guide.webapp
```

This MVP includes its own Streamlit flashcard page using the bundled `data/solvents.csv` table rather than copying AI4Green code. The AI4Green project is AGPL-3.0 licensed, so bundling or modifying its code directly may affect the licensing of redistributed combined software.
