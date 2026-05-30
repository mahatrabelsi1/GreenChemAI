# Demo Script

## Setup

Run:

```bash
cd greenchem_ai
pip install -r requirements.txt
streamlit run app.py
```

Optional:

```bash
ollama pull llama3
ollama serve
```

## Walkthrough

1. Open the Streamlit app.
2. Say: "This is not a chatbot. The LLM does not choose the solvent. The deterministic science engine chooses, and the LLM explains."
3. Keep the default demo values:
   - Reaction type: Esterification
   - Current solvent: DMF
   - Yield: 65
   - Waste: 40
4. Click **Analyze process**.
5. Show the current GreenScore, toxicity score, E-Factor, and atom economy.
6. Point out the top 5 alternatives table.
7. Open the score contribution tables and explain that every contribution is transparent.
8. Read the XAI trace:
   - Compatibility filter
   - Toxicity change
   - GreenScore change
   - E-Factor change
   - Feedback adjustment
9. Show the RAG sources panel.
10. Explain that Llama only summarizes already calculated values.
11. Click **Accept**, **Reject**, or **Request Alternative** to store human validation.
12. Open **Feedback History** from the sidebar.
13. Download the PDF scientific report.

## Judge-Friendly One-Liner

GreenChem AI combines deterministic scientific scoring, solvent similarity analysis, scientific literature retrieval, expert-memory learning, and human validation to help chemists identify safer and more sustainable solvent alternatives before experimentation.
