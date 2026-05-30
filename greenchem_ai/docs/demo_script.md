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
2. Keep the default demo values:
   - Reaction type: Esterification
   - Current solvent: DMF
   - Yield: 65
   - Waste: 40
3. Click **Analyze process**.
4. Show the current GreenScore, toxicity score, E-Factor, and atom economy.
5. Point out the top 5 alternatives table.
6. Open the score contribution tables and explain that every contribution is transparent.
7. Read the XAI trace:
   - Compatibility filter
   - Toxicity change
   - GreenScore change
   - E-Factor change
   - Feedback adjustment
8. Show the RAG sources panel.
9. Explain that Llama only summarizes already calculated values.
10. Click **Accept**, **Reject**, or **Request Alternative** to store human validation.
11. Open **Feedback History** from the sidebar.
12. Download the PDF scientific report.

## Judge-Friendly One-Liner

GreenChem AI turns solvent substitution into a transparent, local, human-validated scientific scoring workflow where the LLM explains the result but never makes the decision.
