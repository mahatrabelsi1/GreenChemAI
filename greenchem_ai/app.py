from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ai.llama_explainer import explain_with_ollama
from ai.rag_engine import retrieve_expert_memory_context, retrieve_scientific_context
from core.feedback_engine import load_feedback, store_feedback
from core.optimization_engine import analyze_current_process, recommend_solvents
from core.reaction_database import get_reaction, load_reactions, reaction_types
from core.solvent_database import get_solvent, load_solvents, solvent_names
from core.xai_engine import recommendation_xai


st.set_page_config(
    page_title="GreenChem AI",
    page_icon="GC",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    :root {
        --gc-primary: #0F766E;
        --gc-secondary: #14B8A6;
        --gc-accent: #84CC16;
        --gc-bg: #F1F5F9;
        --gc-card: #FFFFFF;
        --gc-text: #0F172A;
        --gc-muted: #475569;
        --gc-success: #22C55E;
        --gc-warning: #F59E0B;
        --gc-danger: #EF4444;
        --gc-border: #CBD5E1;
    }
    .stApp {
        background: var(--gc-bg);
        color: var(--gc-text);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    section[data-testid="stSidebar"] {
        background: #E2E8F0;
        border-right: 1px solid var(--gc-border);
    }
    h1, h2, h3, h4, h5, h6,
    label,
    .stMarkdown,
    .stText,
    .stCaption,
    p {
        color: var(--gc-text);
    }
    .stCaption,
    .small-muted,
    div[data-testid="stCaptionContainer"] {
        color: var(--gc-muted);
    }
    div[data-testid="stMetric"] {
        background: var(--gc-card);
        border: 1px solid var(--gc-border);
        border-radius: 8px;
        padding: 14px;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricDelta"] {
        color: var(--gc-muted);
    }
    div[data-testid="stMetricValue"] {
        color: var(--gc-text);
        font-size: 1.7rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--gc-border);
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-baseweb="select"] > div,
    textarea,
    input {
        background-color: #FFFFFF !important;
        color: var(--gc-text) !important;
        border-color: var(--gc-border) !important;
    }
    button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stDownloadButton"] button {
        background: var(--gc-primary);
        border: 1px solid var(--gc-secondary);
        color: #F8FAFC;
    }
    button:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover {
        border-color: var(--gc-accent);
        color: #F8FAFC;
    }
    .gc-panel {
        background: var(--gc-card);
        border: 1px solid var(--gc-border);
        border-radius: 8px;
        padding: 14px;
    }
    .gc-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #0F172A;
    }
    .gc-hero-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        min-height: 148px;
    }
    .gc-hero-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 8px 0;
    }
    .gc-hero-value {
        color: #0F172A;
        font-size: 34px;
        font-weight: 850;
        line-height: 1.05;
        margin: 0;
    }
    .gc-hero-sub {
        color: #475569;
        font-size: 14px;
        margin: 10px 0 0 0;
    }
    .gc-factor {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
        color: #0F172A;
        font-weight: 750;
        text-align: center;
    }
    .gc-check {
        color: #22C55E;
        font-size: 18px;
        font-weight: 900;
    }
    .gc-method-card {
        background: #F0FDFA;
        border: 1px solid #99F6E4;
        border-radius: 8px;
        padding: 14px;
        color: #134E4A;
        margin: 8px 0 18px 0;
    }
    .gc-method-card b {
        color: #0F766E;
    }
    .gc-flow {
        display: grid;
        grid-template-columns: 1fr 80px 1fr;
        gap: 14px;
        align-items: stretch;
        margin: 14px 0;
    }
    .gc-flow-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }
    .gc-flow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0F766E;
        font-size: 30px;
        font-weight: 900;
    }
    .gc-compare-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(150px, 1fr));
        gap: 12px;
        margin: 14px 0 16px 0;
    }
    .gc-compare-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    .gc-compare-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 8px 0;
    }
    .gc-compare-value {
        color: #0F172A;
        font-size: 24px;
        font-weight: 850;
        margin: 0;
    }
    .gc-compare-note {
        color: #475569;
        font-size: 13px;
        margin: 8px 0 0 0;
    }
    .gc-insight {
        background: #F0FDFA;
        border: 1px solid #99F6E4;
        border-radius: 8px;
        padding: 14px;
        color: #134E4A;
        margin: 10px 0 16px 0;
    }
    .gc-rec-hero {
        background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
        border-radius: 8px;
        padding: 22px;
        color: #F8FAFC;
        box-shadow: 0 16px 34px rgba(15, 118, 110, 0.22);
        margin: 12px 0 16px 0;
    }
    .gc-rec-hero p,
    .gc-rec-hero h2 {
        color: #F8FAFC;
        margin: 0;
    }
    .gc-rec-hero .muted {
        color: #CCFBF1;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .gc-rec-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(150px, 1fr));
        gap: 12px;
        margin: 12px 0 16px 0;
    }
    .gc-rec-tile {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        min-height: 128px;
    }
    .gc-rec-icon {
        width: 30px;
        height: 30px;
        border-radius: 999px;
        background: #D1FAE5;
        color: #0F766E;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        margin-bottom: 8px;
    }
    .gc-rec-title {
        color: #0F172A;
        font-weight: 850;
        margin: 0 0 6px 0;
    }
    .gc-rec-text {
        color: #475569;
        font-size: 13px;
        margin: 0;
    }
    .gc-memory-box {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 8px;
        padding: 14px;
        color: #78350F;
        margin: 8px 0 16px 0;
    }
    .gc-decision-card {
        background: #FFFFFF;
        border: 1px solid #99F6E4;
        border-left: 6px solid #0F766E;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 12px 30px rgba(15, 118, 110, 0.10);
        margin: 12px 0 16px 0;
    }
    .gc-decision-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(150px, 1fr));
        gap: 10px;
        margin-top: 12px;
    }
    .gc-decision-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px;
    }
    .gc-confidence-grid {
        display: grid;
        grid-template-columns: 1.1fr 2fr;
        gap: 12px;
        margin: 12px 0 16px 0;
    }
    .gc-confidence-score {
        background: linear-gradient(135deg, #0F766E 0%, #22C55E 100%);
        color: #F8FAFC;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 12px 30px rgba(15, 118, 110, 0.18);
    }
    .gc-confidence-score p,
    .gc-confidence-score h2 {
        color: #F8FAFC;
        margin: 0;
    }
    .gc-confidence-breakdown {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
    }
    .gc-risk-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 12px;
        color: #0F172A;
        font-size: 13px;
        font-weight: 900;
    }
    .gc-why-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 13px;
        margin-bottom: 10px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }
    .gc-alt-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(150px, 1fr));
        gap: 10px;
        margin: 14px 0 16px 0;
    }
    .gc-alt-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        min-height: 188px;
    }
    .gc-alt-rank {
        display: inline-block;
        background: #0F766E;
        color: #F8FAFC;
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 12px;
        font-weight: 850;
        margin-bottom: 8px;
    }
    .gc-mini-bar {
        height: 7px;
        background: #E2E8F0;
        border-radius: 999px;
        overflow: hidden;
        margin: 7px 0 10px 0;
    }
    .gc-mini-fill {
        height: 7px;
        border-radius: 999px;
    }
    .gc-source-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }
    .gc-component-panel {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }
    .gc-component-row {
        margin-bottom: 10px;
    }
    .gc-component-head {
        display: flex;
        justify-content: space-between;
        color: #0F172A;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .gc-component-bar {
        height: 9px;
        background: #E2E8F0;
        border-radius: 999px;
        overflow: hidden;
    }
    .gc-component-fill {
        height: 9px;
        background: linear-gradient(90deg, #0F766E, #22C55E);
        border-radius: 999px;
    }
    .gc-assistant-hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #ECFEFF 100%);
        border: 1px solid #99F6E4;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 12px 30px rgba(15, 118, 110, 0.10);
        margin-bottom: 16px;
    }
    .gc-route-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        margin: 10px 0;
    }
    .gc-route-badge {
        display: inline-block;
        background: #D1FAE5;
        color: #0F766E;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 900;
        margin-bottom: 8px;
    }
    .gc-extract-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(140px, 1fr));
        gap: 10px;
        margin: 12px 0;
    }
    .gc-extract-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px;
    }
</style>
"""


@st.cache_data
def cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_solvents(), load_reactions()


def metric_row(current: dict, best: dict | None = None) -> None:
    cols = st.columns(4)
    cols[0].metric("GreenScore", f"{current['green_score']:.1f}", None if not best else f"{best['green_score'] - current['green_score']:.1f}")
    cols[1].metric("Toxicity", f"{current['toxicity_score']:.1f}", None if not best else f"{best['toxicity_score'] - current['toxicity_score']:.1f}", delta_color="inverse")
    cols[2].metric("E-Factor", f"{current['e_factor']:.3f}", None if not best else f"{best['e_factor'] - current['e_factor']:.3f}", delta_color="inverse")
    cols[3].metric("Atom Economy", f"{current['atom_economy']:.1f}%")


def assistant_voice_player(script: str, key: str) -> None:
    """Local browser text-to-speech output. No external AI API is called."""
    if st.session_state.get("voice_muted", False):
        st.caption("Assistant voice is muted.")
        return
    payload = json.dumps(script)
    components.html(
        f"""
        <button id="play-{key}" style="
            width:100%;
            border:1px solid #0F766E;
            background:#0F766E;
            color:#F8FAFC;
            border-radius:8px;
            padding:10px 14px;
            font-weight:800;
            cursor:pointer;
            font-family:Arial, sans-serif;">
            Play assistant voice
        </button>
        <div style="font-family:Arial, sans-serif; color:#475569; font-size:12px; margin-top:6px;">
            Auto voice starts when supported by the browser. Use the button if your browser blocks autoplay.
        </div>
        <script>
        const text = {payload};
        const button = document.getElementById("play-{key}");
        const storageKey = "greenchem_voice_auto_{key}";
        const speak = () => {{
            if (!("speechSynthesis" in window)) return;
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 2.0;
            utterance.pitch = 1.12;
            utterance.volume = 1.0;
            const pickVoice = () => {{
                const voices = window.speechSynthesis.getVoices();
                const femaleVoice =
                    voices.find(v => /female|woman|zira|samantha|victoria|karen|serena|susan|hazel|aria|jenny|natural/i.test(v.name) && /english|en-/i.test(v.lang)) ||
                    voices.find(v => /female|woman|zira|samantha|victoria|karen|serena|susan|hazel|aria|jenny|natural/i.test(v.name)) ||
                    voices.find(v => /english|en-/i.test(v.lang));
                if (femaleVoice) utterance.voice = femaleVoice;
                window.speechSynthesis.speak(utterance);
            }};
            if (window.speechSynthesis.getVoices().length) {{
                pickVoice();
            }} else {{
                window.speechSynthesis.onvoiceschanged = pickVoice;
            }}
        }};
        button.addEventListener("click", speak);
        setTimeout(() => {{
            if (sessionStorage.getItem(storageKey) !== text) {{
                sessionStorage.setItem(storageKey, text);
                speak();
            }}
        }}, 300);
        </script>
        """,
        height=74,
    )


def contributions_table(contributions: dict[str, float]) -> pd.DataFrame:
    labels = {
        "solvent_green": "Solvent green score",
        "toxicity": "Low toxicity",
        "biodegradability": "Biodegradability",
        "regulatory": "Low regulatory risk",
        "yield": "Yield",
        "efactor": "Low E-Factor",
        "atom_economy": "Atom economy",
    }
    return pd.DataFrame(
        [{"Factor": labels[key], "Weighted contribution": value} for key, value in contributions.items()]
    )


def score_color(score: float) -> str:
    if score < 40:
        return "#EF4444"
    if score < 70:
        return "#F59E0B"
    return "#22C55E"


def hero_card(label: str, value: str, subtext: str, color: str = "#0F172A") -> str:
    return f"""
    <div class="gc-hero-card">
        <p class="gc-hero-label">{label}</p>
        <p class="gc-hero-value" style="color:{color};">{value}</p>
        <p class="gc-hero-sub">{subtext}</p>
    </div>
    """


def pipeline_diagram() -> str:
    return """
    <div class="gc-method-card">
        <b>Method:</b> the app filters compatible solvents, calculates deterministic GreenScore metrics,
        retrieves scientific and expert-memory context, then asks the chemist to validate the recommendation.
        Llama explains the result only after scoring is complete.
    </div>
    """


def process_flow(current: dict, best: dict) -> str:
    return f"""
    <div class="gc-flow">
        <div class="gc-flow-card">
            <p class="gc-hero-label">Current Process</p>
            <h2 style="color:#EF4444; margin:0;">{current['solvent']}</h2>
            <p style="color:#0F172A; font-size:28px; font-weight:850; margin:8px 0;">GreenScore {current['green_score']:.1f}</p>
            <p style="color:#475569; margin:0;">Toxicity {current['toxicity_score']:.1f} | E-Factor {current['e_factor']:.3f}</p>
        </div>
        <div class="gc-flow-arrow">-&gt;</div>
        <div class="gc-flow-card">
            <p class="gc-hero-label">Recommended Process</p>
            <h2 style="color:#22C55E; margin:0;">{best['solvent']}</h2>
            <p style="color:#0F172A; font-size:28px; font-weight:850; margin:8px 0;">GreenScore {best['green_score']:.1f}</p>
            <p style="color:#475569; margin:0;">Toxicity {best['toxicity_score']:.1f} | E-Factor {best['e_factor']:.3f}</p>
        </div>
    </div>
    """


def before_after_table(current: dict, best: dict) -> pd.DataFrame:
    rows = [
        ("GreenScore", current["green_score"], best["green_score"], "higher"),
        ("Toxicity", current["toxicity_score"], best["toxicity_score"], "lower"),
        ("E-Factor", current["e_factor"], best["e_factor"], "lower"),
        ("Yield", current["yield_percent"], best["yield_percent"], "higher"),
        ("Waste kg", current["waste_kg"], best["waste_kg"], "lower"),
        ("Atom Economy", current["atom_economy"], best["atom_economy"], "higher"),
    ]
    table = []
    for metric, current_value, optimized_value, preferred in rows:
        current_float = float(current_value)
        optimized_float = float(optimized_value)
        change = round(optimized_float - current_float, 3)
        percent = 0.0 if current_float == 0 else round((change / abs(current_float)) * 100.0, 1)
        improved = change > 0 if preferred == "higher" else change < 0
        if change == 0:
            status = "No change"
        else:
            status = "Improved" if improved else "Worse"
        direction = "up is better" if preferred == "higher" else "down is better"
        table.append(
            {
                "Metric": metric,
                "Current": current_value,
                "Optimized": optimized_value,
                "Change": change,
                "% Change": f"{percent:+.1f}%",
                "Interpretation": f"{status} ({direction})",
            }
        )
    return pd.DataFrame(table)


def before_after_insights(current: dict, best: dict) -> str:
    green_delta = best["green_score"] - current["green_score"]
    toxicity_delta = current["toxicity_score"] - best["toxicity_score"]
    efactor_delta = current["e_factor"] - best["e_factor"]
    waste_delta = current["waste_kg"] - best["waste_kg"]
    return f"""
    <div class="gc-compare-grid">
        <div class="gc-compare-card">
            <p class="gc-compare-label">GreenScore Gain</p>
            <p class="gc-compare-value" style="color:{score_color(best['green_score'])};">+{green_delta:.1f}</p>
            <p class="gc-compare-note">Higher means greener overall process performance.</p>
        </div>
        <div class="gc-compare-card">
            <p class="gc-compare-label">Toxicity Reduction</p>
            <p class="gc-compare-value" style="color:#22C55E;">-{toxicity_delta:.1f}</p>
            <p class="gc-compare-note">Lower toxicity score means safer solvent profile.</p>
        </div>
        <div class="gc-compare-card">
            <p class="gc-compare-label">E-Factor Reduction</p>
            <p class="gc-compare-value" style="color:#22C55E;">-{efactor_delta:.3f}</p>
            <p class="gc-compare-note">Lower E-Factor means less waste per product mass.</p>
        </div>
        <div class="gc-compare-card">
            <p class="gc-compare-label">Waste Reduction</p>
            <p class="gc-compare-value" style="color:#22C55E;">-{waste_delta:.1f} kg</p>
            <p class="gc-compare-note">Estimated process waste reduction from substitution.</p>
        </div>
    </div>
    <div class="gc-insight">
        <b>How to read this step:</b> good substitutions increase GreenScore and yield, while decreasing toxicity, E-Factor, regulatory burden, and waste. These values are calculated before the LLM explanation is generated.
    </div>
    """


def recommendation_story(current: dict, best: dict, improvement: float, toxicity_reduction: float, toxicity_pct: float) -> str:
    waste_reduction = current["waste_kg"] - best["waste_kg"]
    memory = best.get("feedback_adjustment", 0.0)
    memory_text = (
        f"Expert memory contributed {memory:+.1f} ranking points from prior validation."
        if memory
        else "No matching expert-memory adjustment was applied yet."
    )
    return f"""
    <div class="gc-rec-hero">
        <p class="muted">Selected recommendation</p>
        <h2 style="font-size:36px; font-weight:900;">{best['solvent']}</h2>
        <p style="margin-top:8px; font-size:16px;">Chosen by deterministic scoring for {current['reaction_type']} solvent substitution.</p>
    </div>
    <div class="gc-rec-grid">
        <div class="gc-rec-tile">
            <div class="gc-rec-icon">1</div>
            <p class="gc-rec-title">Toxicity Drops</p>
            <p class="gc-rec-text">{current['toxicity_score']:.1f} -> {best['toxicity_score']:.1f}<br><b>{toxicity_reduction:.1f} points lower</b> ({toxicity_pct:.1f}%).</p>
        </div>
        <div class="gc-rec-tile">
            <div class="gc-rec-icon">2</div>
            <p class="gc-rec-title">GreenScore Rises</p>
            <p class="gc-rec-text">{current['green_score']:.1f} -> {best['green_score']:.1f}<br><b>{improvement:+.1f} points</b> better.</p>
        </div>
        <div class="gc-rec-tile">
            <div class="gc-rec-icon">3</div>
            <p class="gc-rec-title">Reaction Compatible</p>
            <p class="gc-rec-text">The solvent passed the {current['reaction_type']} compatibility filter before ranking.</p>
        </div>
        <div class="gc-rec-tile">
            <div class="gc-rec-icon">4</div>
            <p class="gc-rec-title">Less Waste</p>
            <p class="gc-rec-text">{current['waste_kg']:.1f} kg -> {best['waste_kg']:.1f} kg<br><b>{waste_reduction:.1f} kg reduction</b>.</p>
        </div>
    </div>
    <div class="gc-memory-box">
        <b>Adaptive expert memory:</b> {memory_text}
    </div>
    """


def risk_level(candidate: dict) -> dict:
    risk_points = 0
    if candidate["toxicity_score"] >= 60:
        risk_points += 2
    elif candidate["toxicity_score"] >= 35:
        risk_points += 1
    if candidate.get("gsk_score", 3) >= 7:
        risk_points += 2
    elif candidate.get("gsk_score", 3) >= 5:
        risk_points += 1
    if candidate["chem21_classification"] in {"Hazardous", "Problematic"}:
        risk_points += 2
    elif candidate["chem21_classification"] == "Recommended":
        risk_points += 1
    if candidate.get("feedback_adjustment", 0) < 0:
        risk_points += 1

    if risk_points <= 1:
        return {"label": "LOW", "color": "#22C55E", "summary": "Low hazard and strong sustainability profile."}
    if risk_points <= 3:
        return {"label": "MEDIUM", "color": "#F59E0B", "summary": "Manageable risk; validate performance and workup carefully."}
    return {"label": "HIGH", "color": "#EF4444", "summary": "High hazard or uncertainty; experimental validation is critical."}


def recommendation_confidence(current: dict, best: dict, recommendations: list[dict]) -> dict:
    rank_gap = 0.0
    if len(recommendations) > 1:
        rank_gap = max(0.0, recommendations[0]["rank_score"] - recommendations[1]["rank_score"])
    compatibility = 95.0
    data_coverage = 92.0
    historical = 70.0 + min(20.0, max(-20.0, best.get("feedback_adjustment", 0.0) * 4.0))
    metric_strength = min(95.0, 65.0 + max(0.0, best["green_score"] - current["green_score"]) * 0.45 + rank_gap * 0.25)
    confidence = round((compatibility * 0.25 + data_coverage * 0.2 + historical * 0.2 + metric_strength * 0.35), 1)
    return {
        "score": confidence,
        "compatibility": "High",
        "historical": "High" if historical >= 78 else "Medium" if historical >= 60 else "Low",
        "data_coverage": "High",
        "rank_separation": "High" if rank_gap >= 5 else "Medium" if rank_gap >= 2 else "Low",
    }


def decision_card(current: dict, best: dict, recommendations: list[dict]) -> str:
    confidence = recommendation_confidence(current, best, recommendations)
    risk = risk_level(best)
    tox_delta = current["toxicity_score"] - best["toxicity_score"]
    waste_delta = current["waste_kg"] - best["waste_kg"]
    score_delta = best["green_score"] - current["green_score"]
    return f"""
    <div class="gc-decision-card">
        <p style="color:#0F766E; font-size:12px; font-weight:900; text-transform:uppercase; letter-spacing:0.08em; margin:0 0 8px 0;">Decision card</p>
        <h2 style="color:#0F172A; margin:0;">Replace {current['solvent']} with {best['solvent']}</h2>
        <p style="color:#475569; margin:8px 0 0 0;">Primary reason: {tox_delta:.1f} toxicity-point reduction, {waste_delta:.1f} kg estimated waste reduction, and GreenScore {score_delta:+.1f}.</p>
        <div class="gc-decision-grid">
            <div class="gc-decision-item"><b>Confidence</b><br><span style="color:#0F766E; font-size:22px; font-weight:900;">{confidence['score']:.0f}%</span></div>
            <div class="gc-decision-item"><b>Experimental Risk</b><br><span class="gc-risk-badge" style="background:{risk['color']};">{risk['label']}</span></div>
            <div class="gc-decision-item"><b>Current Solvent</b><br>{current['solvent']}</div>
            <div class="gc-decision-item"><b>Recommended</b><br>{best['solvent']}</div>
        </div>
    </div>
    """


def confidence_panel(current: dict, best: dict, recommendations: list[dict]) -> str:
    confidence = recommendation_confidence(current, best, recommendations)
    return f"""
    <div class="gc-confidence-grid">
        <div class="gc-confidence-score">
            <p style="font-size:12px; font-weight:900; text-transform:uppercase; letter-spacing:0.08em;">Recommendation Confidence</p>
            <h2 style="font-size:42px; font-weight:950;">{confidence['score']:.0f}%</h2>
        </div>
        <div class="gc-confidence-breakdown">
            <p style="color:#0F172A; font-weight:850; margin:0 0 8px 0;">Confidence basis</p>
            <p style="color:#475569; margin:4px 0;"><b>Compatibility:</b> {confidence['compatibility']}</p>
            <p style="color:#475569; margin:4px 0;"><b>Historical acceptance:</b> {confidence['historical']}</p>
            <p style="color:#475569; margin:4px 0;"><b>Data coverage:</b> {confidence['data_coverage']}</p>
            <p style="color:#475569; margin:4px 0;"><b>Rank separation:</b> {confidence['rank_separation']}</p>
        </div>
    </div>
    """


def why_not_alternatives(best: dict, recommendations: list[dict]) -> str:
    cards = []
    for rec in recommendations[1:5]:
        reasons = []
        if rec["green_score"] < best["green_score"]:
            reasons.append(f"GreenScore is {best['green_score'] - rec['green_score']:.1f} points lower.")
        if rec["toxicity_score"] > best["toxicity_score"]:
            reasons.append(f"Toxicity score is {rec['toxicity_score'] - best['toxicity_score']:.1f} points higher.")
        if rec["e_factor"] > best["e_factor"]:
            reasons.append(f"E-Factor is {rec['e_factor'] - best['e_factor']:.3f} higher.")
        if rec.get("feedback_adjustment", 0) < best.get("feedback_adjustment", 0):
            reasons.append("Expert-memory support is weaker.")
        if not reasons:
            reasons.append("It ranked lower after all weighted score components were combined.")
        cards.append(
            f"""
            <div class="gc-why-card">
                <p style="color:#0F172A; font-weight:850; margin:0 0 6px 0;">Why not {rec['solvent']}?</p>
                <p style="color:#475569; margin:0;">{reasons[0]}</p>
            </div>
            """
        )
    return "".join(cards)


def expert_memory_summary(recommendations: list[dict]) -> str:
    rows = []
    for rec in recommendations:
        memories = rec.get("expert_memory_matches", [])
        accepted = sum(1 for item in memories if item["decision"] == "Accept")
        rejected = sum(1 for item in memories if item["decision"] == "Reject")
        alternative = sum(1 for item in memories if item["decision"] == "Request Alternative")
        if not memories and rec.get("feedback_adjustment", 0) == 0:
            continue
        rows.append(
            f"""
            <div class="gc-source-card" style="background:#F0FDFA; border-color:#99F6E4;">
                <p style="color:#0F766E; font-weight:900; margin:0 0 6px 0;">Similar historical case: {rec['reaction_type']} + {rec['solvent']}</p>
                <p style="color:#475569; margin:0;">Accepted: {accepted} | Rejected: {rejected} | Alternatives requested: {alternative} | Ranking memory: {rec.get('feedback_adjustment', 0):+.1f}</p>
            </div>
            """
        )
    if not rows:
        return '<div class="gc-source-card"><p style="color:#475569; margin:0;">No similar historical cases yet. Expert decisions saved today will appear here in future predictions.</p></div>'
    return "".join(rows)


def alternatives_cards(recommendations: list[dict]) -> str:
    cards = []
    for idx, rec in enumerate(recommendations, start=1):
        fill = score_color(rec["green_score"])
        cards.append(
            f"""
            <div class="gc-alt-card">
                <span class="gc-alt-rank">#{idx}</span>
                <h3 style="color:#0F172A; margin:0 0 6px 0; font-size:18px;">{rec['solvent']}</h3>
                <p style="color:#64748B; margin:0 0 8px 0; font-size:12px; font-weight:800;">{rec['chem21_classification']}</p>
                <p style="color:#0F172A; margin:0;"><b>GreenScore</b> {rec['green_score']:.1f}</p>
                <div class="gc-mini-bar"><div class="gc-mini-fill" style="width:{rec['green_score']}%; background:{fill};"></div></div>
                <p style="color:#475569; margin:0; font-size:13px;">Toxicity {rec['toxicity_score']:.1f}</p>
                <p style="color:#475569; margin:2px 0 0 0; font-size:13px;">E-Factor {rec['e_factor']:.3f}</p>
                <p style="color:#475569; margin:2px 0 0 0; font-size:13px;">Memory {rec.get('feedback_adjustment', 0):+.1f}</p>
            </div>
            """
        )
    return f'<div class="gc-alt-grid">{"".join(cards)}</div>'


def recommendations_table(recommendations: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(recommendations)[
        [
            "solvent",
            "rank_score",
            "green_score",
            "toxicity_score",
            "e_factor",
            "yield_percent",
            "waste_kg",
            "chem21_classification",
            "feedback_adjustment",
        ]
    ].copy()
    return df.rename(
        columns={
            "solvent": "Solvent",
            "rank_score": "Rank Score",
            "green_score": "GreenScore",
            "toxicity_score": "Toxicity",
            "e_factor": "E-Factor",
            "yield_percent": "Yield %",
            "waste_kg": "Waste kg",
            "chem21_classification": "Class",
            "feedback_adjustment": "Expert Memory",
        }
    )


def contribution_bars(contributions: dict[str, float]) -> str:
    labels = {
        "solvent_green": "Solvent green score",
        "toxicity": "Low toxicity",
        "biodegradability": "Biodegradability",
        "regulatory": "Low regulatory risk",
        "yield": "Yield",
        "efactor": "Low E-Factor",
        "atom_economy": "Atom economy",
    }
    max_value = max(contributions.values()) if contributions else 1
    rows = []
    for key, value in sorted(contributions.items(), key=lambda item: item[1], reverse=True):
        width = max(4, (value / max_value) * 100)
        rows.append(
            f"""
            <div class="gc-component-row">
                <div class="gc-component-head"><span>{labels.get(key, key)}</span><span>{value:.2f}</span></div>
                <div class="gc-component-bar"><div class="gc-component-fill" style="width:{width:.1f}%;"></div></div>
            </div>
            """
        )
    return f'<div class="gc-component-panel">{"".join(rows)}</div>'


def route_chemist_problem(problem: str, solvents: pd.DataFrame, reactions: pd.DataFrame) -> dict:
    text = problem.casefold()
    extracted = {
        "reaction_type": None,
        "current_solvent": None,
        "yield_percent": None,
        "waste_kg": None,
        "notes": problem.strip(),
    }

    for reaction in reaction_types(reactions):
        if reaction.casefold() in text:
            extracted["reaction_type"] = reaction
            break

    for solvent in solvent_names(solvents):
        if solvent.casefold() in text:
            extracted["current_solvent"] = solvent
            break

    yield_match = re.search(r"(?:yield|conversion)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", text)
    if yield_match:
        extracted["yield_percent"] = float(yield_match.group(1))

    waste_match = re.search(r"(?:waste|byproduct|effluent)[^\d]*(\d+(?:\.\d+)?)\s*(?:kg|kilogram)", text)
    if waste_match:
        extracted["waste_kg"] = float(waste_match.group(1))

    reasons = []
    feature = "Process Analysis"
    confidence = 0.55

    if any(word in text for word in ["toxic", "toxicity", "hazard", "dmf", "dichloromethane", "replace", "substitute", "solvent"]):
        reasons.append("Solvent hazard or substitution problem detected.")
        feature = "Process Analysis"
        confidence += 0.2
    if any(word in text for word in ["yield", "waste", "e-factor", "efactor", "optimize", "optimise", "process"]):
        reasons.append("Process optimization metrics detected.")
        feature = "Process Analysis"
        confidence += 0.15
    compact_text = re.sub(r"[^a-z0-9]+", "", text)
    if (
        any(word in text for word in ["compare solvents", "solvent library", "flashcard", "flash card", "properties"])
        or "solventflash" in compact_text
        or "solventflsh" in compact_text
        or "flshcard" in compact_text
    ):
        reasons.append("Solvent property comparison requested.")
        feature = "Solvent Flashcards"
        confidence += 0.2
    if any(word in text for word in ["feedback", "accepted", "rejected", "history", "expert decision"]):
        reasons.append("Expert validation history requested.")
        feature = "Feedback History"
        confidence += 0.2
    if any(word in text for word in ["dataset", "data table", "raw data", "reactions csv", "solvents csv"]):
        reasons.append("Raw local data inspection requested.")
        feature = "Local Data"
        confidence += 0.2

    missing = [label for key, label in [
        ("reaction_type", "reaction type"),
        ("current_solvent", "current solvent"),
        ("yield_percent", "yield"),
        ("waste_kg", "waste amount"),
    ] if extracted[key] is None]

    if not reasons:
        reasons.append("No narrow request detected, so the safest starting point is guided process analysis.")

    return {
        "feature": feature,
        "confidence": min(confidence, 0.95),
        "extracted": extracted,
        "missing": missing,
        "reasons": reasons,
    }


def assistant_page(solvents: pd.DataFrame, reactions: pd.DataFrame) -> None:
    st.title("AI Process Assistant")
    st.caption("Describe the chemistry problem in plain language. The assistant routes you to the right GreenChem AI workflow.")

    st.markdown(
        """
        <div class="gc-assistant-hero">
            <p style="color:#0F766E; font-size:12px; font-weight:900; text-transform:uppercase; letter-spacing:0.08em; margin:0 0 8px 0;">Guided intake</p>
            <h2 style="color:#0F172A; margin:0;">Tell me the process problem, not the form fields.</h2>
            <p style="color:#475569; margin:8px 0 0 0;">Example: "I am running an esterification in DMF, yield is 65%, waste is 40 kg, and I want a safer solvent."</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    problem = st.text_area(
        "Chemist problem statement",
        height=150,
        key="assistant_problem",
        placeholder="Describe the reaction, current solvent, yield, waste, safety issue, or optimization goal...",
    )

    if st.button("Route my problem", type="primary"):
        if not problem.strip():
            st.warning("Describe the chemistry problem first.")
            return
        st.session_state["assistant_route"] = route_chemist_problem(problem, solvents, reactions)

    if "assistant_route" not in st.session_state:
        return

    route = st.session_state["assistant_route"]
    extracted = route["extracted"]
    other_workflows = [
        name for name in ["Process Analysis", "Solvent Flashcards", "Feedback History", "Local Data"]
        if name != route["feature"]
    ]
    other_workflow_text = ", ".join(other_workflows)
    assistant_script = (
        f"Hello. I recommend the {route['feature']} workflow for this request. "
        f"You can also use {other_workflow_text} if you want a different view. "
        "Let's get you started."
    )

    st.markdown(
        f"""
        <div class="gc-route-card">
            <span class="gc-route-badge">Recommended workflow</span>
            <h2 style="color:#0F172A; margin:0;">{route['feature']}</h2>
            <p style="color:#475569; margin:8px 0 0 0;">Confidence: {route['confidence'] * 100:.0f}%</p>
            <p style="color:#0F172A; margin:12px 0 0 0;"><b>Assistant says:</b> {assistant_script}</p>
            <div style="margin-top:14px; display:grid; grid-template-columns:repeat(3,minmax(140px,1fr)); gap:10px;">
                <div class="gc-extract-item"><b>Also available</b><br>{other_workflows[0]}</div>
                <div class="gc-extract-item"><b>Also available</b><br>{other_workflows[1]}</div>
                <div class="gc-extract-item"><b>Also available</b><br>{other_workflows[2]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    assistant_voice_player(assistant_script, f"assistant-route-{route['feature'].replace(' ', '-')}")

    if st.button("Open recommended workflow", use_container_width=True):
        defaults = {
            "reaction_type": extracted["reaction_type"] or "Esterification",
            "current_solvent": extracted["current_solvent"] or "DMF",
            "yield_percent": extracted["yield_percent"] or 65.0,
            "waste_kg": extracted["waste_kg"] or 40.0,
            "notes": extracted["notes"],
        }
        if route["feature"] == "Process Analysis":
            st.session_state["assistant_prefill"] = defaults
            st.session_state["analysis_step"] = 1
            st.session_state.pop("analysis", None)
            st.session_state["current_page"] = "Analysis"
        else:
            st.session_state["current_page"] = route["feature"]
        st.rerun()


def build_report(current: dict, best: dict, recommendations: list[dict], xai: dict, explanation: str, sources: list[str]) -> str:
    lines = [
        "# GreenChem AI Scientific Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Current Process",
        f"- Reaction type: {current['reaction_type']}",
        f"- Current solvent: {current['solvent']}",
        f"- GreenScore: {current['green_score']}",
        f"- Toxicity score: {current['toxicity_score']}",
        f"- E-Factor: {current['e_factor']}",
        f"- Atom economy approximation: {current['atom_economy']}%",
        "",
        "## Recommended Substitution",
        f"- Recommended solvent: {best['solvent']}",
        f"- Optimized GreenScore: {best['green_score']}",
        f"- Toxicity score: {best['toxicity_score']}",
        f"- E-Factor: {best['e_factor']}",
        f"- Yield assumption: {best['yield_percent']}%",
        f"- Waste estimate: {best['waste_kg']} kg",
        "",
        "## Top Alternatives",
    ]
    for idx, rec in enumerate(recommendations, start=1):
        lines.append(f"{idx}. {rec['solvent']} - rank score {rec['rank_score']}, GreenScore {rec['green_score']}")
    lines.extend(
        [
            "",
            "## Deterministic Decision Trace",
            *[f"- {item}" for item in xai["decision_trace"]],
            "",
            "## LLM Explanation",
            explanation,
            "",
            "## RAG Sources",
            *[f"- {source}" for source in sources],
            "",
            "## Limitations",
            *[f"- {item}" for item in xai["limitations"]],
        ]
    )
    return "\n".join(lines)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _minimal_pdf_from_text(text: str) -> bytes:
    """Tiny dependency-free PDF fallback for environments without reportlab."""
    safe_lines = []
    for raw_line in text.splitlines():
        line = raw_line.replace("#", "").strip()
        if not line:
            safe_lines.append("")
            continue
        while len(line) > 88:
            safe_lines.append(line[:88])
            line = line[88:]
        safe_lines.append(line)

    pages = []
    for start in range(0, len(safe_lines), 42):
        chunk = safe_lines[start : start + 42]
        content = ["BT", "/F1 10 Tf", "50 780 Td", "14 TL"]
        for line in chunk:
            content.append(f"({_escape_pdf_text(line)}) Tj")
            content.append("T*")
        content.append("ET")
        pages.append("\n".join(content).encode("latin-1", errors="replace"))

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_object_ids = []
    content_object_ids = []
    next_id = 4
    for page_content in pages:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_object_ids.append(page_id)
        content_object_ids.append(content_id)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length " + str(len(page_content)).encode() + b" >>\nstream\n" + page_content + b"\nendstream"
        )

    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode()

    pdf = BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(pdf.tell())
        pdf.write(f"{obj_id} 0 obj\n".encode())
        pdf.write(obj)
        pdf.write(b"\nendobj\n")
    xref_pos = pdf.tell()
    pdf.write(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.write(f"{offset:010d} 00000 n \n".encode())
    pdf.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    )
    return pdf.getvalue()


def build_pdf_report(current: dict, best: dict, recommendations: list[dict], xai: dict, explanation: str, sources: list[str]) -> bytes:
    report_text = build_report(current, best, recommendations, xai, explanation, sources)
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=34, bottomMargin=34)
        styles = getSampleStyleSheet()
        styles["Title"].textColor = colors.HexColor("#0F172A")
        styles["Title"].fontSize = 22
        styles["Heading2"].textColor = colors.HexColor("#0F766E")
        styles["Heading2"].fontSize = 13
        styles["BodyText"].textColor = colors.HexColor("#334155")
        styles["BodyText"].fontSize = 9
        styles["BodyText"].leading = 12

        def p(text: str, style: str = "BodyText") -> Paragraph:
            return Paragraph(text, styles[style])

        def card(label: str, value: str, detail: str, color_hex: str):
            return [
                p(f'<font color="#64748B" size="7"><b>{label}</b></font>'),
                p(f'<font color="{color_hex}" size="18"><b>{value}</b></font>'),
                p(f'<font color="#475569" size="8">{detail}</font>'),
            ]

        story = []
        story.append(
            Table(
                [[p("GreenChem AI", "Title"), p(f"Scientific Report<br/>{datetime.now().date().isoformat()}")]],
                colWidths=[4.6 * inch, 2.2 * inch],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ECFEFF")),
                        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#99F6E4")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 14),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                        ("TOPPADDING", (0, 0), (-1, -1), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                ),
            )
        )
        story.append(Spacer(1, 12))

        improvement = best["green_score"] - current["green_score"]
        confidence = recommendation_confidence(current, best, recommendations)
        risk = risk_level(best)
        summary_cards = Table(
            [
                [
                    card("CURRENT SCORE", f"{current['green_score']:.1f}", current["solvent"], score_color(current["green_score"])),
                    card("RECOMMENDED", best["solvent"], best["chem21_classification"], "#0F766E"),
                    card("OPTIMIZED SCORE", f"{best['green_score']:.1f}", f"{improvement:+.1f} point gain", score_color(best["green_score"])),
                ]
            ],
            colWidths=[2.18 * inch, 2.18 * inch, 2.18 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            ),
        )
        story.append(summary_cards)
        story.append(Spacer(1, 14))

        story.append(p("Decision Card", "Heading2"))
        story.append(
            p(
                f"Recommendation confidence: <b>{confidence['score']:.0f}%</b>. "
                f"Experimental risk: <b>{risk['label']}</b>. "
                f"Compatibility: {confidence['compatibility']}. Historical acceptance: {confidence['historical']}. "
                f"Data coverage: {confidence['data_coverage']}."
            )
        )
        story.append(Spacer(1, 10))

        story.append(p("Before vs After", "Heading2"))
        ba_rows = [
            ["GreenScore", f"{current['green_score']:.1f}", f"{best['green_score']:.1f}", f"{best['green_score'] - current['green_score']:+.1f}"],
            ["Toxicity", f"{current['toxicity_score']:.1f}", f"{best['toxicity_score']:.1f}", f"{best['toxicity_score'] - current['toxicity_score']:+.1f}"],
            ["E-Factor", f"{current['e_factor']:.3f}", f"{best['e_factor']:.3f}", f"{best['e_factor'] - current['e_factor']:+.3f}"],
            ["Yield", f"{current['yield_percent']:.1f}", f"{best['yield_percent']:.1f}", f"{best['yield_percent'] - current['yield_percent']:+.1f}"],
            ["Waste kg", f"{current['waste_kg']:.1f}", f"{best['waste_kg']:.1f}", f"{best['waste_kg'] - current['waste_kg']:+.1f}"],
            ["Atom Economy", f"{current['atom_economy']:.1f}", f"{best['atom_economy']:.1f}", f"{best['atom_economy'] - current['atom_economy']:+.1f}"],
        ]
        ba_data = [["Metric", "Current", "Optimized", "Change"]] + ba_rows
        ba_table = Table(
            ba_data,
            colWidths=[2.1 * inch, 1.45 * inch, 1.45 * inch, 1.1 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        )
        story.append(ba_table)
        story.append(Spacer(1, 14))

        toxicity_reduction = current["toxicity_score"] - best["toxicity_score"]
        story.append(p("Recommendation Summary", "Heading2"))
        bullets = [
            f"<b>{best['solvent']}</b> was selected by deterministic scoring, not by the LLM.",
            f"Toxicity score reduced by <b>{toxicity_reduction:.1f}</b> points.",
            f"GreenScore improved by <b>{improvement:.1f}</b> points.",
            f"Compatible with <b>{current['reaction_type']}</b>.",
            f"Estimated waste changed from <b>{current['waste_kg']:.1f} kg</b> to <b>{best['waste_kg']:.1f} kg</b>.",
        ]
        for item in bullets:
            story.append(p(f'<font color="#22C55E"><b>OK</b></font> {item}'))
        story.append(Spacer(1, 12))

        story.append(p("Top Greener Alternatives", "Heading2"))
        alt_data = [["Rank", "Solvent", "Rank Score", "GreenScore", "Toxicity", "E-Factor"]]
        for idx, rec in enumerate(recommendations, start=1):
            alt_data.append(
                [
                    str(idx),
                    rec["solvent"],
                    f"{rec['rank_score']:.1f}",
                    f"{rec['green_score']:.1f}",
                    f"{rec['toxicity_score']:.1f}",
                    f"{rec['e_factor']:.3f}",
                ]
            )
        alt_table = Table(
            alt_data,
            colWidths=[0.45 * inch, 1.9 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14B8A6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        )
        story.append(alt_table)
        story.append(Spacer(1, 14))

        story.append(p("Decision Factors", "Heading2"))
        factor_table = Table(
            [[
                p('<font color="#22C55E"><b>OK</b></font><br/>Compatibility'),
                p('<font color="#22C55E"><b>OK</b></font><br/>Toxicity Reduction'),
                p('<font color="#22C55E"><b>OK</b></font><br/>GreenScore Gain'),
                p('<font color="#22C55E"><b>OK</b></font><br/>Waste Reduction'),
            ]],
            colWidths=[1.6 * inch, 1.6 * inch, 1.6 * inch, 1.6 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDFA")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#99F6E4")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCFBF1")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            ),
        )
        story.append(factor_table)
        story.append(Spacer(1, 12))

        story.append(p("Why Not the Other Alternatives?", "Heading2"))
        for rec in recommendations[1:4]:
            reason = "It ranked lower after all weighted score components were combined."
            if rec["green_score"] < best["green_score"]:
                reason = f"GreenScore is {best['green_score'] - rec['green_score']:.1f} points lower than {best['solvent']}."
            elif rec["toxicity_score"] > best["toxicity_score"]:
                reason = f"Toxicity score is {rec['toxicity_score'] - best['toxicity_score']:.1f} points higher than {best['solvent']}."
            elif rec["e_factor"] > best["e_factor"]:
                reason = f"E-Factor is {rec['e_factor'] - best['e_factor']:.3f} higher than {best['solvent']}."
            story.append(p(f"- Why not {rec['solvent']}? {reason}"))

        story.append(p("Explanation", "Heading2"))
        story.append(p(explanation[:1200]))
        story.append(Spacer(1, 10))

        story.append(p("Scientific Sources", "Heading2"))
        for source in sources:
            story.append(p(f"- {source}"))

        story.append(Spacer(1, 10))
        story.append(p("Model Limitations", "Heading2"))
        for item in xai["limitations"]:
            story.append(p(f"- {item}"))

        doc.build(story)
        return buffer.getvalue()
    except Exception:
        return _minimal_pdf_from_text(report_text)


def analysis_page(solvents: pd.DataFrame, reactions: pd.DataFrame) -> None:
    st.title("GreenChem AI")
    st.caption("Deterministic green solvent substitution and process optimization. Llama explains; it does not decide.")

    if "analysis_step" not in st.session_state:
        st.session_state["analysis_step"] = 1

    step = st.session_state["analysis_step"]
    steps = ["Input", "Summary", "Before / After", "Recommendation", "Advanced"]
    progress_cols = st.columns(len(steps))
    for idx, label in enumerate(steps, start=1):
        active = idx == step
        done = idx < step
        bg = "#0F766E" if active else "#D1FAE5" if done else "#FFFFFF"
        fg = "#F8FAFC" if active else "#0F766E" if done else "#64748B"
        border = "#0F766E" if active or done else "#CBD5E1"
        progress_cols[idx - 1].markdown(
            f"""
            <div style="border:1px solid {border}; background:{bg}; color:{fg}; border-radius:8px; padding:10px; text-align:center; font-weight:800;">
                {idx}. {label}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if step == 1:
        st.subheader("Process Input")
        st.markdown(pipeline_diagram(), unsafe_allow_html=True)
        prefill = st.session_state.get("assistant_prefill", {})
        reaction_options = reaction_types(reactions)
        solvent_options = solvent_names(solvents)
        default_reaction = prefill.get("reaction_type", "Esterification")
        default_solvent = prefill.get("current_solvent", "DMF")
        reaction_index = reaction_options.index(default_reaction) if default_reaction in reaction_options else reaction_options.index("Esterification")
        solvent_index = solvent_options.index(default_solvent) if default_solvent in solvent_options else solvent_options.index("DMF")
        with st.form("process-form"):
            c1, c2 = st.columns(2)
            reaction_type = c1.selectbox("Reaction type", reaction_options, index=reaction_index)
            current_solvent = c2.selectbox("Current solvent", solvent_options, index=solvent_index)
            c3, c4 = st.columns(2)
            yield_percent = c3.number_input("Yield percentage", min_value=1.0, max_value=100.0, value=float(prefill.get("yield_percent", 65.0)), step=1.0)
            waste_kg = c4.number_input("Waste produced in kg", min_value=0.0, value=float(prefill.get("waste_kg", 40.0)), step=1.0)
            notes = st.text_area("Optional reaction notes", value=prefill.get("notes", ""), placeholder="Catalyst, temperature, workup concerns, solvent constraints...")
            model_name = st.text_input("Local Ollama model", value="llama3")
            submitted = st.form_submit_button("Analyze Process", type="primary")

        st.info("Demo input: Esterification, DMF, 65% yield, 40 kg waste.")

        if submitted:
            reaction = get_reaction(reaction_type, reactions)
            solvent = get_solvent(current_solvent, solvents)
            current = analyze_current_process(reaction, solvent, yield_percent, waste_kg, solvents)
            recommendations = recommend_solvents(reaction, solvent, yield_percent, waste_kg, solvents)
            if not recommendations:
                st.error("No compatible alternatives found for this reaction in the local solvent database.")
                return
            best = recommendations[0]
            xai = recommendation_xai(best, current)
            query = f"{reaction_type} solvent substitution E-Factor toxicity {notes}"
            rag_context, rag_sources = retrieve_scientific_context(query)
            expert_context, expert_sources = retrieve_expert_memory_context(
                reaction_type,
                current_solvent,
                recommended_solvent=best["solvent"],
            )
            explanation, ollama_ok = explain_with_ollama(
                current,
                best,
                xai,
                rag_context,
                expert_context=expert_context,
                model_name=model_name,
            )
            st.session_state["analysis"] = {
                "current": current,
                "recommendations": recommendations,
                "best": best,
                "xai": xai,
                "rag_context": rag_context,
                "rag_sources": rag_sources,
                "expert_context": expert_context,
                "expert_sources": expert_sources,
                "explanation": explanation,
                "ollama_ok": ollama_ok,
                "notes": notes,
            }
            st.session_state["analysis_step"] = 2
            st.rerun()
        return

    if "analysis" not in st.session_state:
        st.warning("Start with process input first.")
        if st.button("Go to input"):
            st.session_state["analysis_step"] = 1
            st.rerun()
        return

    result = st.session_state["analysis"]
    current = result["current"]
    best = result["best"]
    recommendations = result["recommendations"]
    xai = result["xai"]
    improvement = round(best["green_score"] - current["green_score"], 2)
    toxicity_reduction = round(current["toxicity_score"] - best["toxicity_score"], 2)
    toxicity_pct = round((toxicity_reduction / max(current["toxicity_score"], 1.0)) * 100.0, 1)

    def nav_buttons(back_step: int | None, next_step: int | None, next_label: str = "Next") -> None:
        left, right = st.columns([0.5, 0.5])
        if back_step is not None and left.button("Back", use_container_width=True):
            st.session_state["analysis_step"] = back_step
            st.rerun()
        if next_step is not None and right.button(next_label, type="primary", use_container_width=True):
            st.session_state["analysis_step"] = next_step
            st.rerun()

    if step == 2:
        st.subheader("Step 2: Results Summary")
        st.markdown(decision_card(current, best, recommendations), unsafe_allow_html=True)
        st.markdown(process_flow(current, best), unsafe_allow_html=True)
        st.markdown(confidence_panel(current, best, recommendations), unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        h1.markdown(
            hero_card(
                "Current Process",
                f"{current['green_score']:.1f}",
                f"{current['solvent']} GreenScore",
                score_color(current["green_score"]),
            ),
            unsafe_allow_html=True,
        )
        h2.markdown(
            hero_card(
                "Recommended Solvent",
                best["solvent"],
                f"{best['chem21_classification']} classification",
                "#0F766E",
            ),
            unsafe_allow_html=True,
        )
        h3.markdown(
            hero_card(
                "Optimized Score",
                f"{best['green_score']:.1f}",
                f"{improvement:+.1f} point improvement",
                score_color(best["green_score"]),
            ),
            unsafe_allow_html=True,
        )
        nav_buttons(1, 3)
        return

    if step == 3:
        st.subheader("Step 3: Before / After")
        st.markdown(before_after_insights(current, best), unsafe_allow_html=True)
        st.dataframe(before_after_table(current, best), hide_index=True, use_container_width=True)
        with st.expander("What these metrics mean"):
            st.write("- GreenScore combines solvent sustainability and process performance into a 0-100 score.")
            st.write("- Toxicity is a hazard score, so lower is better.")
            st.write("- E-Factor estimates waste per useful product mass, so lower is better.")
            st.write("- Yield and atom economy are efficiency metrics, so higher is better.")
        nav_buttons(2, 4)
        return

    if step == 4:
        st.subheader("Step 4: Recommendation Summary")
        st.markdown(decision_card(current, best, recommendations), unsafe_allow_html=True)
        st.markdown(
            recommendation_story(current, best, improvement, toxicity_reduction, toxicity_pct),
            unsafe_allow_html=True,
        )
        st.markdown("#### Why not the other alternatives?")
        st.markdown(why_not_alternatives(best, recommendations), unsafe_allow_html=True)
        st.caption("The LLM did not choose this solvent. It only explains the deterministic result below.")
        with st.expander("Show full explanation"):
            status = "Ollama explanation" if result["ollama_ok"] else "Fallback explanation"
            st.caption(status)
            for paragraph in result["explanation"].split("\n\n"):
                if paragraph.strip():
                    st.write(paragraph.strip())
        nav_buttons(3, 5, "Advanced analysis")
        return

    if step == 5:
        st.subheader("Step 5: Advanced Analysis")
        st.caption("Expert review workspace: compare alternatives, inspect score drivers, review retrieved sources, and validate the recommendation.")
        overview_tab, xai_tab, sources_tab, validation_tab = st.tabs(
            ["Overview", "XAI", "Scientific Sources", "Validation"]
        )

        with overview_tab:
            st.markdown("#### Ranked Greener Alternatives")
            st.markdown(alternatives_cards(recommendations), unsafe_allow_html=True)
            st.markdown("#### Similar Historical Cases")
            st.markdown(expert_memory_summary(recommendations), unsafe_allow_html=True)
            with st.expander("Detailed ranking table"):
                st.dataframe(recommendations_table(recommendations), hide_index=True, use_container_width=True)

        with xai_tab:
            st.markdown("#### Decision Factors")
            f1, f2, f3, f4 = st.columns(4)
            f1.markdown('<div class="gc-factor"><span class="gc-check">OK</span><br>Compatibility</div>', unsafe_allow_html=True)
            f2.markdown('<div class="gc-factor"><span class="gc-check">OK</span><br>Toxicity Reduction</div>', unsafe_allow_html=True)
            f3.markdown('<div class="gc-factor"><span class="gc-check">OK</span><br>GreenScore Gain</div>', unsafe_allow_html=True)
            f4.markdown('<div class="gc-factor"><span class="gc-check">OK</span><br>Waste Reduction</div>', unsafe_allow_html=True)
            st.markdown(confidence_panel(current, best, recommendations), unsafe_allow_html=True)
            risk = risk_level(best)
            st.markdown(
                f"""
                <div class="gc-source-card">
                    <p style="color:#64748B; font-size:12px; font-weight:900; text-transform:uppercase; margin:0 0 6px 0;">Experimental risk</p>
                    <span class="gc-risk-badge" style="background:{risk['color']};">{risk['label']}</span>
                    <p style="color:#475569; margin:8px 0 0 0;">{risk['summary']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### GreenScore Components")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Current process")
                st.markdown(contribution_bars(current["contributions"]), unsafe_allow_html=True)
            with c2:
                st.caption("Optimized process")
                st.markdown(contribution_bars(best["contributions"]), unsafe_allow_html=True)
            st.info(
                "The ranking is deterministic. Llama receives these calculated values only after the scoring engine has selected the recommendation."
            )

            with st.expander("Technical details"):
                for item in xai["decision_trace"]:
                    st.write(f"- {item}")
                st.markdown("##### Score Contributions")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Current process")
                    st.dataframe(contributions_table(current["contributions"]), hide_index=True, use_container_width=True)
                with c2:
                    st.caption("Optimized process")
                    st.dataframe(contributions_table(best["contributions"]), hide_index=True, use_container_width=True)

            with st.expander("About model limitations"):
                for item in xai["limitations"]:
                    st.write(f"- {item}")

        with sources_tab:
            st.markdown("#### Scientific RAG Context")
            for source, chunk in zip(result["rag_sources"], result["rag_context"]):
                st.markdown(
                    f"""
                    <div class="gc-source-card">
                        <p style="color:#0F766E; font-weight:850; margin:0 0 6px 0;">{source}</p>
                        <p style="color:#475569; margin:0;">{chunk}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("#### Expert Memory RAG")
            if result.get("expert_context"):
                for source, chunk in zip(result["expert_sources"], result["expert_context"]):
                    st.markdown(
                        f"""
                        <div class="gc-source-card" style="background:#FFFBEB; border-color:#FCD34D;">
                            <p style="color:#92400E; font-weight:850; margin:0 0 6px 0;">{source}</p>
                            <p style="color:#78350F; margin:0;">{chunk}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No prior expert validation matched this process yet. New validation will be retrieved in future predictions.")

        with validation_tab:
            st.markdown("#### Human Validation")
            st.markdown(
                f"""
                <div class="gc-source-card">
                    <p style="color:#64748B; font-size:12px; font-weight:850; text-transform:uppercase; margin:0 0 6px 0;">Recommendation to validate</p>
                    <h3 style="color:#0F172A; margin:0;">{best['solvent']}</h3>
                    <p style="color:#475569; margin:6px 0 0 0;">Your decision will be saved as expert memory and can adjust future rankings for similar processes.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            comment = st.text_input("Feedback comment", key="feedback-comment")
            b1, b2, b3 = st.columns(3)
            for button, decision in [(b1, "Accept"), (b2, "Reject"), (b3, "Request Alternative")]:
                if button.button(decision, use_container_width=True):
                    store_feedback(current["reaction_type"], current["solvent"], best["solvent"], decision, comment)
                    st.success(f"Stored feedback: {decision}. Future rankings for similar processes will use this expert memory.")

            combined_sources = result["rag_sources"] + result.get("expert_sources", [])
            report = build_pdf_report(current, best, recommendations, xai, result["explanation"], combined_sources)
            st.download_button(
                "Download PDF report",
                data=report,
                file_name="greenchem_ai_report.pdf",
                mime="application/pdf",
            )

        nav_buttons(4, None)
        if st.button("Start new analysis"):
            st.session_state["analysis_step"] = 1
            st.session_state.pop("analysis", None)
            st.rerun()
        return

    st.session_state["analysis_step"] = 1
    st.rerun()


def feedback_page() -> None:
    st.title("Feedback History")
    feedback = load_feedback()
    if feedback.empty:
        st.info("No feedback recorded yet.")
        return
    st.dataframe(feedback.sort_values("timestamp", ascending=False), hide_index=True, use_container_width=True)
    st.download_button("Download feedback CSV", feedback.to_csv(index=False), "feedback.csv", "text/csv")


def data_page(solvents: pd.DataFrame, reactions: pd.DataFrame) -> None:
    st.title("Local Scientific Data")
    tab1, tab2 = st.tabs(["Solvents", "Reactions"])
    with tab1:
        st.dataframe(solvents.drop(columns=["name_key"]), hide_index=True, use_container_width=True)
    with tab2:
        st.dataframe(reactions.drop(columns=["reaction_key"]), hide_index=True, use_container_width=True)


def _classification_color(label: str) -> str:
    colors = {
        "Preferred": "#22C55E",
        "Recommended": "#84CC16",
        "Problematic": "#F59E0B",
        "Hazardous": "#EF4444",
    }
    return colors.get(label, "#CBD5E1")


def _score_color(score: float) -> str:
    if score < 40:
        return "#EF4444"
    if score < 70:
        return "#F59E0B"
    return "#22C55E"


def flashcards_page(solvents: pd.DataFrame) -> None:
    st.title("Solvent Flashcards")
    st.caption("Compare solvent hazard, sustainability, process fit, and substitution quality from the local GreenChem AI dataset.")

    classifications = ["All"] + sorted(solvents["chem21_classification"].dropna().unique().tolist())
    c1, c2, c3 = st.columns([0.42, 0.29, 0.29])
    selected = c1.selectbox("Focus solvent", solvent_names(solvents), index=solvent_names(solvents).index("DMF"))
    classification = c2.selectbox("Classification", classifications)
    sort_by = c3.selectbox("Sort cards by", ["Green score", "Lowest toxicity", "Lowest regulatory risk", "Boiling point"])

    focus = solvents[solvents["name"] == selected].iloc[0]
    focus_class_color = _classification_color(focus["chem21_classification"])
    focus_score_color = _score_color(float(focus["green_score"]))

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#FFFFFF 0%,#ECFEFF 100%); border:1px solid #CBD5E1; border-radius:8px; padding:18px; box-shadow:0 10px 30px rgba(15,23,42,0.07); margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; gap:14px; align-items:flex-start; flex-wrap:wrap;">
                <div>
                    <p style="color:#0F766E; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:0.08em; margin:0 0 6px 0;">Focused solvent</p>
                    <h2 style="color:#0F172A; margin:0;">{focus['name']}</h2>
                    <p style="color:#475569; margin:6px 0 0 0;">{focus['smiles']}</p>
                </div>
                <span style="background:{focus_class_color}; color:#0F172A; border-radius:999px; padding:6px 12px; font-size:13px; font-weight:800;">{focus['chem21_classification']}</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:12px; margin-top:16px;">
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px;"><p style="color:#64748B; margin:0; font-size:12px;">Green score</p><strong style="color:{focus_score_color}; font-size:24px;">{focus['green_score']}</strong></div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px;"><p style="color:#64748B; margin:0; font-size:12px;">Toxicity</p><strong style="color:{_score_color(100 - float(focus['toxicity_score']))}; font-size:24px;">{focus['toxicity_score']}</strong></div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px;"><p style="color:#64748B; margin:0; font-size:12px;">Regulatory risk</p><strong style="color:{_score_color(100 - float(focus['regulatory_risk']))}; font-size:24px;">{focus['regulatory_risk']}</strong></div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px;"><p style="color:#64748B; margin:0; font-size:12px;">Boiling point</p><strong style="color:#0F172A; font-size:24px;">{focus['boiling_point']} C</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    view = solvents.copy()
    if classification != "All":
        view = view[view["chem21_classification"] == classification]
    sort_map = {
        "Green score": ("green_score", False),
        "Lowest toxicity": ("toxicity_score", True),
        "Lowest regulatory risk": ("regulatory_risk", True),
        "Boiling point": ("boiling_point", True),
    }
    sort_col, ascending = sort_map[sort_by]
    view = view.sort_values(sort_col, ascending=ascending)

    st.markdown("#### Solvent Library")
    cols = st.columns(3)
    for idx, (_, row) in enumerate(view.iterrows()):
        class_color = _classification_color(row["chem21_classification"])
        score_color = _score_color(float(row["green_score"]))
        toxicity_color = _score_color(100 - float(row["toxicity_score"]))
        reg_color = _score_color(100 - float(row["regulatory_risk"]))
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:14px; min-height:330px; box-shadow:0 8px 24px rgba(15, 23, 42, 0.06); margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; gap:8px; align-items:flex-start;">
                        <h3 style="margin:0 0 8px 0; color:#0F172A;">{row['name']}</h3>
                        <span style="background:{class_color}; color:#0F172A; border-radius:999px; padding:4px 9px; font-size:12px; font-weight:700;">{row['chem21_classification']}</span>
                    </div>
                    <p style="color:#475569; margin:0 0 12px 0;"><strong>SMILES:</strong> {row['smiles']}</p>
                    <p style="color:#64748B; margin:0 0 4px 0; font-size:12px; font-weight:700;">Green score</p>
                    <div style="height:8px; background:#E2E8F0; border-radius:999px; overflow:hidden; margin-bottom:10px;">
                        <div style="width:{row['green_score']}%; height:8px; background:{score_color};"></div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                        <div style="background:#F8FAFC; border-radius:8px; padding:8px;"><span style="color:#64748B; font-size:12px;">Green</span><br><strong style="color:{score_color};">{row['green_score']}</strong></div>
                        <div style="background:#F8FAFC; border-radius:8px; padding:8px;"><span style="color:#64748B; font-size:12px;">Toxicity</span><br><strong style="color:{toxicity_color};">{row['toxicity_score']}</strong></div>
                        <div style="background:#F8FAFC; border-radius:8px; padding:8px;"><span style="color:#64748B; font-size:12px;">Biodeg.</span><br><strong style="color:#0F766E;">{row['biodegradability_score']}</strong></div>
                        <div style="background:#F8FAFC; border-radius:8px; padding:8px;"><span style="color:#64748B; font-size:12px;">Reg. risk</span><br><strong style="color:{reg_color};">{row['regulatory_risk']}</strong></div>
                    </div>
                    <div style="border-top:1px solid #E2E8F0; margin-top:12px; padding-top:10px;">
                        <p style="color:#475569; margin:0 0 4px 0;"><strong>BP:</strong> {row['boiling_point']} C | <strong>Polarity:</strong> {row['polarity_index']}</p>
                        <p style="color:#475569; margin:0;"><strong>GSK-style score:</strong> {row['gsk_score']} | <strong>VOC:</strong> {row['voc_score']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    solvents, reactions = cached_data()
    pages = ["AI Assistant", "Analysis", "Solvent Flashcards", "Feedback History", "Local Data"]
    current_page = st.session_state.get("current_page", "AI Assistant")
    page_index = pages.index(current_page) if current_page in pages else 0
    with st.sidebar:
        st.header("GreenChem AI")
        page = st.radio(
            "Navigation",
            pages,
            index=page_index,
        )
        st.session_state["current_page"] = page
        st.toggle("Mute assistant voice", key="voice_muted", value=st.session_state.get("voice_muted", False))
        st.markdown("---")
        st.caption("Decision engine: RDKit features, Random Forest toxicity support, transparent GreenScore, ChromaDB RAG, local Ollama explanation.")

    if page == "AI Assistant":
        assistant_page(solvents, reactions)
    elif page == "Analysis":
        analysis_page(solvents, reactions)
    elif page == "Solvent Flashcards":
        flashcards_page(solvents)
    elif page == "Feedback History":
        feedback_page()
    else:
        data_page(solvents, reactions)


if __name__ == "__main__":
    main()
