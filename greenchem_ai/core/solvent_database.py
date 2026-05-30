from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "solvents.csv"


def load_solvents(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load solvent property data with normalized helper columns."""
    df = pd.read_csv(path)
    numeric_cols = [
        "toxicity_score",
        "green_score",
        "voc_score",
        "biodegradability_score",
        "regulatory_risk",
        "boiling_point",
        "polarity_index",
        "gsk_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["name_key"] = df["name"].str.casefold()
    return df


def get_solvent(name: str, solvents: pd.DataFrame | None = None) -> pd.Series:
    solvents = solvents if solvents is not None else load_solvents()
    match = solvents[solvents["name_key"] == name.strip().casefold()]
    if match.empty:
        raise ValueError(f"Unknown solvent: {name}")
    return match.iloc[0]


def solvent_names(solvents: pd.DataFrame | None = None) -> list[str]:
    solvents = solvents if solvents is not None else load_solvents()
    return solvents["name"].sort_values().tolist()
