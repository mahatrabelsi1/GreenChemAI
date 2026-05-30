from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "reactions.csv"


def load_reactions(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["average_yield", "atom_economy", "waste_factor", "energy_factor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["reaction_key"] = df["reaction_type"].str.casefold()
    return df


def get_reaction(reaction_type: str, reactions: pd.DataFrame | None = None) -> pd.Series:
    reactions = reactions if reactions is not None else load_reactions()
    match = reactions[reactions["reaction_key"] == reaction_type.strip().casefold()]
    if match.empty:
        raise ValueError(f"Unknown reaction type: {reaction_type}")
    return match.iloc[0]


def reaction_types(reactions: pd.DataFrame | None = None) -> list[str]:
    reactions = reactions if reactions is not None else load_reactions()
    return reactions["reaction_type"].sort_values().tolist()


def compatible_solvents(reaction: pd.Series) -> list[str]:
    raw = str(reaction.get("compatible_solvents", ""))
    return [item.strip() for item in raw.split(";") if item.strip()]
