from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKEND_SOLVENTS_PATH = ROOT / "data" / "solvents_backend.csv"

REACTION_KEY_MAP = {
    "Esterification": "esterification",
    "Amide Coupling": "amide_coupling",
    "Suzuki Coupling": "suzuki_coupling",
    "Grignard Reaction": "grignard",
    "Reductive Amination": "reductive_amination",
    "Aldol Reaction": "aldol",
    "Wittig Reaction": "wittig",
    "SN2 Substitution": "sn2",
}

REACTION_BACKEND = {
    "esterification": {"yield": 0.75, "atom_economy": 0.85, "solvent_ratio": 5.0},
    "amide_coupling": {"yield": 0.80, "atom_economy": 0.45, "solvent_ratio": 10.0},
    "suzuki_coupling": {"yield": 0.85, "atom_economy": 0.72, "solvent_ratio": 8.0},
    "grignard": {"yield": 0.70, "atom_economy": 0.90, "solvent_ratio": 15.0},
    "reductive_amination": {"yield": 0.78, "atom_economy": 0.60, "solvent_ratio": 8.0},
    "aldol": {"yield": 0.65, "atom_economy": 0.95, "solvent_ratio": 6.0},
    "wittig": {"yield": 0.72, "atom_economy": 0.40, "solvent_ratio": 10.0},
    "sn2": {"yield": 0.80, "atom_economy": 0.78, "solvent_ratio": 6.0},
}


def load_backend_solvents(path: Path | str = BACKEND_SOLVENTS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["name_key"] = df["name"].str.casefold()
    return df


def hsp_distance(row_a: pd.Series, row_b: pd.Series) -> float | None:
    required = ["hsp_d", "hsp_p", "hsp_h"]
    if any(col not in row_a or col not in row_b for col in required):
        return None
    return round(
        float(
            np.sqrt(
                4 * (float(row_b["hsp_d"]) - float(row_a["hsp_d"])) ** 2
                + (float(row_b["hsp_p"]) - float(row_a["hsp_p"])) ** 2
                + (float(row_b["hsp_h"]) - float(row_a["hsp_h"])) ** 2
            )
        ),
        2,
    )


def estimate_backend_efactor(reaction_type: str, solvent_name: str, substrate_mass_g: float = 10.0) -> dict:
    backend_key = REACTION_KEY_MAP.get(reaction_type, reaction_type.casefold())
    rxn = REACTION_BACKEND.get(backend_key)
    solvents = load_backend_solvents()
    if rxn is None or solvents.empty:
        return {}
    match = solvents[solvents["name_key"] == solvent_name.casefold()]
    density = 0.9 if match.empty else float(match.iloc[0].get("density", 0.9))
    product_mass_g = substrate_mass_g * rxn["yield"] * rxn["atom_economy"]
    solvent_mass_g = substrate_mass_g * rxn["solvent_ratio"] * density
    reagent_waste_g = substrate_mass_g * 1.2 * (1 - rxn["atom_economy"])
    total_waste_g = solvent_mass_g + reagent_waste_g
    efactor = total_waste_g / product_mass_g if product_mass_g else float("inf")
    return {
        "backend_reaction_key": backend_key,
        "density": round(density, 3),
        "product_mass_g": round(product_mass_g, 2),
        "solvent_mass_g": round(solvent_mass_g, 2),
        "total_waste_g": round(total_waste_g, 2),
        "efactor": round(efactor, 2),
        "atom_economy_pct": round(rxn["atom_economy"] * 100, 1),
        "typical_yield_pct": round(rxn["yield"] * 100, 1),
    }


def structural_alerts(smiles: str) -> list[str]:
    alerts: list[str] = []
    if not smiles:
        return alerts
    if any(token in smiles for token in ["Cl", "Br", "F", "I"]):
        alerts.append("Halogenated structure: potential persistence or bioaccumulation concern.")
    if "[N+](=O)[O-]" in smiles or "N(=O)=O" in smiles:
        alerts.append("Nitro functionality: potential metabolic activation concern.")
    if "C=O" in smiles and "[H]" in smiles:
        alerts.append("Aldehyde-like electrophile: possible reactivity concern.")
    return alerts


def bridge_evidence(reaction_type: str, current_solvent: str, recommended_solvent: str) -> dict:
    solvents = load_backend_solvents()
    if solvents.empty:
        return {"available": False, "reason": "Backend solvent evidence database is not available."}

    current = solvents[solvents["name_key"] == current_solvent.casefold()]
    proposed = solvents[solvents["name_key"] == recommended_solvent.casefold()]
    if current.empty or proposed.empty:
        return {
            "available": False,
            "reason": "Current or recommended solvent is not present in the backend evidence database.",
        }

    current_row = current.iloc[0]
    proposed_row = proposed.iloc[0]
    current_ef = estimate_backend_efactor(reaction_type, current_solvent)
    proposed_ef = estimate_backend_efactor(reaction_type, recommended_solvent)
    ef_delta = None
    if current_ef and proposed_ef:
        ef_delta = round(current_ef["efactor"] - proposed_ef["efactor"], 2)

    return {
        "available": True,
        "source": "First-part backend bridge: HSP + density E-factor + structural alerts",
        "current_solvent": current_solvent,
        "recommended_solvent": recommended_solvent,
        "hsp_distance": hsp_distance(current_row, proposed_row),
        "current_backend_efactor": current_ef,
        "recommended_backend_efactor": proposed_ef,
        "backend_efactor_delta": ef_delta,
        "current_structural_alerts": structural_alerts(str(current_row.get("smiles", ""))),
        "recommended_structural_alerts": structural_alerts(str(proposed_row.get("smiles", ""))),
        "recommended_family": proposed_row.get("family", "unknown"),
        "recommended_density": float(proposed_row.get("density", 0.0)),
    }
