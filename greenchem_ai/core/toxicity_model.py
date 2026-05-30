from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "toxicity_model.joblib"


def _fallback_features(smiles: str) -> list[float]:
    smiles = smiles or ""
    hetero = sum(smiles.count(token) for token in ["O", "N", "S", "Cl", "Br", "F"])
    carbons = smiles.count("C") + smiles.count("c")
    rings = sum(ch.isdigit() for ch in smiles)
    halogens = smiles.count("Cl") + smiles.count("Br") + smiles.count("F")
    return [len(smiles), carbons, hetero, rings, halogens, 0.0, 0.0]


def smiles_features(smiles: str) -> list[float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _fallback_features(smiles)
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
        ]
    except Exception:
        return _fallback_features(smiles)


def train_or_load_model(solvents: pd.DataFrame, path: Path | str = MODEL_PATH) -> RandomForestRegressor:
    path = Path(path)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass
    X = np.array([smiles_features(smi) for smi in solvents["smiles"]])
    y = solvents["toxicity_score"].astype(float).to_numpy()
    model = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=1)
    model.fit(X, y)
    try:
        joblib.dump(model, path)
    except Exception:
        pass
    return model


def predict_toxicity(smiles: str, solvents: pd.DataFrame) -> float:
    try:
        model = train_or_load_model(solvents)
        pred = float(model.predict([smiles_features(smiles)])[0])
        return round(max(0.0, min(100.0, pred)), 2)
    except Exception:
        return round(float(np.mean(solvents["toxicity_score"])), 2)
