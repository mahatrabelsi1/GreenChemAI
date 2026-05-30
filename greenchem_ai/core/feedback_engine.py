from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT / "data" / "feedback.csv"
COLUMNS = [
    "timestamp",
    "reaction_type",
    "current_solvent",
    "recommended_solvent",
    "decision",
    "comment",
    "weight_delta",
]


def load_feedback(path: Path | str = FEEDBACK_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for col in COLUMNS:
        if col not in df:
            df[col] = ""
    return df[COLUMNS]


def store_feedback(
    reaction_type: str,
    current_solvent: str,
    recommended_solvent: str,
    decision: str,
    comment: str = "",
    path: Path | str = FEEDBACK_PATH,
) -> None:
    delta_map = {"Accept": 3.0, "Reject": -5.0, "Request Alternative": -2.0}
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reaction_type": reaction_type,
        "current_solvent": current_solvent,
        "recommended_solvent": recommended_solvent,
        "decision": decision,
        "comment": comment,
        "weight_delta": delta_map.get(decision, 0.0),
    }
    df = load_feedback(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def feedback_adjustment(
    solvent_name: str,
    reaction_type: str,
    feedback: pd.DataFrame | None = None,
    current_solvent: str | None = None,
) -> float:
    feedback = feedback if feedback is not None else load_feedback()
    if feedback.empty:
        return 0.0
    reaction_match = feedback["reaction_type"].str.casefold().eq(reaction_type.casefold())
    solvent_match = feedback["recommended_solvent"].str.casefold().eq(solvent_name.casefold())
    mask = reaction_match & solvent_match
    if current_solvent:
        same_current = feedback["current_solvent"].str.casefold().eq(current_solvent.casefold())
        values = pd.to_numeric(feedback.loc[mask & same_current, "weight_delta"], errors="coerce").fillna(0.0)
        close_values = pd.to_numeric(feedback.loc[mask & ~same_current, "weight_delta"], errors="coerce").fillna(0.0)
        return float(values.tail(10).sum() + close_values.tail(10).sum() * 0.45)
    values = pd.to_numeric(feedback.loc[mask, "weight_delta"], errors="coerce").fillna(0.0)
    return float(values.tail(10).sum())


def expert_memory_matches(
    reaction_type: str,
    current_solvent: str,
    recommended_solvent: str | None = None,
    feedback: pd.DataFrame | None = None,
    limit: int = 5,
) -> list[dict]:
    feedback = feedback if feedback is not None else load_feedback()
    if feedback.empty:
        return []
    scored: list[tuple[float, dict]] = []
    for _, row in feedback.sort_values("timestamp", ascending=False).iterrows():
        score = 0.0
        if str(row["reaction_type"]).casefold() == reaction_type.casefold():
            score += 3.0
        if str(row["current_solvent"]).casefold() == current_solvent.casefold():
            score += 2.0
        if recommended_solvent and str(row["recommended_solvent"]).casefold() == recommended_solvent.casefold():
            score += 2.0
        if score <= 0:
            continue
        scored.append(
            (
                score,
                {
                    "timestamp": row["timestamp"],
                    "reaction_type": row["reaction_type"],
                    "current_solvent": row["current_solvent"],
                    "recommended_solvent": row["recommended_solvent"],
                    "decision": row["decision"],
                    "comment": row["comment"],
                    "weight_delta": row["weight_delta"],
                    "match_score": score,
                },
            )
        )
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]
