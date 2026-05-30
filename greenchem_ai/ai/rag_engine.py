from __future__ import annotations

from pathlib import Path

from core.feedback_engine import expert_memory_matches, load_feedback


ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = ROOT / "data" / "chroma_store"
EXPERT_CHROMA_PATH = ROOT / "data" / "expert_memory_store"

SCIENTIFIC_CHUNKS = [
    {
        "source": "Green Chemistry Principle 1 - Prevention",
        "text": "Green chemistry principle 1 emphasizes prevention: it is better to prevent waste than to treat or clean it up after it has been created.",
    },
    {
        "source": "Solvent substitution screening practice",
        "text": "Solvent substitution should consider hazard, regulatory status, biodegradability, boiling point, recovery, reaction compatibility, and process performance.",
    },
    {
        "source": "E-Factor process metric",
        "text": "E-Factor is the mass ratio of waste to desired product. Lower E-Factors indicate more material-efficient processes.",
    },
    {
        "source": "Atom economy process metric",
        "text": "Atom economy estimates how much reactant mass becomes desired product. Higher atom economy generally indicates a more resource-efficient synthesis.",
    },
    {
        "source": "CHEM21 and GSK solvent guide principles",
        "text": "CHEM21 and GSK solvent guides classify solvents by safety, health, environmental impact, and regulatory concerns.",
    },
    {
        "source": "Bio-derived solvent screening",
        "text": "Bio-derived solvents such as ethyl lactate and glycerol can be attractive when they preserve reaction performance and simplify workup.",
    },
]


def _simple_retrieve(query: str, chunks: list[dict], k: int = 3) -> tuple[list[str], list[str]]:
    terms = {token.lower() for token in query.replace("-", " ").split() if len(token) > 3}
    scored = []
    for chunk in chunks:
        score = sum(1 for term in terms if term in chunk["text"].lower())
        scored.append((score, chunk))
    selected = [chunk for _, chunk in sorted(scored, reverse=True)[:k]]
    return [chunk["text"] for chunk in selected], [chunk["source"] for chunk in selected]


def retrieve_scientific_context(query: str, k: int = 3) -> tuple[list[str], list[str]]:
    """RAG system 1: retrieve scientific reference chunks for explanation."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_or_create_collection("green_chemistry_scientific_refs_v2")
        if collection.count() == 0:
            collection.add(
                documents=[chunk["text"] for chunk in SCIENTIFIC_CHUNKS],
                ids=[f"science-{idx}" for idx in range(len(SCIENTIFIC_CHUNKS))],
                metadatas=[{"source": chunk["source"]} for chunk in SCIENTIFIC_CHUNKS],
            )
        result = collection.query(query_texts=[query], n_results=k)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        sources = [meta.get("source", "Built-in reference") for meta in metas]
        return docs, sources
    except Exception:
        return _simple_retrieve(query, SCIENTIFIC_CHUNKS, k=k)


def retrieve_context(query: str, k: int = 3) -> tuple[list[str], list[str]]:
    """Backward-compatible alias for the scientific RAG system."""
    return retrieve_scientific_context(query, k=k)


def _feedback_documents() -> list[dict]:
    feedback = load_feedback()
    docs: list[dict] = []
    if feedback.empty:
        return docs
    for idx, row in feedback.iterrows():
        comment = str(row.get("comment", "")).strip()
        text = (
            f"Expert feedback: {row['decision']} recommendation {row['recommended_solvent']} "
            f"for {row['reaction_type']} when replacing {row['current_solvent']}. "
            f"Ranking adjustment {row['weight_delta']}. Comment: {comment or 'No comment'}."
        )
        docs.append(
            {
                "id": f"expert-{idx}",
                "source": f"Expert validation {row['timestamp']}",
                "text": text,
            }
        )
    return docs


def retrieve_expert_memory_context(
    reaction_type: str,
    current_solvent: str,
    recommended_solvent: str | None = None,
    k: int = 3,
) -> tuple[list[str], list[str]]:
    """RAG system 2: retrieve prior human validation memory for the next prediction."""
    query = f"{reaction_type} {current_solvent} {recommended_solvent or ''}".strip()
    docs = _feedback_documents()
    if not docs:
        return [], []

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(EXPERT_CHROMA_PATH))
        collection = client.get_or_create_collection("expert_feedback_memory")
        existing = collection.count()
        if existing < len(docs):
            collection.upsert(
                documents=[doc["text"] for doc in docs],
                ids=[doc["id"] for doc in docs],
                metadatas=[{"source": doc["source"]} for doc in docs],
            )
        result = collection.query(query_texts=[query], n_results=min(k, len(docs)))
        retrieved_docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        sources = [meta.get("source", "Expert feedback memory") for meta in metas]
        return retrieved_docs, sources
    except Exception:
        matches = expert_memory_matches(reaction_type, current_solvent, recommended_solvent, limit=k)
        fallback_docs = [
            (
                f"Expert feedback: {match['decision']} {match['recommended_solvent']} for "
                f"{match['reaction_type']} replacing {match['current_solvent']}. "
                f"Adjustment {match['weight_delta']}. Comment: {match['comment'] or 'No comment'}."
            )
            for match in matches
        ]
        fallback_sources = [f"Expert validation {match['timestamp']}" for match in matches]
        return fallback_docs, fallback_sources
