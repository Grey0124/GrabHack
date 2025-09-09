"""Long-term memory (LTM) using Ollama embeddings + ChromaDB.

Provides simple `embed`, `remember`, and `recall` helpers that you can
call before/after agent runs to persist and retrieve relevant incident
summaries.

Environment variables:
- OLLAMA_API_BASE: if set (e.g., http://localhost:11434/v1), used to derive
  the native base for embeddings (strip trailing /v1). Fallback EMBED_BASE
  defaults to http://localhost:11434.
- EMBED_MODEL: override the embedding model name (default: "nomic-embed-text").
- VECTOR_DIR: persistent vector store directory (default: .vectorstore)
"""

from __future__ import annotations

import json
import os
import hashlib
from typing import List

import httpx
from chromadb import PersistentClient


def _native_base_from_env(default_native: str = "http://localhost:11434") -> str:
    base = os.getenv("OLLAMA_API_BASE", "").rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return os.getenv("EMBED_BASE", default_native)


EMBED_BASE = _native_base_from_env()
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
VECTOR_DIR = os.getenv("VECTOR_DIR", ".vectorstore")

os.makedirs(VECTOR_DIR, exist_ok=True)

chroma = PersistentClient(path=VECTOR_DIR)
coll = chroma.get_or_create_collection("synapse_incidents")


def embed(text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "input": text}
    # Use a modest timeout; embeddings are fast
    with httpx.Client(base_url=EMBED_BASE, timeout=httpx.Timeout(30.0)) as client:
        r = client.post("/api/embeddings", json=payload)
        r.raise_for_status()
        data = r.json()
        # Ollama native embeddings return {"embedding": [...]} for a single input
        if "embedding" in data:
            return data["embedding"]
        # Some implementations may return OpenAI-like {"data":[{"embedding": [...]}]}
        if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
            return data["data"][0].get("embedding", [])
        raise ValueError("Unexpected embeddings response structure from Ollama")


def _stable_id(text: str) -> str:
    # Stable hex ID to allow upserts/de-dup
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def remember(run_dict: dict) -> None:
    """Store a compact summary of a run into vector memory.

    Expected keys in run_dict: "goal" (str), "scratchpad" (list).
    """
    goal = run_dict.get("goal", "")
    scratch = run_dict.get("scratchpad") or []
    key_steps = scratch[-3:] if scratch else []
    resolution = scratch[-1] if scratch else {}
    summary_obj = {
        "problem": goal,
        "key_steps": key_steps,
        "resolution": resolution,
    }
    # Deterministic JSON to keep IDs stable for identical summaries
    summary = json.dumps(summary_obj, ensure_ascii=False, sort_keys=True)
    eid = _stable_id(summary)
    vec = embed(summary)
    # Prefer upsert if available; fallback to add with duplicate guard
    if hasattr(coll, "upsert"):
        coll.upsert(ids=[eid], embeddings=[vec], documents=[summary])
    else:
        try:
            coll.add(ids=[eid], embeddings=[vec], documents=[summary])
        except Exception:
            # Duplicate IDs or other benign conflicts can be ignored for now
            pass


def recall(goal: str, k: int = 3) -> List[str]:
    """Retrieve up to k most similar past summaries for a new goal."""
    qv = embed(goal)
    res = coll.query(query_embeddings=[qv], n_results=int(max(1, k)))
    docs = res.get("documents") or []
    if not docs:
        return []
    # Chroma returns a list of lists
    return docs[0] if isinstance(docs[0], list) else docs


__all__ = ["embed", "remember", "recall", "EMBED_BASE", "EMBED_MODEL", "VECTOR_DIR"]

