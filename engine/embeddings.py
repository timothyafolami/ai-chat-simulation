from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Dict, Any, Tuple

from loguru import logger


@lru_cache(maxsize=1)
def _get_openai_embeddings_client():
    """Return a cached LangChain OpenAIEmbeddings client.

    Env:
      - OPENAI_API_KEY (required)
      - OPENAI_EMBEDDINGS_MODEL (default: text-embedding-3-small)
    """
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception as e:
        raise RuntimeError(
            "langchain-openai not installed; required for OpenAI embeddings"
        ) from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for embeddings")
    model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    logger.debug(f"Initializing OpenAIEmbeddings model={model}")
    return OpenAIEmbeddings(model=model, api_key=api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using OpenAI embeddings."""
    emb = _get_openai_embeddings_client()
    vectors = emb.embed_documents(texts)
    return vectors


def persona_to_text(obj: Dict[str, Any]) -> str:
    needs = (obj or {}).get("needs", "")
    personality = (obj or {}).get("personality", "")
    return f"NEEDS: {needs}\nPERSONALITY: {personality}"


def persona_field_text(obj: Dict[str, Any], field: str) -> str:
    if field == "needs":
        return (obj or {}).get("needs", "")
    if field == "personality":
        return (obj or {}).get("personality", "")
    return persona_to_text(obj)


def embed_persona(obj: Dict[str, Any]) -> List[float]:
    """Embed combined persona text (needs + personality)."""
    txt = persona_to_text(obj)
    return embed_texts([txt])[0]


def embed_persona_fields(obj: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Embed needs and personality separately (double matching)."""
    texts = [persona_field_text(obj, "needs"), persona_field_text(obj, "personality")]
    vecs = embed_texts(texts)
    return vecs[0], vecs[1]


def embedding_dimension() -> int:
    """Return vector dimension for configured embedding model.

    Uses common defaults; if unknown, attempts a one-off embed to infer.
    """
    model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    known = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    dim = known.get(model)
    if dim:
        return dim
    # Fallback: embed a short dummy text to infer dimension
    try:
        v = embed_texts(["dimension probe"])[0]
        return len(v)
    except Exception as e:
        logger.warning(f"Could not infer embedding dimension for {model}: {e}")
        # Reasonable default for OpenAI embeddings
        return 1536
