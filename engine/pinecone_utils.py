from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from loguru import logger


def get_pinecone_client():
    try:
        from pinecone import Pinecone
    except Exception as e:
        raise RuntimeError(
            "pinecone-client is not installed. Please `pip install pinecone-client>=5`."
        ) from e
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment")
    return Pinecone(api_key=api_key)


def ensure_index(index_name: str, dimension: int, metric: str = "cosine") -> None:
    """Create the Pinecone index if missing (serverless)."""
    pc = get_pinecone_client()
    # Be tolerant of different client return shapes
    try:
        existing = {idx.name for idx in pc.list_indexes()}  # v5 IndexModel objects
    except Exception:
        try:
            existing = {idx["name"] for idx in pc.list_indexes()}  # dict-like
        except Exception:
            try:
                existing = set(pc.list_indexes())  # list[str]
            except Exception:
                existing = set()
    if index_name in existing:
        logger.info("Pinecone index exists: {}", index_name)
        return
    try:
        from pinecone import ServerlessSpec
    except Exception:
        ServerlessSpec = None
    logger.info("Creating Pinecone index: {} (dim={}, metric={})", index_name, dimension, metric)
    if ServerlessSpec is None:
        # Older client fallback (unlikely with v5+ requirement)
        pc.create_index(name=index_name, dimension=dimension, metric=metric)
    else:
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )


def get_index(index_name: str):
    pc = get_pinecone_client()
    return pc.Index(index_name)


def upsert_personas(
    index_name: str,
    items: List[Dict[str, Any]],
    namespace: Optional[str] = None,
) -> None:
    """Upsert persona JSON objects with embeddings into Pinecone.

    Each item must include: id (str), values (List[float]), metadata (dict)
    """
    index = get_index(index_name)
    logger.info("Upserting {} vectors into index={} ns={}...", len(items), index_name, namespace)
    # Pinecone upsert; prefer v5 signature, fallback to older
    try:
        if namespace:
            index.upsert(vectors=items, namespace=namespace)
        else:
            index.upsert(vectors=items)
    except TypeError:
        # Older clients may use 'records' or differ; try alternative kw
        if namespace:
            index.upsert(records=items, namespace=namespace)
        else:
            index.upsert(records=items)
    logger.info("Upsert complete")


def query_top_k(
    index_name: str,
    vector: List[float],
    top_k: int = 5,
    namespace: Optional[str] = None,
    include_metadata: bool = True,
    filter: Optional[dict] = None,
):
    index = get_index(index_name)
    kwargs = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": include_metadata,
    }
    if namespace:
        kwargs["namespace"] = namespace
    if filter:
        # Different client versions: use 'filter' or 'where'
        try:
            kwargs["filter"] = filter
        except Exception:
            kwargs["where"] = filter
    return index.query(**kwargs)
