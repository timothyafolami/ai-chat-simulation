from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path

from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from .llm import openai_chat


def cosine_similarity(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def persona_text(obj: Dict[str, Any]) -> str:
    return f"NEEDS: {obj.get('needs','')}\nPERSONALITY: {obj.get('personality','')}"


def compute_similarity(p1: Dict[str, Any], p2: Dict[str, Any]) -> float:
    """Compute semantic similarity using sentence-transformers if available.

    Fallback: naive 0.0 if model not available.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model_name = (Path.cwd() / 'models' / 'all-MiniLM-L6-v2')
        if model_name.exists():
            model = SentenceTransformer(str(model_name))
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        texts = [persona_text(p1), persona_text(p2)]
        emb = model.encode(texts, normalize_embeddings=True)
        import numpy as np
        sim = float(np.dot(emb[0], emb[1]))
        return max(0.0, min(1.0, sim))
    except Exception as e:
        logger.warning(f"SentenceTransformer unavailable; similarity defaulting to 0.0 ({e})")
        return 0.0


def _load_chat_decision_prompt() -> str:
    p = Path(__file__).resolve().parents[1] / 'prompts' / 'chat_decision_prompt.md'
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        return (
            "You are an impartial reviewer. Read the conversation and return a JSON with keys: "
            "decision ('proceed'|'more_info'|'not_a_fit'), rationale (<=80 words)."
        )


_REVIEW_PROMPT = _load_chat_decision_prompt()


async def review_conversation(
    persona_1: Dict[str, Any],
    persona_2: Dict[str, Any],
    transcript: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if openai_chat is None:
        raise RuntimeError("OPENAI_API_KEY not set; cannot run reviewer")

    # Similarity via ST
    logger.info("similarity:start | computing ST cosine similarity")
    sim = compute_similarity(persona_1, persona_2)
    logger.info(f"similarity:done | score={sim:.3f}")

    # LLM decision
    sys = SystemMessage(content=_REVIEW_PROMPT)
    payload = {
        "persona_1": {"id": persona_1.get("id"), "needs": persona_1.get("needs"), "personality": persona_1.get("personality")},
        "persona_2": {"id": persona_2.get("id"), "needs": persona_2.get("needs"), "personality": persona_2.get("personality")},
        "conversation": transcript,
    }
    usr = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    logger.info("decision:start | invoking LLM reviewer")
    res = await openai_chat.ainvoke([sys, usr])
    logger.info("decision:done | received response")
    txt = res.content or "{}"
    try:
        decision = json.loads(txt)
    except Exception:
        # Try fence strip
        s = txt.strip()
        if s.startswith('```'):
            lines = [ln for ln in s.splitlines() if not ln.strip().startswith('```')]
            s = "\n".join(lines)
        decision = json.loads(s)

    return {
        "similarity_score": sim,
        "chat_decision": decision,
        "chat": transcript,
    }
