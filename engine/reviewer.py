from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List
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


def compute_similarity_signals(p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, float]:
    """Compute asymmetric similarity signals between needs and personality.

    Signals:
    - needs1_vs_personality2
    - needs2_vs_personality1
    - aggregate (mean of the two)

    If the embedding model is unavailable, returns zeros.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model_path = Path.cwd() / 'models' / 'all-MiniLM-L6-v2'
        if model_path.exists():
            model = SentenceTransformer(str(model_path))
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        n1 = (p1.get('needs') or '').strip()
        s1 = (p1.get('personality') or '').strip()
        n2 = (p2.get('needs') or '').strip()
        s2 = (p2.get('personality') or '').strip()

        texts = [n1, s1, n2, s2]
        emb = model.encode(texts, normalize_embeddings=True)
        import numpy as np
        # Cross signals
        sim_n1_s2 = float(np.dot(emb[0], emb[3])) if len(emb) >= 4 else 0.0
        sim_n2_s1 = float(np.dot(emb[2], emb[1])) if len(emb) >= 4 else 0.0

        # Clamp
        sim_n1_s2 = max(0.0, min(1.0, sim_n1_s2))
        sim_n2_s1 = max(0.0, min(1.0, sim_n2_s1))
        agg = (sim_n1_s2 + sim_n2_s1) / 2.0

        return {
            'needs1_vs_personality2': sim_n1_s2,
            'needs2_vs_personality1': sim_n2_s1,
            'aggregate': agg,
        }
    except Exception as e:
        logger.warning(f"SentenceTransformer unavailable; similarity signals defaulting to 0.0 ({e})")
        return {
            'needs1_vs_personality2': 0.0,
            'needs2_vs_personality1': 0.0,
            'aggregate': 0.0,
        }


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
    outcome: str | None = None,
) -> Dict[str, Any]:
    if openai_chat is None:
        raise RuntimeError("OPENAI_API_KEY not set; cannot run reviewer")

    # Similarity via ST (asymmetric signals)
    logger.info("similarity:start | computing cross needs/personality signals")
    sim_signals = compute_similarity_signals(persona_1, persona_2)
    sim = float(sim_signals.get('aggregate', 0.0))
    logger.info(f"similarity:done | agg={sim:.3f} n1~s2={sim_signals.get('needs1_vs_personality2',0.0):.3f} n2~s1={sim_signals.get('needs2_vs_personality1',0.0):.3f}")

    # LLM decision
    sys = SystemMessage(content=_REVIEW_PROMPT)
    payload = {
        "persona_1": {"id": persona_1.get("id"), "needs": persona_1.get("needs"), "personality": persona_1.get("personality")},
        "persona_2": {"id": persona_2.get("id"), "needs": persona_2.get("needs"), "personality": persona_2.get("personality")},
        "similarity_signals": sim_signals,
        "conversation_outcome": outcome,
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

    # Normalize decision based on similarity signals to avoid contradictions
    def _to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    sim_agg = float(sim_signals.get('aggregate', 0.0))
    low_thr = 0.35
    mid_thr = 0.5

    # Heuristic detection of concrete next steps agreed by both
    def has_concrete_next_step(conv: List[Dict[str, Any]]) -> bool:
        if not conv:
            return False
        propose_terms = [
            'schedule', 'book', 'set up', 'arrange', 'meet', 'call', 'demo', 'pilot', 'poc', 'trial',
            'calendar', 'invite', 'intro', 'follow up', 'send', 'share', 'deck', 'proposal', 'contract'
        ]
        ack_terms = [
            "let's", "lets", 'i will', "i'll", 'we will', 'confirm', 'confirmed', 'works', 'sounds good',
            'ok', 'okay', 'great', 'looking forward', 'see you'
        ]
        # Track who proposed and whether the other acknowledged afterwards
        last_proposer = None
        for turn in conv:
            msg = (turn.get('message') or '').lower()
            spk = turn.get('speaker')
            if any(t in msg for t in propose_terms):
                last_proposer = spk
                continue
            if last_proposer and spk and spk != last_proposer:
                if any(t in msg for t in ack_terms):
                    return True
        return False

    if not isinstance(decision, dict):
        decision = {"decision": "more_info", "rationale": "Normalization: invalid reviewer output.", "confidence": 0.3}

    # Ensure required keys
    decision.setdefault('decision', 'more_info')
    decision.setdefault('rationale', '')
    decision.setdefault('confidence', 0.5)

    # Apply gating/capping rules
    try:
        conf = _to_float(decision.get('confidence', 0.5), 0.5)
        dec = str(decision.get('decision', 'more_info'))
        rat = str(decision.get('rationale', ''))

        # Outcome-aware gating
        o = (outcome or '').lower().strip()
        if o in ("not_a_fit",):
            dec = 'not_a_fit'
            conf = min(conf, 0.4)
            rat = (rat + "\nNote: Conversation outcome indicates not_a_fit; normalized decision.").strip()
        elif o in ("needs_more_info", "follow_up_later"):
            agreed_next = has_concrete_next_step(transcript)
            if dec == 'proceed' and not agreed_next:
                dec = 'more_info'
                conf = min(conf, 0.55)
                rat = (rat + "\nNote: Outcome suggests more info/follow-up; downgraded from proceed.").strip()
            else:
                conf = min(conf, 0.65)

        # Similarity-aware gating
        if sim_agg < low_thr:
            agreed_next = has_concrete_next_step(transcript)
            if dec == 'proceed' and not agreed_next:
                dec = 'more_info'
                conf = min(conf, 0.45)
                rat = (rat + "\nNote: Low similarity; proceeding gated without explicit mutually agreed next step.").strip()
            elif dec == 'proceed' and agreed_next:
                conf = min(conf, 0.6)
                rat = (rat + "\nNote: Low similarity; allowing proceed due to explicit next step, confidence capped.").strip()
            else:
                conf = min(conf, 0.55)
                rat = (rat + "\nNote: Low similarity; confidence capped.").strip()
        elif sim_agg < mid_thr:
            conf = min(conf, 0.75)
        # write back
        decision['decision'] = dec
        decision['confidence'] = round(conf, 2)
        decision['rationale'] = rat
    except Exception as e:
        logger.warning(f"decision-normalization: failed to normalize due to {e}")

    return {
        "similarity_score": sim,
        "similarity_signals": sim_signals,
        "chat_decision": decision,
        "chat": transcript,
    }


def review_conversation_sync(
    persona_1: Dict[str, Any],
    persona_2: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    outcome: str | None = None,
) -> Dict[str, Any]:
    """Synchronous variant of review_conversation using blocking LLM calls.

    Intended for environments where calling asyncio.run is problematic (e.g.,
    hosting platforms that already run an event loop).
    """
    if openai_chat is None:
        raise RuntimeError("OPENAI_API_KEY not set; cannot run reviewer")

    # Similarity via ST (asymmetric signals)
    logger.info("similarity:start | computing cross needs/personality signals")
    sim_signals = compute_similarity_signals(persona_1, persona_2)
    sim = float(sim_signals.get('aggregate', 0.0))
    logger.info(
        f"similarity:done | agg={sim:.3f} n1~s2={sim_signals.get('needs1_vs_personality2',0.0):.3f} "
        f"n2~s1={sim_signals.get('needs2_vs_personality1',0.0):.3f}"
    )

    # LLM decision (sync)
    sys = SystemMessage(content=_REVIEW_PROMPT)
    payload = {
        "persona_1": {"id": persona_1.get("id"), "needs": persona_1.get("needs"), "personality": persona_1.get("personality")},
        "persona_2": {"id": persona_2.get("id"), "needs": persona_2.get("needs"), "personality": persona_2.get("personality")},
        "similarity_signals": sim_signals,
        "conversation_outcome": outcome,
        "conversation": transcript,
    }
    usr = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    logger.info("decision:start | invoking LLM reviewer (sync)")
    res = openai_chat.invoke([sys, usr])
    logger.info("decision:done | received response (sync)")
    txt = getattr(res, 'content', None) or "{}"
    try:
        decision = json.loads(txt)
    except Exception:
        # Try fence strip
        s = txt.strip()
        if s.startswith('```'):
            lines = [ln for ln in s.splitlines() if not ln.strip().startswith('```')]
            s = "\n".join(lines)
        decision = json.loads(s)

    # Normalize decision based on similarity signals to avoid contradictions
    def _to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    sim_agg = float(sim_signals.get('aggregate', 0.0))
    low_thr = 0.35
    mid_thr = 0.5

    def has_concrete_next_step(conv: List[Dict[str, Any]]) -> bool:
        if not conv:
            return False
        propose_terms = [
            'schedule', 'book', 'set up', 'arrange', 'meet', 'call', 'demo', 'pilot', 'poc', 'trial',
            'calendar', 'invite', 'intro', 'follow up', 'send', 'share', 'deck', 'proposal', 'contract'
        ]
        ack_terms = [
            "let's", "lets", 'i will', "i'll", 'we will', 'confirm', 'confirmed', 'works', 'sounds good',
            'ok', 'okay', 'great', 'looking forward', 'see you'
        ]
        last_proposer = None
        for turn in conv:
            msg = (turn.get('message') or '').lower()
            spk = turn.get('speaker')
            if any(t in msg for t in propose_terms):
                last_proposer = spk
                continue
            if last_proposer and spk and spk != last_proposer:
                if any(t in msg for t in ack_terms):
                    return True
        return False

    if not isinstance(decision, dict):
        decision = {"decision": "more_info", "rationale": "Normalization: invalid reviewer output.", "confidence": 0.3}

    decision.setdefault('decision', 'more_info')
    decision.setdefault('rationale', '')
    decision.setdefault('confidence', 0.5)

    try:
        conf = _to_float(decision.get('confidence', 0.5), 0.5)
        dec = str(decision.get('decision', 'more_info'))
        rat = str(decision.get('rationale', ''))

        o = (outcome or '').lower().strip()
        if o in ("not_a_fit",):
            dec = 'not_a_fit'
            conf = min(conf, 0.4)
            rat = (rat + "\nNote: Conversation outcome indicates not_a_fit; normalized decision.").strip()
        elif o in ("needs_more_info", "follow_up_later"):
            agreed_next = has_concrete_next_step(transcript)
            if dec == 'proceed' and not agreed_next:
                dec = 'more_info'
                conf = min(conf, 0.55)
                rat = (rat + "\nNote: Outcome suggests more info/follow-up; downgraded from proceed.").strip()
            else:
                conf = min(conf, 0.65)

        if sim_agg < low_thr:
            agreed_next = has_concrete_next_step(transcript)
            if dec == 'proceed' and not agreed_next:
                dec = 'more_info'
                conf = min(conf, 0.45)
                rat = (rat + "\nNote: Low similarity; proceeding gated without explicit mutually agreed next step.").strip()
            elif dec == 'proceed' and agreed_next:
                conf = min(conf, 0.6)
                rat = (rat + "\nNote: Low similarity; allowing proceed due to explicit next step, confidence capped.").strip()
            else:
                conf = min(conf, 0.55)
                rat = (rat + "\nNote: Low similarity; confidence capped.").strip()
        elif sim_agg < mid_thr:
            conf = min(conf, 0.75)
        decision['decision'] = dec
        decision['confidence'] = round(conf, 2)
        decision['rationale'] = rat
    except Exception as e:
        logger.warning(f"decision-normalization: failed to normalize due to {e}")

    return {
        "similarity_score": sim,
        "similarity_signals": sim_signals,
        "chat_decision": decision,
        "chat": transcript,
    }
