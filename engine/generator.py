from __future__ import annotations

import json
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from .llm import openai_chat


def _load_prompt() -> str:
    # Always look for local prompts/persona_generation_prompt.md
    p = Path(__file__).resolve().parents[1] / "prompts" / "persona_generation_prompt.md"
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to load persona generation prompt: {e}")
        return (
            "You generate two concise fields (needs, personality) from PROFILE and RESUME. "
            "Each 40-60 words, one paragraph, professional tone. Output JSON with keys: id, needs, personality."
        )


_SYSTEM_PROMPT = _load_prompt()


def _truncate(txt: Optional[str], limit: int) -> str:
    t = (txt or "").strip()
    return t[:limit]


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove code fences and optional json hint
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        return "\n".join(lines).strip()
    return s


def _parse_json(s: str) -> Dict[str, Any]:
    s = _strip_fences(s)
    try:
        return json.loads(s)
    except Exception:
        # try to extract first {...}
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            inner = s[start : end + 1]
            try:
                return json.loads(inner)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON after fence stripping: {e}")
        raise


async def generate_persona(
    entity_id: str,
    profile_text: Optional[str],
    resume_text: Optional[str],
    max_profile_chars: int = 6000,
    max_resume_chars: int = 12000,
) -> Dict[str, Any]:
    if openai_chat is None:
        raise RuntimeError("OPENAI_API_KEY not set; cannot generate personas")

    sys_msg = SystemMessage(content=_SYSTEM_PROMPT)
    content = (
        f"ID: {entity_id}\n\n"
        f"PROFILE:\n{_truncate(profile_text, max_profile_chars)}\n\n"
        f"RESUME:\n{_truncate(resume_text, max_resume_chars)}\n"
    )
    usr_msg = HumanMessage(content=content)

    logger.debug(f"Generating persona | id={entity_id}")
    resp = await openai_chat.ainvoke([sys_msg, usr_msg])
    raw = resp.content or ""
    try:
        obj = _parse_json(raw)
    except Exception as e:
        logger.error(f"Failed to parse JSON for {entity_id}: {e}")
        raise

    # Minimal validation/defaults
    obj.setdefault("id", entity_id)
    for k in ("needs", "personality"):
        if not isinstance(obj.get(k), str):
            obj[k] = ""
    return obj
