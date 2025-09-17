from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import os

from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from .llm import openai_chat, get_openai_chat
import os


_DEFAULT_PROMPT = (
    "You are a professional conversation agent. Use the provided PROFILE_CONTEXT, which"
    " includes both your principal's profile and the counterpart's profile, to conduct"
    " a concise, outcome-oriented conversation. Ask one focused question per turn and"
    " propose clear next steps when appropriate."
)


def _load_system_prompt() -> str:
    # Allow override via PROMPTS_DIR; else use local prompts/ai_agent_prompt.md
    try:
        base_dir = os.getenv("PROMPTS_DIR")
        if base_dir:
            path = Path(base_dir) / "ai_agent_prompt.md"
        else:
            path = Path(__file__).resolve().parents[1] / "prompts" / "ai_agent_prompt.md"
        text = path.read_text(encoding="utf-8")
        return text
    except Exception as e:
        logger.warning(f"Falling back to default system prompt: {e}")
        return _DEFAULT_PROMPT


_SYSTEM_PROMPT = _load_system_prompt()


class PersonaAgent:
    def __init__(
        self,
        role: str,
        agent_id: str,
        needs: str,
        personality: str,
        counterpart_id: Optional[str] = None,
        counterpart_needs: Optional[str] = None,
        counterpart_personality: Optional[str] = None,
    ) -> None:
        self.role = role
        self.id = agent_id or ""
        self.needs = needs or ""
        self.personality = personality or ""
        self.counterpart_id = counterpart_id or ""
        self.counterpart_needs = counterpart_needs or ""
        self.counterpart_personality = counterpart_personality or ""
        self.llm = openai_chat
        if self.llm is None:
            raise RuntimeError(
                "openai_chat not initialized; set OPENAI_API_KEY (and optionally OPENAI_MODEL)"
            )
        self.system_prompt = _SYSTEM_PROMPT

    def build_system(self) -> SystemMessage:
        return SystemMessage(content=self.system_prompt)

    def build_context(self) -> str:
        import json
        return json.dumps(
            {
                "agent_role": self.role,
                "agent": {
                    "id": self.id,
                    "needs": self.needs,
                    "personality": self.personality,
                },
                "counterpart": {
                    "id": self.counterpart_id,
                    "needs": self.counterpart_needs,
                    "personality": self.counterpart_personality,
                },
            },
            ensure_ascii=False,
        )

    async def respond(self, conversation_history: List[Dict[str, str]]) -> str:
        import time
        # Build a single Human message per turn (normal, robust style)
        turn_index = len(conversation_history) + 1
        blocks = [
            f"PROFILE_CONTEXT:\n{self.build_context()}",
            (
                "STYLE_CONSTRAINTS: 50–80 words, 2–3 sentences, business‑formal. "
                "Reference at least one concrete detail from counterpart.needs or counterpart.personality, "
                "and link it to the agent’s needs or personality. One question max (omit in closing). "
                "Plain sentences only; no lists; avoid greetings after the first turn; avoid generic praise. "
                "Do not invent facts—use only PROFILE_CONTEXT and LAST_MESSAGE; if unknown, say so briefly and ask one clarifying question. "
                "Do not claim commitments (meetings/materials) unless explicitly confirmed."
            ),
            f"TURN_INDEX: {turn_index}",
        ]
        if not conversation_history:
            blocks.append(
                "OPENING_INSTRUCTION: Start with a warm, professional greeting and a concise "
                "self-introduction (1–2 sentences) grounded in PROFILE_CONTEXT. Then ask exactly "
                "one focused question to understand the counterpart’s current top priority."
            )
        else:
            last = conversation_history[-1]
            blocks.append(f"LAST_MESSAGE:\n{last.get('message','')}")
            try:
                last_state = (conversation_history[-1] or {}).get('state')
            except Exception:
                last_state = None
            if last_state == 'closing':
                blocks.append(
                    "CLOSING_INSTRUCTION: Confirm agreed next steps in 1–2 sentences and, if needed, ask one short "
                    "logistics question (timing/link/attachments). If everything is confirmed, end politely."
                )
        blocks.append("REPLY: Provide the response now. Do not leave this blank.")
        user_content = "\n\n".join(blocks)
        messages = [self.build_system(), HumanMessage(content=user_content)]
        t0 = time.perf_counter()
        result = await self.llm.ainvoke(messages)
        dt = time.perf_counter() - t0
        text = (result.content or "").strip()
        logger.info(f"llm_call | role={self.role} id={self.id} dt={dt:.2f}s")
        if not text:
            # Retry once with a concise nudge using the same single-message format
            nudge_blocks = [
                f"PROFILE_CONTEXT:\n{self.build_context()}",
                "Your previous response was empty. Provide a concise reply per STYLE_CONSTRAINTS (<=120 words, 2–4 sentences, one question max).",
            ]
            if conversation_history:
                last = conversation_history[-1]
                nudge_blocks.append(f"LAST_MESSAGE:\n{last.get('message','')}")
            nudge_blocks.append("REPLY: Provide the response now. Do not leave this blank.")
            retry_msgs = [self.build_system(), HumanMessage(content="\n\n".join(nudge_blocks))]
            r0 = time.perf_counter()
            result = await self.llm.ainvoke(retry_msgs)
            text = (result.content or "").strip()
            rdt = time.perf_counter() - r0
            logger.info(f"llm_retry | role={self.role} id={self.id} dt={rdt:.2f}s")
        if not text:
            # Final fallback: try a different model (default gpt-4o-mini) for this turn only
            if os.getenv("CHAT_FALLBACK_ENABLED", "false").lower() in ("1", "true", "yes"):
                fb_model = os.getenv("OPENAI_CHAT_FALLBACK_MODEL", "gpt-4o-mini")
                fb_llm = get_openai_chat(model=fb_model, temperature=None)
                if fb_llm is not None:
                    logger.warning(f"llm_fallback | switching model for turn | model={fb_model}")
                    try:
                        fbr = await fb_llm.ainvoke(messages)
                        text = (fbr.content or "").strip()
                    except Exception as e:
                        logger.error(f"llm_fallback_failed | {e}")
        return text
