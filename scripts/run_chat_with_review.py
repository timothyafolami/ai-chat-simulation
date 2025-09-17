from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from loguru import logger
import sys, time
from pathlib import Path as _PathForSys

# Ensure the ai-chat-simulation package root is on sys.path when run from scripts/
_pkg_root = str(_PathForSys(__file__).resolve().parents[1])
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from engine.agents import PersonaAgent
from engine.manager import ConversationManager
from engine.reviewer import review_conversation


# ==========================
# Configuration (edit here)
# ==========================
# Resolve project root from this script's location
ROOT = _PathForSys(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "generated_personas"

# Choose two persona JSON filenames in GENERATED_DIR (without path)
PERSONA_1_FILE = "001__Ada_Adekunle.json"
PERSONA_2_FILE = "001__Adewale_Afolabi.json"

# Conversation cap (total messages)
MAX_MESSAGES = 10

# Output result path
RESULTS_DIR = ROOT / "chat_results"
RESULT_FILE = RESULTS_DIR / f"{PERSONA_1_FILE.replace('.json','')}__vs__{PERSONA_2_FILE.replace('.json','')}.json"


def load_persona(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error(f"Persona file not found: {path}")
        raise


async def main():
    # Configure verbose logging to stdout
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", colorize=True, format="{time:HH:mm:ss} | {level} | {message}")

    logger.info(f"Loading personas from {GENERATED_DIR}")
    logger.info(f"Persona 1: {PERSONA_1_FILE}")
    logger.info(f"Persona 2: {PERSONA_2_FILE}")
    logger.info(f"Max messages: {MAX_MESSAGES}")

    p1 = load_persona(GENERATED_DIR / PERSONA_1_FILE)
    p2 = load_persona(GENERATED_DIR / PERSONA_2_FILE)

    # Build agents from id/needs/personality
    a1 = PersonaAgent(
        role="profile_1",
        agent_id=p1.get("id", ""),
        needs=p1.get("needs", ""),
        personality=p1.get("personality", ""),
        counterpart_id=p2.get("id", ""),
        counterpart_needs=p2.get("needs", ""),
        counterpart_personality=p2.get("personality", ""),
    )
    a2 = PersonaAgent(
        role="profile_2",
        agent_id=p2.get("id", ""),
        needs=p2.get("needs", ""),
        personality=p2.get("personality", ""),
        counterpart_id=p1.get("id", ""),
        counterpart_needs=p1.get("needs", ""),
        counterpart_personality=p1.get("personality", ""),
    )

    manager = ConversationManager(participant_2=a2, participant_1=a1, max_turns=MAX_MESSAGES)
    logger.info("Starting chat simulation ...")
    t0 = time.perf_counter()
    convo = await manager.run()
    t1 = time.perf_counter()
    logger.info(f"Chat completed in {t1 - t0:.2f}s with {len(convo.get('conversation', []))} messages")

    logger.info("Running post-chat review (similarity + decision) ...")
    t2 = time.perf_counter()
    review = await review_conversation(p1, p2, convo.get("conversation", []))
    t3 = time.perf_counter()
    logger.info(f"Review completed in {t3 - t2:.2f}s | similarity={review.get('similarity_score'):.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Wrote review to {RESULT_FILE}")

    # Print decision summary to console for quick visibility
    try:
        decision = review.get("chat_decision", {})
        logger.info(
            f"Decision: {decision.get('decision','?')} | confidence={decision.get('confidence','?')} | "
            f"similarity={review.get('similarity_score'):.3f}"
        )
        logger.info(f"Rationale: {decision.get('rationale','').strip()}")
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
