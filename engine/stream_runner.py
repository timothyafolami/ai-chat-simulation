from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, Any, Generator

from loguru import logger

from .agents import PersonaAgent
from .states import ConversationState, ConversationOutcome, ConversationMetrics


def _analyze(text: str, role: str, metrics: ConversationMetrics) -> None:
    low = (text or "").lower()
    pos = ["interesting", "impressive", "exciting", "great", "next steps"]
    neg = ["concern", "worried", "risky", "challenging", "difficult"]
    conf = ["validated", "growing", "traction", "customers"]

    if role == 'profile_2':
        for s in pos:
            if s in low:
                metrics.profile2_engagement = min(1.0, metrics.profile2_engagement + 0.05)
                metrics.positive_signals.append(s)
        for s in neg:
            if s in low:
                metrics.profile2_engagement = max(0.0, metrics.profile2_engagement - 0.03)
                metrics.red_flags.append(s)
    else:
        for s in conf:
            if s in low:
                metrics.profile1_engagement = min(1.0, metrics.profile1_engagement + 0.03)


def _outcome(metrics: ConversationMetrics) -> ConversationOutcome:
    iv = metrics.profile2_engagement
    fc = metrics.profile1_engagement
    if iv > 0.7 and fc > 0.7:
        return ConversationOutcome.MUTUAL_INTEREST
    if iv > 0.7:
        return ConversationOutcome.INTERESTED_NEXT_STEPS
    if iv > 0.5:
        return ConversationOutcome.NEEDS_MORE_INFO
    if iv > 0.3:
        return ConversationOutcome.FOLLOW_UP_LATER
    return ConversationOutcome.NOT_A_FIT


def run_chat_stream(
    participant_1: PersonaAgent,
    participant_2: PersonaAgent,
    max_turns: int = 10,
    min_turns: int = 6,
    start_with: str = "profile_1",
) -> Generator[Dict[str, Any], None, None]:
    """Synchronous streaming runner for UI. Yields events as the chat progresses.

    Yields dicts of shape:
      - {type: 'start', data: {...}}
      - {type: 'turn', data: {speaker, message, state, timestamp}}
      - {type: 'end', data: {outcome, final_metrics, conversation}}
    """
    state = ConversationState.INTRODUCTION
    metrics = ConversationMetrics()
    log: list[dict[str, Any]] = []
    current = start_with if start_with in ("profile_1", "profile_2") else "profile_1"
    closing_grace = 2
    hard_max_turns = max_turns + 2 * closing_grace
    closing_turns = 0

    p2_name = getattr(participant_2, 'id', '')
    p1_name = getattr(participant_1, 'id', '')
    logger.info(f"ui_chat_start | profile_2={p2_name} | profile_1={p1_name} | max_turns={max_turns}")
    yield {"type": "start", "data": {"profile_2": p2_name, "profile_1": p1_name, "max_turns": max_turns, "start_with": current}}

    def turns_in_state(window: int = 10) -> int:
        return sum(1 for e in log[-window:] if e.get('state') == state.value)

    def should_transition() -> bool:
        target = {
            ConversationState.INTRODUCTION: 2,
            ConversationState.DISCOVERY: 3,
            ConversationState.DEEP_DIVE: 3,
            ConversationState.CHALLENGES: 2,
            ConversationState.CLOSING: closing_grace,
        }
        return turns_in_state() >= target.get(state, 3)

    while True:
        metrics.turn_count += 1

        if should_transition():
            nxt = {
                ConversationState.INTRODUCTION: ConversationState.DISCOVERY,
                ConversationState.DISCOVERY: ConversationState.DEEP_DIVE,
                ConversationState.DEEP_DIVE: ConversationState.CHALLENGES,
                ConversationState.CHALLENGES: ConversationState.CLOSING,
                ConversationState.CLOSING: ConversationState.ENDED,
            }
            state = nxt.get(state, ConversationState.ENDED)
            if state == ConversationState.CLOSING:
                closing_turns = 0
            logger.info(f"ui_chat_state_transition | state={state.value}")

        # End conditions (mirror manager)
        if metrics.turn_count >= max_turns and state != ConversationState.CLOSING:
            state = ConversationState.CLOSING
            logger.info("ui_chat_force_closing | reached max_turns; grace window")

        # Produce a turn synchronously (await inside)
        if current == 'profile_2':
            text = asyncio.run(participant_2.respond(log))
            _analyze(text, 'profile_2', metrics)
            log.append({'speaker': 'profile_2', 'message': text, 'state': state.value, 'timestamp': datetime.utcnow().isoformat() + 'Z'})
            yield {"type": "turn", "data": log[-1]}
            if state == ConversationState.CLOSING:
                closing_turns += 1
                if closing_turns >= 2:
                    oc = _outcome(metrics)
                    result = {
                        'outcome': oc.value,
                        'final_metrics': {
                            'profile2_engagement': metrics.profile2_engagement,
                            'profile1_engagement': metrics.profile1_engagement,
                        },
                        'conversation': log,
                    }
                    yield {"type": "end", "data": result}
                    return
            current = 'profile_1'
        else:
            text = asyncio.run(participant_1.respond(log))
            _analyze(text, 'profile_1', metrics)
            log.append({'speaker': 'profile_1', 'message': text, 'state': state.value, 'timestamp': datetime.utcnow().isoformat() + 'Z'})
            yield {"type": "turn", "data": log[-1]}
            if state == ConversationState.CLOSING:
                closing_turns += 1
                if closing_turns >= 2:
                    oc = _outcome(metrics)
                    result = {
                        'outcome': oc.value,
                        'final_metrics': {
                            'profile2_engagement': metrics.profile2_engagement,
                            'profile1_engagement': metrics.profile1_engagement,
                        },
                        'conversation': log,
                    }
                    yield {"type": "end", "data": result}
                    return
            current = 'profile_2'
