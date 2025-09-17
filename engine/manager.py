from __future__ import annotations

from typing import Dict, Any, Tuple
from datetime import datetime

from loguru import logger

from .agents import PersonaAgent
from .states import ConversationState, ConversationOutcome, ConversationMetrics


class ConversationManager:
    def __init__(
        self,
        participant_2: PersonaAgent,
        participant_1: PersonaAgent,
        max_turns: int = 10,
        min_turns: int = 6,
        ensure_completion: bool = True,
        closing_grace: int = 2,
        hard_max_turns: int | None = None,
        start_with: str = "profile_1",
    ) -> None:
        self.participant_1 = participant_1
        self.participant_2 = participant_2
        self.state = ConversationState.INTRODUCTION
        self.metrics = ConversationMetrics()
        self.log: list[dict[str, Any]] = []
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.ensure_completion = ensure_completion
        self.closing_grace = max(1, int(closing_grace))
        self.hard_max_turns = hard_max_turns or (max_turns + 2 * self.closing_grace)
        self._closing_turns = 0
        self.start_with = start_with if start_with in ("profile_1", "profile_2") else "profile_1"

    def _turns_in_state(self, window: int = 10) -> int:
        return sum(1 for e in self.log[-window:] if e.get('state') == self.state.value)

    def should_transition_state(self) -> bool:
        target = {
            ConversationState.INTRODUCTION: 2,
            ConversationState.DISCOVERY: 3,
            ConversationState.DEEP_DIVE: 3,
            ConversationState.CHALLENGES: 2,
            ConversationState.CLOSING: max(1, self.closing_grace),
        }
        return self._turns_in_state() >= target.get(self.state, 3)

    def transition(self) -> None:
        nxt = {
            ConversationState.INTRODUCTION: ConversationState.DISCOVERY,
            ConversationState.DISCOVERY: ConversationState.DEEP_DIVE,
            ConversationState.DEEP_DIVE: ConversationState.CHALLENGES,
            ConversationState.CHALLENGES: ConversationState.CLOSING,
            ConversationState.CLOSING: ConversationState.ENDED,
        }
        self.state = nxt.get(self.state, ConversationState.ENDED)
        if self.state == ConversationState.CLOSING:
            self._closing_turns = 0
        logger.info(f"chat_state_transition | state={self.state.value}")

    def analyze(self, text: str, role: str) -> None:
        low = text.lower()
        pos = ["interesting", "impressive", "exciting", "great", "next steps"]
        neg = ["concern", "worried", "risky", "challenging", "difficult"]
        conf = ["validated", "growing", "traction", "customers"]

        if role == 'profile_2':
            for s in pos:
                if s in low:
                    self.metrics.profile2_engagement = min(1.0, self.metrics.profile2_engagement + 0.05)
                    self.metrics.positive_signals.append(s)
            for s in neg:
                if s in low:
                    self.metrics.profile2_engagement = max(0.0, self.metrics.profile2_engagement - 0.03)
                    self.metrics.red_flags.append(s)
        else:
            for s in conf:
                if s in low:
                    self.metrics.profile1_engagement = min(1.0, self.metrics.profile1_engagement + 0.03)

    def should_end(self) -> Tuple[bool, ConversationOutcome | None]:
        if self.state == ConversationState.ENDED:
            return True, self._outcome()
        if self.ensure_completion:
            if self.metrics.turn_count >= self.max_turns and self.state != ConversationState.CLOSING:
                self.state = ConversationState.CLOSING
                logger.info("chat_force_closing | reached max_turns; allowing grace turns to complete")
                return False, None
            if self.state == ConversationState.CLOSING and self._closing_turns >= self.closing_grace:
                return True, self._outcome()
            if self.metrics.turn_count >= self.hard_max_turns:
                return True, self._outcome()
        else:
            if self.metrics.turn_count >= self.max_turns:
                return True, ConversationOutcome.FOLLOW_UP_LATER
        if self.metrics.turn_count >= self.min_turns:
            if self.metrics.profile2_engagement < 0.3:
                return True, ConversationOutcome.NOT_A_FIT
            if self.metrics.profile2_engagement > 0.8 and self.metrics.profile1_engagement > 0.7:
                return True, ConversationOutcome.MUTUAL_INTEREST
        return False, None

    def _outcome(self) -> ConversationOutcome:
        iv = self.metrics.profile2_engagement
        fc = self.metrics.profile1_engagement
        if iv > 0.7 and fc > 0.7:
            return ConversationOutcome.MUTUAL_INTEREST
        if iv > 0.7:
            return ConversationOutcome.INTERESTED_NEXT_STEPS
        if iv > 0.5:
            return ConversationOutcome.NEEDS_MORE_INFO
        if iv > 0.3:
            return ConversationOutcome.FOLLOW_UP_LATER
        return ConversationOutcome.NOT_A_FIT

    async def run(self) -> dict[str, Any]:
        current = self.start_with
        p2_name = getattr(self.participant_2, 'id', '')
        p1_name = getattr(self.participant_1, 'id', '')
        logger.info(f"ai_chat_start | profile_2={p2_name} | profile_1={p1_name} | max_turns={self.max_turns}")

        while True:
            self.metrics.turn_count += 1

            if self.should_transition_state():
                self.transition()
            # If we have reached the cap and aren't closing yet, enter closing
            # so the model can perform closing turns instead of injecting text.
            if self.ensure_completion and self.metrics.turn_count >= self.max_turns and self.state != ConversationState.CLOSING:
                self.state = ConversationState.CLOSING
                self._closing_turns = 0
                logger.info("chat_force_closing | reached max_turns; allowing grace turns to complete")

            # If we already consider the conversation ended (e.g., transitioned to ENDED)
            # return without injecting any synthetic closing text.
            ended, outcome = self.should_end()
            if ended and self.state == ConversationState.ENDED:
                return {
                    'outcome': outcome.value if outcome else 'unknown',
                    'final_metrics': {
                        'profile2_engagement': self.metrics.profile2_engagement,
                        'profile1_engagement': self.metrics.profile1_engagement,
                    },
                    'conversation': self.log,
                }

            if current == 'profile_2':
                text = await self.participant_2.respond(self.log)
                self.analyze(text, 'profile_2')
                self.log.append({'speaker': 'profile_2', 'message': text, 'state': self.state.value, 'timestamp': datetime.utcnow().isoformat() + 'Z'})
                self._log_turn('profile_2', text)
                if self.state == ConversationState.CLOSING:
                    self._closing_turns += 1
                ended, outcome = self.should_end()
                if ended:
                    return {
                        'outcome': outcome.value if outcome else 'unknown',
                        'final_metrics': {
                            'profile2_engagement': self.metrics.profile2_engagement,
                            'profile1_engagement': self.metrics.profile1_engagement,
                        },
                        'conversation': self.log,
                    }
                current = 'profile_1'
            else:
                text = await self.participant_1.respond(self.log)
                self.analyze(text, 'profile_1')
                self.log.append({'speaker': 'profile_1', 'message': text, 'state': self.state.value, 'timestamp': datetime.utcnow().isoformat() + 'Z'})
                self._log_turn('profile_1', text)
                if self.state == ConversationState.CLOSING:
                    self._closing_turns += 1
                ended, outcome = self.should_end()
                if ended:
                    return {
                        'outcome': outcome.value if outcome else 'unknown',
                        'final_metrics': {
                            'profile2_engagement': self.metrics.profile2_engagement,
                            'profile1_engagement': self.metrics.profile1_engagement,
                        },
                        'conversation': self.log,
                    }
                current = 'profile_2'

    def _closing(self, outcome: ConversationOutcome | None) -> str:
        o = outcome or ConversationOutcome.FOLLOW_UP_LATER
        msgs = {
            ConversationOutcome.MUTUAL_INTEREST: "Great discussion—let’s schedule next steps and align on stakeholders.",
            ConversationOutcome.INTERESTED_NEXT_STEPS: "I’d like to dig deeper; could you share relevant materials for a follow‑up?",
            ConversationOutcome.NEEDS_MORE_INFO: "Thanks—could you send more detail and examples so we can evaluate fit?",
            ConversationOutcome.NOT_A_FIT: "Appreciate the conversation—this doesn’t seem like the right fit right now.",
            ConversationOutcome.FOLLOW_UP_LATER: "Promising direction—let’s reconnect after a few milestones."
        }
        logger.info(
            f"ai_chat_end | outcome={o.value} | turns={self.metrics.turn_count} | "
            f"p2={self.metrics.profile2_engagement:.2f} | p1={self.metrics.profile1_engagement:.2f}"
        )
        return msgs[o]

    def _log_turn(self, role: str, text: str) -> None:
        raw = text or ""
        snippet = raw if len(raw) <= 400 else raw[:400] + '...'
        # Sanitize to single line so it always prints visibly
        one_line = ' '.join(snippet.split())
        logger.info(
            f"ai_chat_turn | spk={role} st={self.state.value} t={self.metrics.turn_count} "
            f"p2={self.metrics.profile2_engagement:.2f} p1={self.metrics.profile1_engagement:.2f} | msg='{one_line}'"
        )
