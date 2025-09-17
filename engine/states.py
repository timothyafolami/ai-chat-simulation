from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class ConversationState(Enum):
    INTRODUCTION = "introduction"
    DISCOVERY = "discovery"
    DEEP_DIVE = "deep_dive"
    CHALLENGES = "challenges"
    CLOSING = "closing"
    ENDED = "ended"


class ConversationOutcome(Enum):
    INTERESTED_NEXT_STEPS = "interested_next_steps"
    NEEDS_MORE_INFO = "needs_more_info"
    NOT_A_FIT = "not_a_fit"
    MUTUAL_INTEREST = "mutual_interest"
    FOLLOW_UP_LATER = "follow_up_later"


@dataclass
class ConversationMetrics:
    turn_count: int = 0
    profile1_engagement: float = 0.5
    profile2_engagement: float = 0.5
    key_concerns_addressed: List[str] | None = None
    positive_signals: List[str] | None = None
    red_flags: List[str] | None = None

    def __post_init__(self):
        if self.key_concerns_addressed is None:
            self.key_concerns_addressed = []
        if self.positive_signals is None:
            self.positive_signals = []
        if self.red_flags is None:
            self.red_flags = []

