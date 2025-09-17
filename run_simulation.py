from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict

from loguru import logger

from engine.agents import PersonaAgent
from engine.manager import ConversationManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a standalone AI-to-AI chat simulation")
    p.add_argument("--max-turns", type=int, default=12, help="Max conversation turns before forced closing")
    p.add_argument("--start-with", type=str, choices=["profile_1", "profile_2"], default="profile_1", help="Which profile starts the conversation")
    # Profile 1 inputs
    p.add_argument("--p1-profile-json", type=str, help="Path to JSON file for profile_1 profile object")
    p.add_argument("--p1-name", type=str, default="Profile_1", help="Fallback name_id if no profile JSON provided")
    p.add_argument("--p1-needs", type=str, default="", help="Optional needs text for profile_1")
    p.add_argument("--p1-personality", type=str, default="", help="Optional personality text for profile_1")
    # Profile 2 inputs
    p.add_argument("--p2-profile-json", type=str, help="Path to JSON file for profile_2 profile object")
    p.add_argument("--p2-name", type=str, default="Profile_2", help="Fallback name_id if no profile JSON provided")
    p.add_argument("--p2-needs", type=str, default="", help="Optional needs text for profile_2")
    p.add_argument("--p2-personality", type=str, default="", help="Optional personality text for profile_2")
    return p.parse_args()


def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def main() -> None:
    args = parse_args()

    if args.p1_profile_json:
        p1_profile = load_json_file(args.p1_profile_json)
    else:
        p1_profile = {"name_id": args.p1_name}

    if args.p2_profile_json:
        p2_profile = load_json_file(args.p2_profile_json)
    else:
        p2_profile = {"name_id": args.p2_name}

    p1 = PersonaAgent(
        role="profile_1",
        agent_id=p1_profile.get("name_id", args.p1_name),
        needs=args.p1_needs,
        personality=args.p1_personality,
        counterpart_id=p2_profile.get("name_id", args.p2_name),
        counterpart_needs=args.p2_needs,
        counterpart_personality=args.p2_personality,
    )
    p2 = PersonaAgent(
        role="profile_2",
        agent_id=p2_profile.get("name_id", args.p2_name),
        needs=args.p2_needs,
        personality=args.p2_personality,
        counterpart_id=p1_profile.get("name_id", args.p1_name),
        counterpart_needs=args.p1_needs,
        counterpart_personality=args.p1_personality,
    )

    manager = ConversationManager(participant_2=p2, participant_1=p1, max_turns=args.max_turns, start_with=args.start_with)
    result = await manager.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
