# AI Chat Simulation (Standalone)

This folder contains a minimal, standalone copy of the AI‑to‑AI chat engine so you can run special experiments without the full API.

What’s included
- engine/agents.py — PersonaAgent with prompt + PROFILE_CONTEXT injection
- engine/manager.py — ConversationManager state machine and heuristics
- engine/states.py — ConversationState/Outcome + metrics
- engine/llm.py — Minimal OpenAI Chat client via LangChain
- prompts/ai_agent_prompt.md — Unified system prompt
- prompts/chat_decision_prompt.md — Reviewer prompt for post‑chat decision
- run_simulation.py — CLI entrypoint to run a conversation and print JSON
- scripts/aggregate_persona_data.py — Flatten persona batches into one folder
- scripts/generate_personas_from_text.py — Generate needs/personality JSON from profile+resume
- scripts/run_chat_with_review.py — Run a 12‑message chat and LLM review
- streamlit_app.py — Streamlit UI to select personas, stream chat, and view review

Quick start
1) Create a virtualenv and install minimal deps
```bash
cd ai-chat-simulation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2) Set your environment
```bash
export OPENAI_API_KEY="..."
# Optional
export OPENAI_MODEL="gpt-4o-mini"
# To override prompt location
# export PROMPTS_DIR="/path/to/prompts"
```
3) Run a simulation (with inline minimal profiles)
```bash
python run_simulation.py \
  --max-turns 10 \
  --start-with profile_1 \
  --p1-name Dev_Example --p1-needs "Build data infra" --p1-personality "Direct" \
  --p2-name Inv_Example --p2-needs "Meet data founders" --p2-personality "Analytical"
```

Or load full profile JSON objects
```bash
python run_simulation.py \
  --p1-profile-json /path/to/profile_1.json \
  --p2-profile-json /path/to/profile_2.json \
  --p1-needs "..." --p1-personality "..." \
  --p2-needs "..." --p2-personality "..." \
  --start-with profile_1   # or profile_2
```

Notes
- Output is JSON with `outcome`, `final_metrics`, and `conversation` (turns with timestamps/state).
- Logging uses Loguru; see console for `ai_chat_turn` and state transitions.
- By default `profile_1` starts; pass `--start-with profile_2` to change the opener. The manager alternates turns and forces a closing phase when approaching max turns. Closing statements are produced by the model (no synthetic injected lines).

Troubleshooting
- If you see `openai_chat not initialized`, ensure `OPENAI_API_KEY` is set in your environment.
- If you want to modify agent behavior, edit `prompts/ai_agent_prompt.md` or the style constraints in `engine/agents.py`.
- You can safely tweak turn limits and phase pacing in `engine/manager.py` (e.g., `max_turns`, `closing_grace`).

Aggregating persona files (CSV + people folders)
```bash
python scripts/aggregate_persona_data.py \
  --src ai-chat-simulation/persona_data \
  --dst ai-chat-simulation/aggregated_personas

# Options:
#   --no-root-csvs   keep each batch index.csv inside a per-batch subfolder instead of root
#   --overwrite      overwrite existing files in destination
```
Result:
- Destination contains all batch index CSVs, renamed to compact numeric names like 001.csv, 002.csv, ...
- ~N person folders named like 004__Kwame_Bassey (format: <batchId>__<Name>), with profile.txt and resume.txt when available. Duplicate names within a batch get a --1, --2 suffix.

Generate personas (needs + personality, ~40–60 words each)
```bash
# Requires OPENAI_API_KEY in env
export OPENAI_API_KEY="..."

python scripts/generate_personas_from_text.py

# Outputs JSON files named like 004__Kwame_Bassey.json
# Schema:
# {
#   "id": "004__Kwame_Bassey",
#   "needs": "... 40–60 words ...",
#   "personality": "... 40–60 words ..."
# }
```
To change batch, limit, or concurrency, edit the constants at the top of:
- ai-chat-simulation/scripts/generate_personas_from_text.py

Run a 12‑message chat + review
```bash
# Ensure you have two JSON files in ai-chat-simulation/generated_personas

python scripts/run_chat_with_review.py

# Edit constants inside the script to choose personas and output path.
```

Streamlit app
```bash
cd ai-chat-simulation
streamlit run streamlit_app.py
```
In the sidebar:
- Pick Persona 1 and Persona 2 from generated_personas
- Choose "Who starts?" (Persona 1 or Persona 2)
- Click "Preview Profiles" to view each persona’s Needs and Personality
- Click Start Chat to begin the conversation
- When it finishes, the sidebar shows similarity score and decision

Chat view behavior:
- When you click "Preview Profiles", the chat shows each selected persona’s Needs and Personality for quick context.
- Messages then stream live with role avatars, phase tags, and timestamps.

Grounding & truthfulness:
- The agent prompt includes rules to avoid invention and unconfirmed commitments. Agents only rely on PROFILE_CONTEXT and the latest message; if info is missing, they ask one concise clarifying question.
