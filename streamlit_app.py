from __future__ import annotations

import json
from pathlib import Path
import time

import streamlit as st
from loguru import logger

from engine.agents import PersonaAgent
from engine.stream_runner import run_chat_stream
from engine.reviewer import review_conversation


ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated_personas"
RESULTS_DIR = ROOT / "chat_results"


def list_personas() -> list[str]:
    if not GENERATED_DIR.exists():
        return []
    return sorted([p.name for p in GENERATED_DIR.glob("*.json")])


def load_persona(name: str) -> dict:
    return json.loads((GENERATED_DIR / name).read_text(encoding="utf-8"))


def parse_persona_json(txt: str) -> dict | None:
    try:
        obj = json.loads(txt or "")
        if not isinstance(obj, dict):
            return None
        idv = obj.get("id")
        needs = obj.get("needs")
        pers = obj.get("personality")
        if all(isinstance(x, str) and x.strip() for x in (idv, needs, pers)):
            return {"id": idv.strip(), "needs": needs.strip(), "personality": pers.strip()}
        return None
    except Exception:
        return None


st.set_page_config(page_title="AI↔AI Chat Simulator", page_icon="🤖", layout="wide")

st.sidebar.title("AI↔AI Chat – Controls")
files = list_personas()
if len(files) < 2:
    st.sidebar.warning("Fewer than 2 persona JSONs found. You can still use Custom JSON inputs.")

st.sidebar.subheader("Persona Sources")
col1, col2 = st.sidebar.columns(2)
with col1:
    p1_source = st.selectbox("P1 Source", ["Existing", "Custom JSON"], index=0, key="p1_source")
with col2:
    p2_source = st.selectbox("P2 Source", ["Existing", "Custom JSON"], index=0, key="p2_source")

if p1_source == "Existing":
    if files:
        p1_file = st.selectbox("Persona 1", files, index=0, key="p1_file")
    else:
        st.info("No existing personas found. Switch to Custom JSON.")
else:
    st.text("Provide JSON with keys: id, needs, personality")
    p1_json = st.text_area(
        "Persona 1 JSON",
        placeholder='{"id":"P1_Name","needs":"...","personality":"..."}',
        height=160,
        key="p1_json",
    )

if p2_source == "Existing":
    if files:
        default_index = 1 if len(files) > 1 else 0
        p2_file = st.selectbox("Persona 2", files, index=default_index, key="p2_file")
    else:
        st.info("No existing personas found. Switch to Custom JSON.")
else:
    st.text("Provide JSON with keys: id, needs, personality")
    p2_json = st.text_area(
        "Persona 2 JSON",
        placeholder='{"id":"P2_Name","needs":"...","personality":"..."}',
        height=160,
        key="p2_json",
    )

max_turns = st.sidebar.slider("Max messages", min_value=6, max_value=14, value=10, step=1)
starter_choice = st.sidebar.radio("Who starts?", ["Persona 1", "Persona 2"], index=0)
preview_btn = st.sidebar.button("Preview Profiles")
start_btn = st.sidebar.button("Start Chat", type="primary")

out_box = st.sidebar.container()

st.title("Live AI↔AI Conversation")
preview_area = st.container()
chat_area = st.container()
status_text = st.empty()

# Remember preview toggle across reruns
if "_preview_active" not in st.session_state:
    st.session_state["_preview_active"] = False
if preview_btn:
    st.session_state["_preview_active"] = True

# Show preview when requested (before starting chat)
if st.session_state["_preview_active"]:
    with preview_area:
        st.subheader("Preview")
        p1_avatar = "🟦"
        p2_avatar = "🟩"
        # Resolve P1
        if st.session_state.get("p1_source", "Existing") == "Existing":
            try:
                p1_prev = load_persona(st.session_state.get("p1_file"))
            except Exception:
                p1_prev = None
        else:
            p1_prev = parse_persona_json(st.session_state.get("p1_json", ""))
        # Resolve P2
        if st.session_state.get("p2_source", "Existing") == "Existing":
            try:
                p2_prev = load_persona(st.session_state.get("p2_file"))
            except Exception:
                p2_prev = None
        else:
            p2_prev = parse_persona_json(st.session_state.get("p2_json", ""))

        any_ok = False
        cols = st.columns(2)
        # Persona 1 preview (if available)
        with cols[0]:
            st.caption("Persona 1")
            if p1_prev:
                any_ok = True
                with st.chat_message("user", avatar=p1_avatar):
                    st.markdown(
                        f"Preview — {p1_prev.get('id','')}\n\n"
                        f"Needs: {p1_prev.get('needs','').strip()}\n\n"
                        f"Personality: {p1_prev.get('personality','').strip()}"
                    )
            else:
                st.info("Waiting for valid Persona 1 (select file or enter JSON).")
        # Persona 2 preview (if available)
        with cols[1]:
            st.caption("Persona 2")
            if p2_prev:
                any_ok = True
                with st.chat_message("assistant", avatar=p2_avatar):
                    st.markdown(
                        f"Preview — {p2_prev.get('id','')}\n\n"
                        f"Needs: {p2_prev.get('needs','').strip()}\n\n"
                        f"Personality: {p2_prev.get('personality','').strip()}"
                    )
            else:
                st.info("Waiting for valid Persona 2 (select file or enter JSON).")
        if not any_ok:
            st.warning("Please provide at least one valid persona to preview (file or valid JSON with id, needs, personality).")

if start_btn:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Resolve P1
    if st.session_state.get("p1_source", "Existing") == "Existing":
        try:
            p1 = load_persona(st.session_state.get("p1_file"))
        except Exception:
            st.sidebar.error("Select a valid Persona 1 file or provide valid JSON")
            st.stop()
    else:
        p1 = parse_persona_json(st.session_state.get("p1_json", ""))
        if not p1:
            st.sidebar.error("Invalid Persona 1 JSON. Expect keys: id, needs, personality (non-empty strings).")
            st.stop()

    # Resolve P2
    if st.session_state.get("p2_source", "Existing") == "Existing":
        try:
            p2 = load_persona(st.session_state.get("p2_file"))
        except Exception:
            st.sidebar.error("Select a valid Persona 2 file or provide valid JSON")
            st.stop()
    else:
        p2 = parse_persona_json(st.session_state.get("p2_json", ""))
        if not p2:
            st.sidebar.error("Invalid Persona 2 JSON. Expect keys: id, needs, personality (non-empty strings).")
            st.stop()

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

    with chat_area:
        st.write(f"Chat between: {p1.get('id')} ↔ {p2.get('id')}")
        rows = []
        t0 = time.perf_counter()

        # Choose avatars
        p1_avatar = "🟦"
        p2_avatar = "🟩"

        start_with = 'profile_1' if starter_choice == 'Persona 1' else 'profile_2'

        for event in run_chat_stream(participant_1=a1, participant_2=a2, max_turns=max_turns, start_with=start_with):
            if event["type"] == "turn":
                d = event["data"]
                rows.append(d)
                role = "user" if d["speaker"] == "profile_1" else "assistant"
                avatar = p1_avatar if d["speaker"] == "profile_1" else p2_avatar
                with st.chat_message(role, avatar=avatar):
                    st.markdown(
                        f"{d['message']}\n\n"
                        f"<span style='color:gray;font-size:smaller'>[{d['state']}] {d['timestamp']}</span>",
                        unsafe_allow_html=True,
                    )
                status_text.info(f"{len(rows)} / {max_turns} messages")

            elif event["type"] == "end":
                result = event["data"]
                t1 = time.perf_counter()
                status_text.success(f"Chat completed in {t1 - t0:.1f}s | outcome: {result.get('outcome')}")

                # Post-chat review (blocking)
                import asyncio
                rev = asyncio.run(review_conversation(p1, p2, result.get("conversation", [])))

                # Sidebar summary
                with out_box:
                    st.subheader("Review")
                    st.metric("Similarity", f"{rev.get('similarity_score', 0):.3f}")
                    dec = rev.get("chat_decision", {})
                    st.write(f"Decision: {dec.get('decision','?')} (conf: {dec.get('confidence','?')})")
                    st.caption(dec.get("rationale", ""))

                # Persist
                out_name = f"{p1.get('id')}__vs__{p2.get('id')}.json"
                (RESULTS_DIR / out_name).write_text(json.dumps(rev, ensure_ascii=False, indent=2), encoding="utf-8")
                break

else:
    st.info("Select two personas and click Start Chat")
