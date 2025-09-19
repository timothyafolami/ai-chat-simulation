from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
from loguru import logger

from engine.embeddings import embed_persona_fields
from engine.pinecone_utils import query_top_k


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "generated_personas"


def list_personas() -> list[str]:
    if not GENERATED_DIR.exists():
        return []
    return sorted([p.name for p in GENERATED_DIR.glob("*.json")])


def load_persona_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


## Removed profile-file preview helpers for deployment simplicity.


st.set_page_config(page_title="Matchmaking", page_icon="🧭", layout="wide")
st.title("Persona Matchmaking")
st.caption("Choose or enter a persona on the sidebar, then view side‑by‑side previews with top matches.")

# Sidebar controls
sb = st.sidebar
sb.title("Matchmaking – Controls")
source_choice = sb.radio("Query source", ["Local persona JSON", "Upload JSON", "Custom JSON"], index=0)
local_files = list_personas()
uploaded_obj: Optional[Dict[str, Any]] = None
selected_file: Optional[str] = None
custom_obj: Optional[Dict[str, Any]] = None

if source_choice == "Local persona JSON":
    if not local_files:
        sb.error("No persona JSONs found in generated_personas.")
    else:
        selected_file = sb.selectbox("Choose query persona", local_files, index=0)
else:
    up = sb.file_uploader("Upload persona JSON", type=["json"], accept_multiple_files=False)
    if up is not None:
        try:
            uploaded_obj = json.loads(up.read().decode("utf-8"))
            sb.success(f"Loaded uploaded JSON with id={uploaded_obj.get('id','?')}")
        except Exception as e:
            sb.error(f"Failed to parse JSON: {e}")
    # If upload selected but nothing uploaded yet, show a hint
    else:
        sb.info("Upload a JSON file with keys: id, needs, personality.")

if source_choice == "Custom JSON":
    sb.caption("Provide JSON with keys: id, needs, personality")
    if "mm_custom_json" not in st.session_state:
        st.session_state["mm_custom_json"] = json.dumps(
            {"id": "Query_Name", "needs": "", "personality": ""}, ensure_ascii=False, indent=2
        )
    sb.text_area("Persona JSON", key="mm_custom_json", height=220)
    # Parse lazily on run, but try early parse for feedback
    try:
        maybe_obj = json.loads(st.session_state.get("mm_custom_json", ""))
        if isinstance(maybe_obj, dict) and all(
            isinstance(maybe_obj.get(k), str) and maybe_obj.get(k, "").strip() for k in ("id", "needs", "personality")
        ):
            custom_obj = {
                "id": maybe_obj["id"].strip(),
                "needs": maybe_obj["needs"].strip(),
                "personality": maybe_obj["personality"].strip(),
            }
            sb.success(f"Parsed JSON for id={custom_obj['id']}")
        else:
            sb.info("Enter non-empty strings for id, needs, personality.")
    except Exception as e:
        sb.error(f"Invalid JSON: {e}")

sb.divider()
# Use two fixed Pinecone indexes (kept in code)
# - "new-needs": stores embeddings for each persona's needs
# - "new-personal": stores embeddings for each persona's personality
INDEX_NEEDS = os.getenv("PINECONE_INDEX_NEEDS", "new-needs")
INDEX_PERSONALITY = os.getenv("PINECONE_INDEX_PERSONALITY", "new-personal")
top_k = sb.slider("Top K", min_value=3, max_value=10, value=5, step=1)

# Combine two scores equally; no UI weight toggle
w1, w2 = 0.5, 0.5

go = sb.button("Find Matches", type="primary")

results_area = st.container()
with results_area:
    if go:
        if source_choice == "Local persona JSON":
            if not selected_file:
                st.error("Pick a persona first")
                st.stop()
            query_path = GENERATED_DIR / selected_file
            try:
                query_obj = load_persona_json(query_path)
            except Exception as e:
                st.error(f"Failed to load query JSON: {e}")
                st.stop()
        elif source_choice == "Upload JSON":
            if not uploaded_obj:
                st.error("Upload a persona JSON first")
                st.stop()
            query_obj = uploaded_obj
        else:
            if not custom_obj:
                st.error("Enter a valid Persona JSON first (id, needs, personality)")
                st.stop()
            query_obj = custom_obj

        try:
            v_needs, v_pers = embed_persona_fields(query_obj)
        except Exception as e:
            st.error(f"Embedding model error: {e}")
            st.stop()

        # Double matching
        try:
            # Double-match across dedicated indexes
            # needs(query) → personalities(corpus)
            res1 = query_top_k(index_name=INDEX_PERSONALITY, vector=v_needs, top_k=max(20, top_k), include_metadata=True)
            # personality(query) → needs(corpus)
            res2 = query_top_k(index_name=INDEX_NEEDS, vector=v_pers, top_k=max(20, top_k), include_metadata=True)
        except Exception as e:
            st.error(f"Pinecone query failed: {e}")
            st.info("Ensure PINECONE_API_KEY is set and index exists. If not, run the embed+upsert script first.")
            st.stop()

        def extract(res):
            if isinstance(res, dict):
                return res.get("matches", []) or []
            return getattr(res, "matches", []) or []

        m1 = extract(res1)
        m2 = extract(res2)

        # Aggregate weighted scores; keep individual similarity components
        agg: Dict[str, Dict[str, Any]] = {}
        for m in m1:
            cid = m.get("id")
            agg.setdefault(cid, {"score1": 0.0, "score2": 0.0, "meta": m.get("metadata", {}) or {}})
            agg[cid]["score1"] = float(m.get("score") or 0.0)
        for m in m2:
            cid = m.get("id")
            agg.setdefault(cid, {"score1": 0.0, "score2": 0.0, "meta": m.get("metadata", {}) or {}})
            agg[cid]["score2"] = float(m.get("score") or 0.0)
        ranking = sorted(
            (
                (
                    cid,
                    w1 * v["score1"] + w2 * v["score2"],  # combined
                    v["score1"],  # needs(query) -> personality(corpus)
                    v["score2"],  # personality(query) -> needs(corpus)
                    v["meta"],
                )
                for cid, v in agg.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        # Exclude self if query has an id
        qid = (query_obj or {}).get("id")
        filtered = [(cid, combined, s1, s2, meta) for cid, combined, s1, s2, meta in ranking if cid != qid]
        matches = filtered[: top_k]
        if not matches:
            st.warning("No matches returned.")
        else:
            st.subheader("Side‑by‑Side Review")

            # Base (selected) persona object once
            # Base persona object depending on source
            if source_choice == "Local persona JSON":
                base_obj = load_persona_json(GENERATED_DIR / selected_file)
            else:
                base_obj = query_obj
            p1_avatar = "🟦"
            p2_avatar = "🟩"

            # Render 5 boxes: left = selected persona, right = match profile
            for rank, item in enumerate(matches, start=1):
                mid, combined_score, score1, score2, meta = item
                # Build right card data; enrich from local JSON if metadata is partial
                right_obj = {"id": mid, "needs": meta.get("needs", ""), "personality": meta.get("personality", "")}
                if not right_obj.get("needs") or not right_obj.get("personality"):
                    local_match = GENERATED_DIR / f"{mid}.json"
                    if local_match.exists():
                        try:
                            full = load_persona_json(local_match)
                            right_obj["needs"] = right_obj.get("needs") or full.get("needs", "")
                            right_obj["personality"] = right_obj.get("personality") or full.get("personality", "")
                        except Exception:
                            pass

                with st.container(border=True):
                    st.markdown(f"### {rank}. {base_obj.get('id','')} ↔ {mid}")
                    st.caption(
                        f"similarity: combined={combined_score:.4f} • n→p={score1:.4f} • p→n={score2:.4f}"
                    )
                    cols_pair = st.columns(2)
                    with cols_pair[0]:
                        st.caption("Persona 1")
                        with st.chat_message("user", avatar=p1_avatar):
                            st.markdown(
                                f"Preview — {base_obj.get('id','')}\n\n"
                                f"Needs: {base_obj.get('needs','').strip()}\n\n"
                                f"Personality: {base_obj.get('personality','').strip()}"
                            )
                    with cols_pair[1]:
                        st.caption("Match Profile")
                        with st.chat_message("assistant", avatar=p2_avatar):
                            st.markdown(
                                f"Preview — {right_obj.get('id','')}\n\n"
                                f"Needs: {right_obj.get('needs','').strip()}\n\n"
                                f"Personality: {right_obj.get('personality','').strip()}"
                            )
                    # Note: external profile files removed; deployment keeps inline cards only.
