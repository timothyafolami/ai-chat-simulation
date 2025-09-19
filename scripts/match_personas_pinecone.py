from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional

from loguru import logger

# Ensure project root is importable when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.embeddings import embed_persona_fields
from engine.pinecone_utils import query_top_k


def load_query_persona(query_json: Optional[str], query_id: Optional[str], base_dir: Path) -> dict:
    if query_json:
        fp = Path(query_json)
        if not fp.exists():
            raise SystemExit(f"Query JSON not found: {fp}")
        return json.loads(fp.read_text(encoding="utf-8"))
    if query_id:
        fp = base_dir / f"{query_id}.json"
        if not fp.exists():
            raise SystemExit(f"Persona id JSON not found: {fp}")
        return json.loads(fp.read_text(encoding="utf-8"))
    raise SystemExit("Provide --query-json or --query-id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Pinecone for top matches to a persona")
    parser.add_argument("--query-json", type=str, default=None, help="Path to persona JSON to use as query")
    parser.add_argument("--query-id", type=str, default=None, help="ID (filename stem) under generated_personas to use as query")
    parser.add_argument("--index-name", type=str, default=os.getenv("PINECONE_INDEX", "personas-match"))
    parser.add_argument("--namespace", type=str, default=os.getenv("PINECONE_NAMESPACE", None))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-dir", type=str, default=str(Path.cwd() / "generated_personas"))
    parser.add_argument("--w12", type=float, default=0.5, help="Weight for needs(query)->personality(target). 0..1")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")

    query_obj = load_query_persona(args.query_json, args.query_id, base_dir)
    v_needs, v_pers = embed_persona_fields(query_obj)

    base_ns = args.namespace or "default"
    ns_needs = f"{base_ns}__needs"
    ns_pers = f"{base_ns}__personality"

    # Double matching:
    # - needs(query) -> personality(namespace)
    # - personality(query) -> needs(namespace)
    res1 = query_top_k(args.index_name, v_needs, top_k=max(20, args.top_k), namespace=ns_pers, include_metadata=True)
    res2 = query_top_k(args.index_name, v_pers, top_k=max(20, args.top_k), namespace=ns_needs, include_metadata=True)

    def extract(res):
        if isinstance(res, dict):
            return res.get("matches", []) or []
        return getattr(res, "matches", []) or []

    m1 = extract(res1)
    m2 = extract(res2)

    # Aggregate scores per candidate id
    agg = {}
    for m in m1:
        cid = m.get("id")
        agg.setdefault(cid, {"score1": 0.0, "score2": 0.0, "meta": m.get("metadata", {}) or {}})
        agg[cid]["score1"] = float(m.get("score") or 0.0)
    for m in m2:
        cid = m.get("id")
        agg.setdefault(cid, {"score1": 0.0, "score2": 0.0, "meta": m.get("metadata", {}) or {}})
        agg[cid]["score2"] = float(m.get("score") or 0.0)

    # Weighted sum (double matching)
    w1 = max(0.0, min(1.0, args.w12))
    w2 = 1.0 - w1
    ranking = sorted(
        (
            (cid, w1 * v["score1"] + w2 * v["score2"], v["meta"]) for cid, v in agg.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    # Exclude self if id available
    qid = (query_obj or {}).get("id")
    filtered = [(cid, score, meta) for cid, score, meta in ranking if cid != qid]

    print("Top matches:")
    for i, (cid, score, meta) in enumerate(filtered[: args.top_k], start=1):
        print(f"{i}. {cid}  combined_score={score:.4f}")
        if meta:
            print(f"   needs: {meta.get('needs','')[:80]}...")
            print(f"   personality: {meta.get('personality','')[:80]}...")


if __name__ == "__main__":
    main()
