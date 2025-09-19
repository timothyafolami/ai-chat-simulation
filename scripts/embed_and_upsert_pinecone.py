from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Dict, Any

from loguru import logger

# Ensure project root is importable when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.embeddings import embed_persona_fields, embedding_dimension
from engine.pinecone_utils import ensure_index, upsert_personas


def load_persona_files(src_dir: Path) -> List[Path]:
    return sorted([p for p in src_dir.glob("*.json")])


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed persona JSONs and upsert to Pinecone")
    default_src = str((Path(__file__).resolve().parents[1] / "generated_personas"))
    parser.add_argument("--src-dir", type=str, default=default_src, help="Folder with persona JSONs (searches recursively by default)")
    parser.add_argument("--recursive", action="store_true", default=True, help="Recursively search for *.json under src-dir")
    parser.add_argument("--index-name", type=str, default=os.getenv("PINECONE_INDEX", "personas-match"))
    parser.add_argument("--namespace", type=str, default=os.getenv("PINECONE_NAMESPACE", None), help="Base namespace; script creates '<ns>__needs' and '<ns>__personality'")
    parser.add_argument("--batch", type=int, default=100, help="Upsert batch size")
    args = parser.parse_args()

    src = Path(args.src_dir)
    if not src.exists():
        alt = Path(__file__).resolve().parents[1] / "generated_personas"
        if alt.exists():
            logger.warning("Provided src not found ({}). Falling back to {}", src, alt)
            src = alt
        else:
            raise SystemExit(f"Source directory not found: {src}")

    if args.recursive:
        files = sorted(src.rglob("*.json"))
    else:
        files = load_persona_files(src)
    logger.info("Discovered {} JSON files under {} (recursive={})", len(files), src, args.recursive)
    if not files:
        raise SystemExit(f"No persona JSONs found in: {src}")

    logger.info("Preparing Pinecone index: {}", args.index_name)
    ensure_index(args.index_name, dimension=embedding_dimension(), metric="cosine")

    # Process in batches
    # We will upsert into two namespaces: needs and personality
    base_ns = args.namespace or "default"
    ns_needs = f"{base_ns}__needs"
    ns_pers = f"{base_ns}__personality"

    batch_needs: List[Dict[str, Any]] = []
    batch_pers: List[Dict[str, Any]] = []
    total_needs = 0
    total_pers = 0
    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read JSON {fp}: {e}")
            continue
        pid = obj.get("id") or fp.stem
        v_needs, v_pers = embed_persona_fields(obj)
        # Make 'source' relative to src dir (or repo root) for stable metadata
        try:
            source_rel = str(fp.relative_to(src))
        except Exception:
            try:
                source_rel = str(fp.relative_to(ROOT))
            except Exception:
                source_rel = str(fp)
        meta = {
            "id": pid,
            "needs": obj.get("needs", ""),
            "personality": obj.get("personality", ""),
            "source": source_rel,
        }
        batch_needs.append({"id": str(pid), "values": v_needs, "metadata": meta})
        batch_pers.append({"id": str(pid), "values": v_pers, "metadata": meta})

        if len(batch_needs) >= args.batch:
            upsert_personas(args.index_name, batch_needs, namespace=ns_needs)
            total_needs += len(batch_needs)
            batch_needs.clear()
        if len(batch_pers) >= args.batch:
            upsert_personas(args.index_name, batch_pers, namespace=ns_pers)
            total_pers += len(batch_pers)
            batch_pers.clear()

    if batch_needs:
        upsert_personas(args.index_name, batch_needs, namespace=ns_needs)
        total_needs += len(batch_needs)
    if batch_pers:
        upsert_personas(args.index_name, batch_pers, namespace=ns_pers)
        total_pers += len(batch_pers)

    logger.success(
        "Upserted needs={} and personality={} vectors to index='{}' (ns_needs='{}', ns_personality='{}')",
        total_needs,
        total_pers,
        args.index_name,
        ns_needs,
        ns_pers,
    )


if __name__ == "__main__":
    main()
