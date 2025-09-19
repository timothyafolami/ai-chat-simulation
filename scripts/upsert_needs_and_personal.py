from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Dict, Any

from loguru import logger
import re

try:
    # Load .env if present
    from dotenv import load_dotenv
    here = Path(__file__).resolve().parents[1]
    env_candidates = [here / ".env", Path.cwd() / ".env"]
    for env_path in env_candidates:
        if env_path.is_file():
            load_dotenv(dotenv_path=str(env_path), override=False)
            break
except Exception:
    pass

# Ensure project root is on sys.path so `engine` is importable even when run from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.embeddings import embed_persona_fields, embedding_dimension
from engine.pinecone_utils import ensure_index, upsert_personas


def load_persona_files(src_dir: Path) -> List[Path]:
    return sorted([p for p in src_dir.glob("*.json")])


def _sanitize_index_name(name: str) -> str:
    """Sanitize index name to Pinecone rules: lowercase, a-z0-9- only.

    - Replace spaces/underscores with '-'
    - Drop other invalid characters
    - Ensure starts with alphanumeric (prefix 'i' if needed)
    - Max length 45, trim trailing '-'
    """
    s = (name or "").strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-+", "-", s)
    if not s or not re.match(r"^[a-z0-9]", s):
        s = ("i" + s) if s else "idx"
    s = s[:45]
    s = s.rstrip('-') or "idx"
    return s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed persona JSONs and upsert to TWO Pinecone indexes: needs and personal"
    )
    # Default source directory preference: personal_generated (if exists) else generated_personas
    repo_root = Path(__file__).resolve().parents[1]
    personal_default = repo_root / "personal_generated"
    generated_default = repo_root / "generated_personas"
    default_src = str(personal_default if personal_default.exists() else generated_default)
    parser.add_argument(
        "--src-dir",
        type=str,
        default=default_src,
        help="Folder with persona JSONs (defaults to personal_generated if present, else generated_personas)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search for *.json under src-dir",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of files to process (0 = no cap)",
    )
    parser.add_argument("--needs-index", type=str, default=os.getenv("PINECONE_NEEDS_INDEX", "new-needs"))
    parser.add_argument("--personal-index", type=str, default=os.getenv("PINECONE_PERSONAL_INDEX", "new-personal"))
    parser.add_argument("--batch", type=int, default=100, help="Upsert batch size per index")
    args = parser.parse_args()

    src = Path(args.src_dir)
    if not src.exists():
        # Fallbacks: prefer personal_generated then generated_personas under repo root
        alt_personal = repo_root / "personal_generated"
        alt_generated = repo_root / "generated_personas"
        if alt_personal.exists():
            logger.warning("Provided src not found ({}). Falling back to {}", src, alt_personal)
            src = alt_personal
        elif alt_generated.exists():
            logger.warning("Provided src not found ({}). Falling back to {}", src, alt_generated)
            src = alt_generated
        else:
            raise SystemExit(f"Source directory not found: {src}")

    if args.recursive:
        files = sorted(src.rglob("*.json"))
    else:
        files = load_persona_files(src)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
        logger.info(
            "Discovered {} JSON files under {} (recursive={}) | limiting to max_files={}",
            len(files),
            src,
            args.recursive,
            args.max_files,
        )
    else:
        logger.info("Discovered {} JSON files under {} (recursive={})", len(files), src, args.recursive)
    if not files:
        raise SystemExit(f"No persona JSONs found in: {src}")

    needs_index_raw = args.needs_index
    personal_index_raw = args.personal_index
    needs_index = _sanitize_index_name(needs_index_raw)
    personal_index = _sanitize_index_name(personal_index_raw)
    if needs_index != needs_index_raw or personal_index != personal_index_raw:
        logger.warning(
            "Sanitized index names -> needs: '{}' -> '{}' | personal: '{}' -> '{}'",
            needs_index_raw,
            needs_index,
            personal_index_raw,
            personal_index,
        )

    dim = embedding_dimension()
    logger.info("Ensuring Pinecone indexes exist | needs='{}' personal='{}' dim={}", needs_index, personal_index, dim)
    ensure_index(needs_index, dimension=dim, metric="cosine")
    ensure_index(personal_index, dimension=dim, metric="cosine")

    batch_needs: List[Dict[str, Any]] = []
    batch_personal: List[Dict[str, Any]] = []
    total_needs = 0
    total_personal = 0

    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read JSON {fp}: {e}")
            continue
        pid = obj.get("id") or fp.stem
        v_needs, v_personal = embed_persona_fields(obj)
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
        batch_personal.append({"id": str(pid), "values": v_personal, "metadata": meta})

        if len(batch_needs) >= args.batch:
            upsert_personas(needs_index, batch_needs)
            total_needs += len(batch_needs)
            batch_needs.clear()
        if len(batch_personal) >= args.batch:
            upsert_personas(personal_index, batch_personal)
            total_personal += len(batch_personal)
            batch_personal.clear()

    if batch_needs:
        upsert_personas(needs_index, batch_needs)
        total_needs += len(batch_needs)
    if batch_personal:
        upsert_personas(personal_index, batch_personal)
        total_personal += len(batch_personal)

    logger.success(
        "Upserted to two indexes | needs={} -> '{}' | personal={} -> '{}'",
        total_needs,
        needs_index,
        total_personal,
        personal_index,
    )


if __name__ == "__main__":
    main()
