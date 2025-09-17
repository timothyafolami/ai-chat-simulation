from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

import sys as _sys_for_sys
_pkg_root = str((Path(__file__).resolve().parents[1]))
if _pkg_root not in _sys_for_sys.path:
    _sys_for_sys.path.insert(0, _pkg_root)

from engine.generator import generate_persona


# ==========================
# Configuration (edit here)
# ==========================
# Resolve project root from this script's location
ROOT = Path(__file__).resolve().parents[1]

# Source folder containing aggregated person folders like "001__Name".
SRC_DIR = ROOT / "aggregated_personas"

# Output folder for generated JSON files.
OUT_DIR = ROOT / "generated_personas"

# Batch to process (e.g., "001"). Set to None to process all batches.
BATCH_ID = "003"

# Limit number of persons (e.g., first 100 of the selected batch). Set to None for no limit.
LIMIT = 100

# Concurrency for running generations in parallel.
CONCURRENCY = 5

# Overwrite existing JSON outputs if present.
OVERWRITE = False


def read_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def split_folder_name(stem: str) -> Tuple[str, str]:
    # Expect <batchId>__<NameSlug>[--n]
    base = stem.split("--")[0]
    if "__" in base:
        batch, name = base.split("__", 1)
        return batch, name
    return "000", stem


async def process_one(person_dir: Path, out_dir: Path, overwrite: bool) -> Optional[Path]:
    stem = person_dir.name
    out_path = out_dir / f"{stem}.json"
    if out_path.exists() and not overwrite:
        logger.info(f"Skip existing: {out_path}")
        return out_path

    profile = read_file(person_dir / "profile.txt")
    resume = read_file(person_dir / "resume.txt")
    if not profile and not resume:
        logger.warning(f"No profile/resume found in {person_dir}")
        return None

    batch_id, name_slug = split_folder_name(stem)
    entity_id = stem

    try:
        obj = await generate_persona(entity_id, profile, resume)
    except Exception as e:
        logger.error(f"Generation failed for {stem}: {e}")
        # Save raw context for debugging
        debug_dir = out_dir / "_errors"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"{stem}__profile.txt").write_text(profile or "", encoding="utf-8")
        (debug_dir / f"{stem}__resume.txt").write_text(resume or "", encoding="utf-8")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {out_path}")
    return out_path


async def main_async(src: Path, out: Path, concurrency: int, overwrite: bool, batch_id: Optional[str], limit: Optional[int]):
    # Collect person dirs (immediate children that are dirs)
    persons = [p for p in sorted(src.iterdir()) if p.is_dir()]
    if batch_id:
        persons = [p for p in persons if p.name.startswith(f"{batch_id}__")]
    if limit is not None and limit > 0:
        persons = persons[:limit]
    logger.info(f"Selected {len(persons)} person folders from {src} (batch_id={batch_id or 'ALL'}, limit={limit})")

    sem = asyncio.Semaphore(concurrency)

    async def _wrap(p: Path):
        async with sem:
            return await process_one(p, out, overwrite)

    tasks = [asyncio.create_task(_wrap(p)) for p in persons]
    await asyncio.gather(*tasks)


def main():
    src = SRC_DIR.resolve()
    out = OUT_DIR.resolve()
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    batch_id = None if (BATCH_ID is None or (isinstance(BATCH_ID, str) and BATCH_ID.upper() == 'ALL')) else BATCH_ID
    limit = LIMIT if (LIMIT is None or (isinstance(LIMIT, int) and LIMIT > 0)) else None
    asyncio.run(main_async(src, out, CONCURRENCY, OVERWRITE, batch_id, limit))


if __name__ == "__main__":
    main()
