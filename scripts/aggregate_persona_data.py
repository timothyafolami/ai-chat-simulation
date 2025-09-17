from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Iterable


@dataclass
class PersonRow:
    batch: str
    uuid: str
    name: str
    path: str


def _safe_slug(text: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in (text or "").strip()]
    slug = "".join(keep).strip("_")
    return slug or "person"


def _iter_batches(src_root: Path) -> list[Path]:
    return [p for p in sorted(src_root.glob("profiles_batch_*")) if p.is_dir()]


def _batch_short_id(batch_name: str, fallback_number: int) -> str:
    """Return a compact batch id like 001, 002 from a folder name.

    Extracts digits from the name (e.g., profiles_batch_004 → 004). If no digits
    found, uses fallback_number (1-based) and zero-pads to 3 digits.
    """
    digits = ''.join(ch for ch in batch_name if ch.isdigit())
    if digits:
        try:
            num = int(digits)
        except ValueError:
            num = fallback_number
    else:
        num = fallback_number
    return f"{num:03d}"


def _parse_index_csv(path: Path, batch_name: str) -> Iterable[PersonRow]:
    """Parse an index.csv file, being tolerant to column naming/order.

    Expected columns (observed):
      - first column: uuid
      - second column: name
      - last column: relative path under the batch directory (e.g. People/Role/Name)
    If a header exists with 'path'/'id'/'uuid'/'name', DictReader is used.
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not text:
        return []

    # Peek header to decide
    sniff = csv.Sniffer()
    try:
        dialect = sniff.sniff("\n".join(text[:10]))
    except Exception:
        dialect = csv.excel

    # Try DictReader first
    reader = csv.reader(text, dialect)
    header = next(reader)
    header_lower = [h.strip().lower() for h in header]

    # If it looks like a header row (non-uuid first cell), use DictReader
    looks_header = ("path" in header_lower) or ("uuid" in header_lower) or ("id" in header_lower)
    rows_iter: Iterable[Dict[str, Any]]
    if looks_header:
        dict_reader = csv.DictReader(text, dialect=dialect)
        for row in dict_reader:
            uuid = (row.get("uuid") or row.get("id") or "").strip()
            name = (row.get("name") or row.get("full_name") or row.get("Name") or "").strip()
            path_col = (row.get("path") or row.get("relative_path") or "").strip()
            if not path_col:
                # fallback to last column if missing
                path_col = list(row.values())[-1] if row else ""
            if uuid and path_col:
                yield PersonRow(batch=batch_name, uuid=uuid, name=name, path=path_col)
    else:
        # Fallback: treat first row as data, including columns shown in samples
        # Reconstruct iteration including the first row we consumed
        def row_iter():
            yield header
            for r in reader:
                yield r

        for r in row_iter():
            if not r:
                continue
            uuid = (r[0] if len(r) > 0 else "").strip()
            name = (r[1] if len(r) > 1 else "").strip()
            rel_path = (r[-1] if len(r) > 0 else "").strip()
            if uuid and rel_path:
                yield PersonRow(batch=batch_name, uuid=uuid, name=name, path=rel_path)


def aggregate_personas(
    src_root: Path,
    dst_root: Path,
    keep_csvs_in_root: bool = True,
    overwrite: bool = False,
) -> dict:
    dst_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    batches = _iter_batches(src_root)
    for idx, batch_dir in enumerate(batches, start=1):
        batch_name = batch_dir.name  # e.g., profiles_batch_004
        batch_id = _batch_short_id(batch_name, idx)  # e.g., 004
        index_csv = batch_dir / "index.csv"
        if index_csv.exists():
            target_csv_name = f"{batch_id}.csv" if keep_csvs_in_root else "index.csv"
            target_csv_path = (dst_root / target_csv_name) if keep_csvs_in_root else (dst_root / batch_name / target_csv_name)
            target_csv_path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not target_csv_path.exists():
                shutil.copy2(index_csv, target_csv_path)

        if not index_csv.exists():
            continue
        for row in _parse_index_csv(index_csv, batch_id):
            # Build source person dir
            src_person_dir = batch_dir / row.path
            # Expected files
            prof = src_person_dir / "profile.txt"
            resu = src_person_dir / "resume.txt"
            # Destination folder: <batch>__<NameSlug>, ensure uniqueness with suffixes if needed
            name_slug = _safe_slug(row.name) or "Unknown"
            base_stem = f"{row.batch}__{name_slug}"

            def _unique_dir(root: Path, stem: str) -> Path:
                candidate = root / stem
                if not candidate.exists():
                    return candidate
                i = 1
                while True:
                    c = root / f"{stem}--{i}"
                    if not c.exists():
                        return c
                    i += 1

            if overwrite:
                dst_person_dir = dst_root / base_stem
                dst_person_dir.mkdir(parents=True, exist_ok=True)
            else:
                dst_person_dir = _unique_dir(dst_root, base_stem)
                dst_person_dir.mkdir(parents=True, exist_ok=True)

            # Copy files if present
            if prof.exists():
                dst_prof = dst_person_dir / "profile.txt"
                if overwrite or not dst_prof.exists():
                    shutil.copy2(prof, dst_prof)
            if resu.exists():
                dst_resu = dst_person_dir / "resume.txt"
                if overwrite or not dst_resu.exists():
                    shutil.copy2(resu, dst_resu)

            manifest_rows.append(
                {
                    "batch": row.batch,
                    "uuid": row.uuid,
                    "name": row.name,
                    "folder_name": dst_person_dir.name,
                    "source_path": str(src_person_dir.relative_to(src_root)),
                    "dest_dir": str(dst_person_dir.relative_to(dst_root)),
                    "profile": (prof.exists()),
                    "resume": (resu.exists()),
                }
            )

    # Write a manifest CSV for convenience
    manifest_path = dst_root / "aggregated_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch",
                "uuid",
                "name",
                "folder_name",
                "source_path",
                "dest_dir",
                "profile",
                "resume",
            ],
        )
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)

    return {
        "batches": len(batches),
        "people": len(manifest_rows),
        "manifest": str(manifest_path),
        "destination": str(dst_root),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate persona data into a single folder")
    p.add_argument(
        "--src",
        type=str,
        default=str(Path("ai-chat-simulation") / "persona_data"),
        help="Source root containing profiles_batch_* directories",
    )
    p.add_argument(
        "--dst",
        type=str,
        default=str(Path("ai-chat-simulation") / "aggregated_personas"),
        help="Destination root for aggregated data",
    )
    p.add_argument("--no-root-csvs", action="store_true", help="Do not copy index.csv files to root (keep per-batch)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    result = aggregate_personas(src, dst, keep_csvs_in_root=not args.no_root_csvs, overwrite=args.overwrite)
    print(result)


if __name__ == "__main__":
    main()
