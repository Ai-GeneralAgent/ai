import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import DatasetManifest, DocumentRecord


def ingest_knowledge_expansion(source_dir: Path, target_dir: Path):
    print("=" * 65)
    print("INGESTING AND DEDUPLICATING KNOWLEDGE EXPANSION DATA")
    print("=" * 65)
    print(f"Source Directory: {source_dir.resolve()}")
    print(f"Target Directory: {target_dir.resolve()}")

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest()

    skip_names = {
        "all_text_corpus.txt",
        "books_combined.txt",
        "web_knowledge_combined.txt",
        "starter_corpus.txt",
    }

    seen_hashes = set()
    ingested_count = 0
    skipped_duplicates = 0
    total_chars = 0

    corpus_out_path = PROJECT_ROOT / "data_knowledge_corpus.txt"

    with corpus_out_path.open("w", encoding="utf-8") as corpus_f:
        # Traverse all txt files in datasets
        datasets_dir = source_dir / "datasets" if (source_dir / "datasets").is_dir() else source_dir
        for p in datasets_dir.rglob("*.txt"):
            if p.name in skip_names:
                continue

            raw_text = p.read_text(encoding="utf-8-sig", errors="replace").strip()
            if len(raw_text) < 50:
                continue

            # Compute normalized SHA-256
            h = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(h)

            # Determine category
            rel = p.relative_to(datasets_dir)
            category = rel.parts[0] if len(rel.parts) > 1 else "general"
            if len(rel.parts) > 2:
                category = f"{rel.parts[0]}/{rel.parts[1]}"

            dest_folder = target_dir / category
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_file = dest_folder / p.name

            # Copy file
            dest_file.write_text(raw_text, encoding="utf-8")

            # Append to unified clean corpus
            corpus_f.write(f"\n\n--- [DOCUMENT: {rel.as_posix()}] ---\n\n")
            corpus_f.write(raw_text)
            corpus_f.write("\n")

            total_chars += len(raw_text)
            ingested_count += 1

            record = DocumentRecord(
                doc_id=f"doc_{p.stem}_{h[:8]}",
                filename=dest_file.name,
                domain=category,
                char_count=len(raw_text),
                fingerprint=h[:16],
                split="train",
            )
            manifest.add_document(record)

    # Copy instruction examples if present
    inst_src = source_dir / "instruction_examples.jsonl"
    if inst_src.is_file():
        inst_dest = PROJECT_ROOT / "instruction_examples.jsonl"
        shutil.copy2(inst_src, inst_dest)
        print(f"Copied instruction fine-tuning dataset to: {inst_dest.name}")

    # Save manifest
    manifest_path = PROJECT_ROOT / "manifest_knowledge_expansion.json"
    manifest.save(manifest_path)

    print("\n" + "-" * 65)
    print("INGESTION & DEDUPLICATION SUMMARY")
    print("-" * 65)
    print(f"Unique Documents Ingested: {ingested_count}")
    print(f"Duplicate Files Skipped:  {skipped_duplicates}")
    print(f"Total Characters:          {total_chars:,} ({total_chars / (1024*1024):.2f} MB)")
    print(f"Combined Corpus File:      {corpus_out_path.name}")
    print(f"Manifest File:             {manifest_path.name}")
    print(f"Knowledge Directory:       {target_dir.relative_to(PROJECT_ROOT)}")
    print("=" * 65)


if __name__ == "__main__":
    src = Path("D:/book making/llm_knowledge_expansion")
    tgt = PROJECT_ROOT / "data_knowledge"
    ingest_knowledge_expansion(src, tgt)

