#!/usr/bin/env python3
"""
Combines historical expanded text corpus with modern knowledge corpus
into a unified, deduplicated master training dataset: data_master_combined.txt.
"""
import hashlib
import io
import os
from pathlib import Path
import re
import sys
import time

# Environmental thread protections on Windows CPU
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def normalize_chunk(text: str) -> str:
    """Normalizes line breaks, whitespace, and Unicode quotation marks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def check_chunk_quality(text: str) -> bool:
    """Basic filter for degenerate loops or very short garbage."""
    if len(text) < 40:
        return False
    words = text.split()
    if len(words) >= 30:
        # Check 4-gram repetition
        four_grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        if four_grams:
            ratio = 1.0 - (len(set(four_grams)) / len(four_grams))
            if ratio > 0.50:  # Excessive 4-gram loop
                return False
    return True


def combine_corpora():
    print("=" * 65)
    print("COMBINING HISTORICAL & MODERN CORPORA INTO MASTER DATASET")
    print("=" * 65)

    knowledge_file = PROJECT_ROOT / "data_knowledge_corpus.txt"
    merged_old_file = PROJECT_ROOT / "data_merged_expanded.txt"
    master_out_file = PROJECT_ROOT / "data_master_combined.txt"

    if not knowledge_file.is_file():
        raise FileNotFoundError(f"Knowledge corpus not found: {knowledge_file}")
    if not merged_old_file.is_file():
        raise FileNotFoundError(f"Old merged corpus not found: {merged_old_file}")

    print(f"Knowledge Corpus (New):  {knowledge_file.name} ({knowledge_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Historical Corpus (Old): {merged_old_file.name} ({merged_old_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Master Output:           {master_out_file.name}")

    seen_fingerprints = set()
    total_chars = 0
    total_chunks = 0
    skipped_dups = 0
    t0 = time.time()

    with master_out_file.open("w", encoding="utf-8") as out_f:
        # Step 1: Ingest modern knowledge corpus (high priority)
        print("\n[STEP 1/3] Ingesting modern curated knowledge corpus...")
        raw_knowledge = knowledge_file.read_text(encoding="utf-8", errors="replace")
        # Split by document headers
        doc_pattern = re.compile(r"(--- \[DOCUMENT: [^\]]+\] ---)")
        parts = doc_pattern.split(raw_knowledge)

        current_header = "--- [DOCUMENT: curated/knowledge_base] ---"
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("--- [DOCUMENT:"):
                current_header = part
                continue

            cleaned = normalize_chunk(part)
            if not check_chunk_quality(cleaned):
                continue

            # Fingerprint chunk (first 1000 chars hash)
            fp = hashlib.md5(cleaned[:1000].encode("utf-8")).hexdigest()
            seen_fingerprints.add(fp)

            out_f.write(f"\n\n{current_header}\n\n")
            out_f.write(cleaned)
            out_f.write("\n")
            total_chars += len(cleaned)
            total_chunks += 1

        print(f"  Ingested {total_chunks} knowledge documents ({total_chars / (1024*1024):.2f} MB).")

        # Step 2: Stream in old merged data in paragraph blocks with deduplication
        print("\n[STEP 2/3] Streaming and deduplicating historical corpus...")
        old_chunks = 0
        old_chars = 0
        current_block = []
        current_len = 0

        with merged_old_file.open("r", encoding="utf-8-sig", errors="replace") as old_f:
            for line in old_f:
                current_block.append(line)
                current_len += len(line)

                # Form chunks around 4,000 - 8,000 characters
                if current_len >= 5000 and line.strip() == "":
                    block_text = "".join(current_block)
                    current_block = []
                    current_len = 0

                    cleaned = normalize_chunk(block_text)
                    if not check_chunk_quality(cleaned):
                        continue

                    # Check fingerprint against seen chunks
                    fp = hashlib.md5(cleaned[:1000].encode("utf-8")).hexdigest()
                    if fp in seen_fingerprints:
                        skipped_dups += 1
                        continue
                    seen_fingerprints.add(fp)

                    out_f.write("\n\n--- [DOCUMENT: historical/classic_literature_archive] ---\n\n")
                    out_f.write(cleaned)
                    out_f.write("\n")
                    total_chars += len(cleaned)
                    old_chars += len(cleaned)
                    total_chunks += 1
                    old_chunks += 1

                    if old_chunks % 1000 == 0:
                        print(f"    Progress: {old_chunks} historical blocks ingested ({old_chars / (1024*1024):.2f} MB)...")

            # Remainder
            if current_block:
                block_text = "".join(current_block)
                cleaned = normalize_chunk(block_text)
                if check_chunk_quality(cleaned):
                    fp = hashlib.md5(cleaned[:1000].encode("utf-8")).hexdigest()
                    if fp not in seen_fingerprints:
                        out_f.write("\n\n--- [DOCUMENT: historical/classic_literature_archive] ---\n\n")
                        out_f.write(cleaned)
                        out_f.write("\n")
                        total_chars += len(cleaned)
                        old_chars += len(cleaned)
                        total_chunks += 1
                        old_chunks += 1

        print(f"  Ingested {old_chunks} historical blocks ({old_chars / (1024*1024):.2f} MB).")
        print(f"  Deduplicated & skipped {skipped_dups} duplicate blocks.")

        # Step 3: Append conversational instruction examples and curriculum flashcards
        print("\n[STEP 3/3] Ingesting instruction tuning & curriculum flashcards...")
        inst_file = PROJECT_ROOT / "instruction_examples.jsonl"
        if inst_file.is_file():
            import json
            inst_lines = []
            with inst_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            msgs = obj.get("messages", [])
                            if len(msgs) >= 2:
                                u_text = msgs[0].get("content", "")
                                a_text = msgs[1].get("content", "")
                                inst_lines.append(f"User: {u_text}\nAssistant: {a_text}\n")
                        except Exception:
                            pass
            if inst_lines:
                inst_corpus = "\n\n--- [DOCUMENT: instructions/conversational_sft_seed] ---\n\n" + "\n".join(inst_lines)
                out_f.write(inst_corpus)
                total_chars += len(inst_corpus)
                print(f"  Ingested {len(inst_lines)} multi-turn instruction dialogues.")

    elapsed = time.time() - t0
    final_size_mb = master_out_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("MASTER DATASET CREATION COMPLETE")
    print("=" * 65)
    print(f"Output File:     {master_out_file.name}")
    print(f"Total Size:      {final_size_mb:.2f} MB ({total_chars:,} characters)")
    print(f"Total Documents: {total_chunks:,}")
    print(f"Time Taken:      {elapsed:.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    combine_corpora()

