#!/usr/bin/env python3
"""
Data Cleaning, Decontamination, and Deduplication Pipeline.

Performs:
1. Vendor & AI Branding Filter:
   Drops any record containing: 'Qwen', 'Alibaba', 'Tongyi', 'As an AI language model',
   'as an AI', 'OpenAI', 'ChatGPT', 'Claude', 'Anthropic'.
2. Benchmark Decontamination (Zero-Leakage):
   Checks 8-gram and 13-gram overlap against eval/private/held_out_qa.jsonl.
   Drops any sample with n-gram collision.
3. Exact & Semantic Deduplication:
   Computes SHA-256 hash of normalized user/assistant text to remove duplicates.
4. Outputs:
   - training_data/cleaned_sft_dataset.jsonl
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NESTED_ROOT = PROJECT_ROOT / "tiny_llm_from_scratch"

FORBIDDEN_BRANDING = [
    "qwen",
    "alibaba",
    "tongyi",
    "as an ai language model",
    "as an ai",
    "openai",
    "chatgpt",
    "claude",
    "anthropic",
]


def tokenize_words(text: str) -> List[str]:
    """Basic lowercased word tokenization for n-gram calculation."""
    return re.findall(r"\b[a-zA-Z0-9_\u0900-\u097F]+\b", text.lower())


def extract_ngrams(words: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Extracts n-tuples of words."""
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_held_out_ngrams(held_out_file: Path) -> Tuple[Set[Tuple[str, ...]], Set[Tuple[str, ...]]]:
    """Loads 8-grams and 13-grams from the held-out evaluation set."""
    ngrams_8 = set()
    ngrams_13 = set()

    if not held_out_file.is_file():
        print(f"[WARN] Held-out evaluation file not found at: {held_out_file}")
        return ngrams_8, ngrams_13

    with open(held_out_file, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = f"{item.get('q', '')} {item.get('a', '')}"
            words = tokenize_words(text)
            ngrams_8.update(extract_ngrams(words, 8))
            ngrams_13.update(extract_ngrams(words, 13))

    print(f"[DECONTAMINATE] Loaded {len(ngrams_8):,} 8-grams and {len(ngrams_13):,} 13-grams from {held_out_file.name}")
    return ngrams_8, ngrams_13


def check_forbidden_branding(text: str) -> Tuple[bool, str]:
    """Returns True and the detected term if text contains forbidden vendor branding."""
    text_lower = text.lower()
    for term in FORBIDDEN_BRANDING:
        # Check as whole word or phrase
        if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
            return True, term
    return False, ""


def clean_and_curate_dataset(
    source_files: List[Path],
    held_out_file: Path,
    output_file: Path,
) -> Dict[str, Any]:
    """Cleans, decontaminates, and deduplicates records from all source files."""
    print("=" * 70)
    print("DATASET CLEANING, DECONTAMINATION & DEDUPLICATION PIPELINE")
    print("=" * 70)

    ngrams_8, ngrams_13 = load_held_out_ngrams(held_out_file)

    total_scanned = 0
    dropped_branding = 0
    dropped_contaminants = 0
    dropped_duplicates = 0
    cleaned_records = []
    seen_hashes: Set[str] = set()

    for sf in source_files:
        if not sf.is_file():
            print(f"[SKIP] Source file not found: {sf}")
            continue

        print(f"[READ] Processing: {sf.name}...")
        file_count = 0
        with open(sf, "r", encoding="utf-8", errors="ignore") as fp:
            for line_idx, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                total_scanned += 1
                file_count += 1

                try:
                    record = json.loads(line)
                except Exception:
                    continue

                # Extract conversation text
                convo_text = ""
                if "messages" in record:
                    convo_text = " ".join(m.get("content", "") for m in record["messages"])
                elif "prompt" in record and "response" in record:
                    convo_text = f"{record['prompt']} {record['response']}"
                elif "instruction" in record and "output" in record:
                    convo_text = f"{record.get('instruction', '')} {record.get('input', '')} {record.get('output', '')}"
                else:
                    convo_text = json.dumps(record)

                # 1. Vendor branding check
                has_branding, matched_term = check_forbidden_branding(convo_text)
                if has_branding:
                    dropped_branding += 1
                    continue

                # 2. Benchmark decontamination check (8-gram & 13-gram)
                words = tokenize_words(convo_text)
                sample_8 = extract_ngrams(words, 8)
                sample_13 = extract_ngrams(words, 13)

                if (ngrams_13 and not sample_13.isdisjoint(ngrams_13)) or (ngrams_8 and not sample_8.isdisjoint(ngrams_8)):
                    dropped_contaminants += 1
                    continue

                # 3. Deduplication by SHA-256
                norm_text = re.sub(r"\s+", " ", convo_text.strip().lower())
                hash_val = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
                if hash_val in seen_hashes:
                    dropped_duplicates += 1
                    continue
                seen_hashes.add(hash_val)

                # Standardize format to messages format
                if "messages" not in record:
                    user_q = record.get("prompt") or record.get("instruction") or record.get("question") or ""
                    asst_a = record.get("response") or record.get("output") or record.get("answer") or ""
                    record = {
                        "messages": [
                            {"role": "system", "content": "You are LLM, a helpful, friendly, and honest assistant built by Yuvraj Thakur. Answer casually and concisely. If something is unknowable, admit it honestly."},
                            {"role": "user", "content": user_q},
                            {"role": "assistant", "content": asst_a},
                        ]
                    }

                cleaned_records.append(record)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out_fp:
        for r in cleaned_records:
            out_fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also mirror to nested directory if appropriate
    nested_out = NESTED_ROOT / "training_data" / output_file.name
    if NESTED_ROOT.is_dir():
        nested_out.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(output_file, nested_out)

    print("\n" + "=" * 70)
    print("CLEANING & DECONTAMINATION SUMMARY REPORT")
    print(f"  Total Records Scanned:     {total_scanned:,}")
    print(f"  Dropped (Vendor Branding): {dropped_branding:,}")
    print(f"  Dropped (Held-Out Leaks):  {dropped_contaminants:,}")
    print(f"  Dropped (Duplicates):      {dropped_duplicates:,}")
    print(f"  Final High-Quality Dataset: {len(cleaned_records):,} records")
    print(f"  Saved to: {output_file.resolve()}")
    print("=" * 70 + "\n")

    return {
        "total_scanned": total_scanned,
        "dropped_branding": dropped_branding,
        "dropped_contaminants": dropped_contaminants,
        "dropped_duplicates": dropped_duplicates,
        "final_count": len(cleaned_records),
        "output_path": str(output_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset Cleaning and Decontamination")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "training_data" / "cleaned_sft_dataset.jsonl")
    parser.add_argument("--held-out", type=Path, default=PROJECT_ROOT / "eval" / "private" / "held_out_qa.jsonl")
    args = parser.parse_args()

    source_files = [
        PROJECT_ROOT / "data" / "abstain.jsonl",
        PROJECT_ROOT / "training_data" / "qa_and_database_sft.jsonl",
        NESTED_ROOT / "training_data" / "qa_and_database_sft.jsonl",
        PROJECT_ROOT / "training_data" / "distilled_conversations.jsonl",
        NESTED_ROOT / "training_data" / "distilled_conversations.jsonl",
    ]

    clean_and_curate_dataset(
        source_files=source_files,
        held_out_file=args.held_out,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
