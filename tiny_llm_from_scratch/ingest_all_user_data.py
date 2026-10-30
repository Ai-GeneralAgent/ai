#!/usr/bin/env python3
r"""
Master Data Ingestion & Unification Pipeline (v3.0.0)
Ingests and organizes:
1. LLM QA Training Dataset (C:\Users\YUVRAJ\Downloads\llm_qa_training_dataset\llm_qa_training)
2. 10K QA Dataset (C:\Users\YUVRAJ\Downloads\llm_qa_10k_dataset)
3. Classic Literature Books (D:\book making\more_books_for_llm\books_txt)
4. Subject Curriculum Books (D:\book making\llm_knowledge_expansion\Data\ebalbharati - txt only, NO pdfs)
5. Conversational & Reasoning SFT (D:\book making\llm_knowledge_expansion\datasets: SlimOrca, Coding, SFT)

Merges all sources with the existing master corpus into:
D:\tiny_llm_from_scratch\tiny_llm_from_scratch\data_master_v3_unified.txt
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
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

# Target output directories
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
QA_DIR = TRAINING_DATA_DIR / "qa_instructions"
CURRICULUM_DIR = TRAINING_DATA_DIR / "curriculum_textbooks"
BOOKS_DIR = TRAINING_DATA_DIR / "classic_literature"
REASONING_DIR = TRAINING_DATA_DIR / "reasoning_sft"

for d in [QA_DIR, CURRICULUM_DIR, BOOKS_DIR, REASONING_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OUT_FILE = PROJECT_ROOT / "data_master_v3_unified.txt"


def normalize_text(text: str) -> str:
    """Normalizes whitespace, line breaks, and quotation marks."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def format_qa_pair(question: str, answer: str, category: str = "General") -> str:
    """Formats QA pair into clean training text with reasoning guidance."""
    q = question.strip()
    a = answer.strip()
    return f"<user>\n[{category}] {q}\n</user>\n<assistant>\n{a}\n</assistant>\n"


def format_messages_jsonl(line: str) -> str:
    """Converts a standard chat jsonl line into model training text."""
    try:
        data = json.loads(line)
        msgs = data.get("messages", [])
        if not msgs:
            return ""
        chunks = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            if not content:
                continue
            if role in ["user", "human"]:
                chunks.append(f"<user>\n{content}\n</user>")
            elif role in ["assistant", "gpt", "bot"]:
                chunks.append(f"<assistant>\n{content}\n</assistant>")
            elif role == "system":
                chunks.append(f"<system>\n{content}\n</system>")
        return "\n".join(chunks) + "\n" if chunks else ""
    except Exception:
        return ""


def main():
    print("=" * 68)
    print("   MASTER DATA INGESTION & UNIFICATION PIPELINE (v3.0.0)")
    print("=" * 68)
    t0 = time.time()

    seen_hashes = set()
    total_chars = 0
    total_docs = 0
    domain_counts = {}

    out_handle = open(OUT_FILE, "w", encoding="utf-8", errors="replace")

    def write_doc(text: str, domain: str = "general"):
        nonlocal total_chars, total_docs
        norm = normalize_text(text)
        if len(norm) < 40:
            return
        h = hashlib.sha256(norm[:400].encode("utf-8")).hexdigest()
        if h in seen_hashes:
            return
        seen_hashes.add(h)
        out_handle.write(norm + "\n\n<|endoftext|>\n\n")
        total_chars += len(norm)
        total_docs += 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # -------------------------------------------------------------
    # 1. Existing Combined Master Dataset (data_master_combined.txt)
    # -------------------------------------------------------------
    existing_master = PROJECT_ROOT / "data_master_combined.txt"
    if existing_master.is_file():
        print(f"\n[1/6] Ingesting existing master corpus: {existing_master.name}...")
        count = 0
        with open(existing_master, "r", encoding="utf-8", errors="replace") as f:
            chunk = []
            for line in f:
                if "<|endoftext|>" in line:
                    doc = "".join(chunk)
                    if doc.strip():
                        write_doc(doc, domain="existing_master")
                        count += 1
                    chunk = []
                else:
                    chunk.append(line)
            if chunk:
                doc = "".join(chunk)
                if doc.strip():
                    write_doc(doc, domain="existing_master")
                    count += 1
        print(f"      Ingested {count:,} documents from existing master corpus.")

    # -------------------------------------------------------------
    # 2. LLM QA 10K Dataset (C:\Users\YUVRAJ\Downloads\llm_qa_10k_dataset)
    # -------------------------------------------------------------
    qa_10k_path = Path(r"C:\Users\YUVRAJ\Downloads\llm_qa_10k_dataset")
    if qa_10k_path.is_dir():
        print(f"\n[2/6] Ingesting QA 10K Dataset from {qa_10k_path.name}...")
        chat_file = qa_10k_path / "qa_10k_chat.jsonl"
        count = 0
        if chat_file.is_file():
            shutil.copy2(chat_file, QA_DIR / "qa_10k_chat.jsonl")
            with open(chat_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    doc = format_messages_jsonl(line)
                    if doc:
                        write_doc(doc, domain="qa_10k")
                        count += 1
        print(f"      Ingested {count:,} QA pairs from qa_10k_chat.jsonl.")

    # -------------------------------------------------------------
    # 3. LLM QA Training Dataset (C:\Users\YUVRAJ\Downloads\llm_qa_training_dataset\llm_qa_training)
    # -------------------------------------------------------------
    qa_train_path = Path(r"C:\Users\YUVRAJ\Downloads\llm_qa_training_dataset\llm_qa_training")
    if qa_train_path.is_dir():
        print(f"\n[3/6] Ingesting QA Training Dataset from {qa_train_path.name}...")
        chat_file = qa_train_path / "qa_all_chat.jsonl"
        count = 0
        if chat_file.is_file():
            shutil.copy2(chat_file, QA_DIR / "qa_all_chat.jsonl")
            with open(chat_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    doc = format_messages_jsonl(line)
                    if doc:
                        write_doc(doc, domain="qa_curriculum")
                        count += 1
        print(f"      Ingested {count:,} QA pairs from qa_all_chat.jsonl.")

    # -------------------------------------------------------------
    # 4. Classic Literature Books (D:\book making\more_books_for_llm\books_txt)
    # -------------------------------------------------------------
    books_txt_path = Path(r"D:\book making\more_books_for_llm\books_txt")
    if books_txt_path.is_dir():
        print(f"\n[4/6] Ingesting Classic Books from {books_txt_path.name}...")
        count = 0
        for txt_file in sorted(books_txt_path.glob("*.txt")):
            dest = BOOKS_DIR / txt_file.name
            shutil.copy2(txt_file, dest)
            try:
                content = txt_file.read_text(encoding="utf-8", errors="replace")
                # Split large books into chapters or 4,000-char chunks for clean learning
                paragraphs = content.split("\n\n")
                current_chunk = []
                cur_len = 0
                for p in paragraphs:
                    current_chunk.append(p)
                    cur_len += len(p)
                    if cur_len >= 3000:
                        doc = f"### [Classic Literature: {txt_file.stem.replace('_', ' ').title()}]\n\n" + "\n\n".join(current_chunk)
                        write_doc(doc, domain="literature")
                        count += 1
                        current_chunk = []
                        cur_len = 0
                if current_chunk:
                    doc = f"### [Classic Literature: {txt_file.stem.replace('_', ' ').title()}]\n\n" + "\n\n".join(current_chunk)
                    write_doc(doc, domain="literature")
                    count += 1
            except Exception as e:
                print(f"      Error reading {txt_file.name}: {e}")
        print(f"      Ingested {count:,} chunks from {len(list(books_txt_path.glob('*.txt')))} classic books.")

    # -------------------------------------------------------------
    # 5. Curriculum Textbooks (Math, Science, Philosophy, etc.)
    # Note: Strictly NO PDFs - only .txt files as instructed
    # -------------------------------------------------------------
    ebalbharati_path = Path(r"D:\book making\llm_knowledge_expansion\Data\ebalbharati")
    if ebalbharati_path.is_dir():
        print(f"\n[5/6] Ingesting Subject Curriculum Textbooks from ebalbharati (NO PDFs)...")
        count = 0
        source_dirs = [
            ebalbharati_path / "cleaned_text",
            ebalbharati_path / "english_only",
        ]
        for sdir in source_dirs:
            if sdir.is_dir():
                for txt_file in sorted(sdir.glob("*.txt")):
                    dest = CURRICULUM_DIR / txt_file.name
                    if not dest.is_file():
                        shutil.copy2(txt_file, dest)
                    try:
                        content = txt_file.read_text(encoding="utf-8", errors="replace")
                        # Classify subject hint from filename
                        subject = "General Curriculum"
                        name_lower = txt_file.name.lower()
                        if "गण_त" in name_lower or "math" in name_lower:
                            subject = "Mathematics"
                        elif "स_यन_स" in name_lower or "science" in name_lower:
                            subject = "Science & Technology"
                        elif "तत_वज_ञ_न" in name_lower or "philosophy" in name_lower:
                            subject = "Philosophy & Logic"
                        elif "भ_ग_ल" in name_lower or "geography" in name_lower:
                            subject = "Geography & Earth Sciences"
                        elif "र_ज_यश_स_त_र" in name_lower or "politic" in name_lower:
                            subject = "Political Science & Civics"
                        elif "सम_जश_स_त_र" in name_lower or "sociology" in name_lower:
                            subject = "Sociology"

                        paragraphs = content.split("\n\n")
                        cur_chunk = []
                        c_len = 0
                        for p in paragraphs:
                            cur_chunk.append(p)
                            c_len += len(p)
                            if c_len >= 3000:
                                doc = f"### [Curriculum Subject: {subject} | Book: {txt_file.stem}]\n\n" + "\n\n".join(cur_chunk)
                                write_doc(doc, domain="curriculum")
                                count += 1
                                cur_chunk = []
                                c_len = 0
                        if cur_chunk:
                            doc = f"### [Curriculum Subject: {subject} | Book: {txt_file.stem}]\n\n" + "\n\n".join(cur_chunk)
                            write_doc(doc, domain="curriculum")
                            count += 1
                    except Exception as e:
                        pass
        print(f"      Ingested {count:,} textbook chunks (Math, Science, Logic, Social Studies).")

    # -------------------------------------------------------------
    # 6. Conversational, Step-by-Step Reasoning & Agentic Coding
    # -------------------------------------------------------------
    datasets_path = Path(r"D:\book making\llm_knowledge_expansion\datasets")
    if datasets_path.is_dir():
        print(f"\n[6/6] Ingesting Reasoning, Coding & SFT Conversations...")
        sft_files = [
            (datasets_path / "extra_sources" / "source_cache" / "slimorca.jsonl", "reasoning_slimorca"),
            (datasets_path / "extra_sources" / "source_cache" / "coding.jsonl", "agentic_coding"),
            (datasets_path / "conversations" / "chat_sft.jsonl", "conversations_dolly"),
        ]
        for sf, dom in sft_files:
            if sf.is_file():
                dest = REASONING_DIR / sf.name
                shutil.copy2(sf, dest)
                count = 0
                with open(sf, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        doc = format_messages_jsonl(line)
                        if doc:
                            write_doc(doc, domain=dom)
                            count += 1
                print(f"      Ingested {count:,} examples from {sf.name} ({dom}).")

    out_handle.close()
    elapsed = time.time() - t0
    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 68)
    print("   INGESTION & UNIFICATION COMPLETE")
    print("=" * 68)
    print(f" Master Corpus File:   {OUT_FILE.name}")
    print(f" Total File Size:      {size_mb:.2f} MB")
    print(f" Total Unique Chunks:  {total_docs:,} documents")
    print(f" Total Characters:     {total_chars:,} characters")
    print(f" Execution Elapsed:    {elapsed:.2f} seconds")
    print("-" * 68)
    print(" Domain Breakdown:")
    for dom, cnt in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {dom:22}: {cnt:,} documents")
    print("=" * 68)

    # Save Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_file": str(OUT_FILE),
        "size_mb": round(size_mb, 2),
        "total_documents": total_docs,
        "total_characters": total_chars,
        "domains": domain_counts,
    }
    with open(TRAINING_DATA_DIR / "ingestion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
