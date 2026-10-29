#!/usr/bin/env python3
"""
Convert user-provided PDFs to text, clean and structure them,
and ingest them into the knowledge base, curriculum dataset, and token cache.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

# Environmental thread protections for Windows CPU
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

import pypdf
from src.data.manifest import DatasetManifest, DocumentRecord
from src.data.pipeline import CorpusPipeline
from src.data.token_cache import CompactTokenCache


PDF_SPECS = [
    {
        "source": Path(r"C:\Users\YUVRAJ\Downloads\Learning_Python.pdf"),
        "dest_rel": Path("books/python/Learning_Python.txt"),
        "curriculum_rel": None,
        "title": "Learning Python",
        "domain": "programming/python",
    },
    {
        "source": Path(r"C:\Users\YUVRAJ\Downloads\1103010502.pdf"),
        "dest_rel": Path("books/geology/Geology_Class11_The_Earth_Matters.txt"),
        "curriculum_rel": Path("grade_11/1103010502_Geology_Class11_The_Earth_Matters.txt"),
        "title": "NCERT Class 11 Geology - The Earth Matters",
        "domain": "science/geology",
    },
    {
        "source": Path(r"C:\Users\YUVRAJ\Downloads\the-complete-reference-html-css-fifth-edition.pdf"),
        "dest_rel": Path("books/web_development/HTML_CSS_The_Complete_Reference.txt"),
        "curriculum_rel": None,
        "title": "HTML & CSS: The Complete Reference (Fifth Edition)",
        "domain": "programming/web_development",
    },
    {
        "source": Path(r"C:\Users\YUVRAJ\Downloads\javascript.pdf"),
        "dest_rel": Path("books/javascript/JavaScript_Reference.txt"),
        "curriculum_rel": None,
        "title": "JavaScript Reference Guide",
        "domain": "programming/javascript",
    },
]


def clean_page_text(text: str) -> str:
    """Cleans extracted page text while preserving code blocks and math symbols."""
    if not text:
        return ""
    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking space and other unicode space variants
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Normalize common quotation marks
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Remove control characters except tab and newline
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Collapse 4+ consecutive empty lines into 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def extract_pdf_streaming(source_path: Path, dest_path: Path) -> dict:
    """Streams pages from PDF and writes clean text to destination."""
    t0 = time.time()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    reader = pypdf.PdfReader(str(source_path))
    total_pages = len(reader.pages)
    non_empty_pages = 0
    total_chars = 0

    print(f"\n[EXTRACTING] {source_path.name}")
    print(f"  Source: {source_path.resolve()} ({source_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Target: {dest_path.resolve()}")
    print(f"  Pages:  {total_pages}")

    with dest_path.open("w", encoding="utf-8") as out_f:
        for idx, page in enumerate(reader.pages):
            try:
                page_raw = page.extract_text() or ""
                cleaned = clean_page_text(page_raw)
                if cleaned:
                    non_empty_pages += 1
                    total_chars += len(cleaned)
                    out_f.write(f"\n--- [Page {idx + 1}] ---\n")
                    out_f.write(cleaned)
                    out_f.write("\n")
            except Exception as e:
                print(f"    Warning: page {idx+1} extract failed: {e}")

            if (idx + 1) % 100 == 0 or (idx + 1) == total_pages:
                print(f"    Progress: page {idx + 1}/{total_pages} ({non_empty_pages} non-empty)...")

    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.2f}s: {non_empty_pages}/{total_pages} pages, {total_chars:,} chars ({total_chars / (1024*1024):.2f} MB).")

    return {
        "pages": total_pages,
        "non_empty_pages": non_empty_pages,
        "chars": total_chars,
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 65)
    print("PDF CONVERSION & KNOWLEDGE INGESTION PIPELINE")
    print("=" * 65)

    knowledge_dir = PROJECT_ROOT / "data_knowledge"
    curriculum_dir = PROJECT_ROOT / "data_curriculum"
    corpus_file = PROJECT_ROOT / "data_knowledge_corpus.txt"
    manifest_file = PROJECT_ROOT / "manifest_knowledge_expansion.json"
    tokenizer_file = PROJECT_ROOT / "tokenizer_bpe.json"

    # Step 1: Verify all source files exist
    for spec in PDF_SPECS:
        src = spec["source"]
        if not src.is_file():
            raise FileNotFoundError(f"Source PDF not found: {src}")

    # Step 2: Convert each PDF
    conversion_results = []
    for spec in PDF_SPECS:
        src = spec["source"]
        dest = knowledge_dir / spec["dest_rel"]
        stats = extract_pdf_streaming(src, dest)
        stats["spec"] = spec
        stats["dest_path"] = dest
        conversion_results.append(stats)

        # Copy to curriculum folder if designated
        if spec["curriculum_rel"]:
            curr_dest = curriculum_dir / spec["curriculum_rel"]
            curr_dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(dest, curr_dest)
            print(f"  [CURRICULUM] Copied to: {curr_dest.resolve()}")

    # Step 3: Append new books to data_knowledge_corpus.txt
    print("\n" + "=" * 65)
    print("INGESTING EXTRACTED TEXT INTO UNIFIED TRAINING CORPUS")
    print("=" * 65)

    initial_corpus_size = corpus_file.stat().st_size if corpus_file.is_file() else 0
    appended_chars = 0

    with corpus_file.open("a", encoding="utf-8") as corpus_f:
        for res in conversion_results:
            spec = res["spec"]
            dest = res["dest_path"]
            text = dest.read_text(encoding="utf-8", errors="replace")

            # Document header
            corpus_f.write(f"\n\n--- [DOCUMENT: {spec['dest_rel'].as_posix()}] ---\n")
            corpus_f.write(f"--- [TITLE: {spec['title']}] ---\n")
            corpus_f.write(f"--- [DOMAIN: {spec['domain']}] ---\n\n")
            corpus_f.write(text)
            corpus_f.write("\n")
            appended_chars += len(text)
            print(f"  • Ingested: {spec['title']} ({len(text):,} chars)")

    final_corpus_size = corpus_file.stat().st_size
    print(f"\nCorpus updated: {initial_corpus_size / (1024*1024):.2f} MB -> {final_corpus_size / (1024*1024):.2f} MB (+{appended_chars / (1024*1024):.2f} MB)")

    # Step 4: Update Manifest
    print("\nUpdating dataset manifest...")
    manifest = DatasetManifest.load(manifest_file)
    for res in conversion_results:
        spec = res["spec"]
        dest = res["dest_path"]
        text = dest.read_text(encoding="utf-8", errors="replace")
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()

        rec = DocumentRecord(
            doc_id=f"doc_{dest.stem}_{h[:8]}",
            filename=str(spec["dest_rel"].as_posix()),
            domain=spec["domain"],
            char_count=len(text),
            fingerprint=h[:16],
            split="train",
        )
        manifest.add_document(rec)

    manifest.save(manifest_file)
    summary = manifest.get_summary()
    print(f"Manifest updated: {summary['total_documents']} total documents across {len(summary['domains'])} domains.")

    # Step 5: Rebuild Compact Token Cache
    print("\n" + "=" * 65)
    print("UPDATING COMPACT UINT16 TOKEN CACHE")
    print("=" * 65)
    cache = CompactTokenCache.build_or_load(
        data_path=corpus_file,
        tokenizer_path=tokenizer_file,
    )
    print(f"Token cache successfully synchronized: {cache.token_count:,} tokens.")
    cache.close()

    print("\n" + "=" * 65)
    print("ALL 4 BOOKS SUCCESSFULLY CONVERTED AND INGESTED")
    print("=" * 65)


if __name__ == "__main__":
    main()

