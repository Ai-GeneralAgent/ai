#!/usr/bin/env python3
"""
CLI Script: Extract, Normalize, Deduplicate, Classify, and Manifest Data (Section 5)
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tiny_llm_from_scratch"))

from src.data.pipeline import CorpusPipeline
from src.utils import configure_utf8_console

configure_utf8_console()


def main():
    parser = argparse.ArgumentParser(description="Process, Clean & Deduplicate Corpus Data")
    parser.add_argument("--source-dir", default="data_knowledge", help="Directory of source documents")
    parser.add_argument("--manifest", default="dataset_manifest.json", help="Manifest output path")
    parser.add_argument("--out-corpus", default="data_knowledge_cleaned.txt", help="Combined clean corpus text")
    args = parser.parse_args()

    s_dir = PROJECT_ROOT / "tiny_llm_from_scratch" / args.source_dir
    if not s_dir.is_dir():
        s_dir = Path(args.source_dir)

    print(f"[DATA] Scanning source directory: {s_dir.resolve()}...")
    pipeline = CorpusPipeline(manifest_path=Path(args.manifest))

    files = sorted(list(s_dir.glob("**/*.*")))
    clean_texts = []
    processed = 0

    for f in files:
        if f.suffix.lower() in [".txt", ".md", ".jsonl", ".pdf"]:
            res = pipeline.process_document(f)
            if res:
                doc_record, text = res
                clean_texts.append(text)
                processed += 1
                print(f"  [ACCEPTED] {f.name} ({doc_record.domain}, {len(text):,} chars)")
            else:
                print(f"  [EXCLUDED] {f.name}")

    if clean_texts:
        out_p = PROJECT_ROOT / "tiny_llm_from_scratch" / args.out_corpus
        out_p.write_text("\n\n\n".join(clean_texts), encoding="utf-8")
        print(f"\n[DATA] Clean corpus written to: {out_p.resolve()} ({sum(len(t) for t in clean_texts):,} chars)")

    pipeline.manifest.save(Path(args.manifest))
    print(f"[MANIFEST] Saved manifest to: {args.manifest}")


if __name__ == "__main__":
    main()
