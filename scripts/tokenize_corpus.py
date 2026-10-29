#!/usr/bin/env python3
"""
CLI Script: Chunked, Memory-Efficient Tokenizer Cache Builder (Section 6)
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tiny_llm_from_scratch"))

from tokenizers import Tokenizer
from src.data.token_cache import CompactTokenCache
from src.utils import configure_utf8_console

configure_utf8_console()


def main():
    parser = argparse.ArgumentParser(description="Build Memory-Mapped Compact uint16 Token Cache")
    parser.add_argument("--corpus", default="alice.txt", help="Input corpus file")
    parser.add_argument("--tokenizer", default="tokenizer_bpe.json", help="Trained BPE tokenizer file")
    parser.add_argument("--chunk-chars", type=int, default=500_000, help="Chars per streaming chunk")
    args = parser.parse_args()

    tok_path = PROJECT_ROOT / "tiny_llm_from_scratch" / args.tokenizer
    if not tok_path.is_file():
        tok_path = Path(args.tokenizer)
    if not tok_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    corpus_path = PROJECT_ROOT / "tiny_llm_from_scratch" / args.corpus
    if not corpus_path.is_file():
        corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"[TOKENIZE] Building compact token cache for {corpus_path.name}...")
    cache = CompactTokenCache.build_or_load(corpus_path, tok_path, tokenizer=tokenizer, chunk_size_chars=args.chunk_chars)
    print(f"[SUCCESS] Token cache ready: {cache.cache_path.name}")
    print(f"Total Tokens: {cache.token_count:,}")
    cache.close()


if __name__ == "__main__":
    main()

