#!/usr/bin/env python3
"""Builds uint16 token cache for the master combined dataset."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.token_cache import CompactTokenCache

if __name__ == "__main__":
    data_f = PROJECT_ROOT / "data_master_combined.txt"
    tok_f = PROJECT_ROOT / "tokenizer_bpe.json"
    print("Building master token cache for:", data_f.name)
    cache = CompactTokenCache.build_or_load(data_f, tok_f)
    print("Master token cache ready:", cache.token_count, "tokens.")
    cache.close()

