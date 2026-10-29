#!/usr/bin/env python3
"""
Expands BPE vocabulary from 8,000 to 10,000 tokens:
1. Adds reasoning and thinking special tokens (<think>, </think>, etc.)
2. Adds programming and markup syntax (Python, JS, HTML, CSS)
3. Adds mathematical and scientific LaTeX symbols and domain terms
4. Mines high-frequency composite subwords from the master corpus
5. Preserves all existing token IDs 0..7999 with 100% backward compatibility.
"""
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer


# Domain-specific essential tokens to guarantee in vocabulary
REASONING_TOKENS = [
    "<think>", "</think>", "<thought>", "</thought>",
    "<reasoning>", "</reasoning>", "<solution>", "</solution>",
    "<step>", "</step>", "<math>", "</math>", "<code>", "</code>",
    "<system>", "</system>", "<user>", "</user>", "<assistant>", "</assistant>",
    "<context>", "</context>", "<source>", "</source>",
]

CODE_TOKENS = [
    "def ", "return ", "import ", "from ", "class ", "const ", "let ", "var ",
    "function ", "console.log", "undefined", "nullptr", "async ", "await ",
    "lambda ", "elif ", "self.", "__init__", "None", "True", "False",
    "print(", "len(", "range(", "int(", "str(", "list(", "dict(", "set(",
    "<div>", "</div>", "<span>", "</span>", "<p>", "</p>", "<a>", "</a>",
    "<ul>", "</ul>", "<li>", "</li>", "<style>", "</style>", "<html>", "</html>",
    "margin:", "padding:", "display:", "flex", "color:", "font-size:", "border:",
    "width:", "height:", "background:", "position:", "relative", "absolute",
    "px;", "em;", "rem;", "%;", "-->", "<!--",
]

MATH_SCIENCE_TOKENS = [
    r"\frac", r"\sum", r"\int", r"\partial", r"\infty", r"\sqrt",
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\theta", r"\lambda", r"\pi", r"\sigma",
    r"\approx", r"\neq", r"\leq", r"\geq", r"\pm", r"\times", r"\div",
    "equation", "theorem", "derivative", "integral", "matrix", "vector",
    "gravitational", "acceleration", "velocity", "momentum", "wavelength",
    "photosynthesis", "weathering", "pedology", "geothermal", "drumlin",
    "metamorphic", "sedimentary", "equigranular", "mineralogy", "crystallization",
]


def expand_vocabulary(
    base_tokenizer_path: Path,
    corpus_path: Path,
    target_vocab_size: int = 10000,
    output_path: Path = None,
):
    print("=" * 65)
    print("EXPANDING BPE VOCABULARY TO TARGET SIZE:", target_vocab_size)
    print("=" * 65)

    if output_path is None:
        output_path = base_tokenizer_path

    # Step 1: Backup existing tokenizer
    backup_path = base_tokenizer_path.with_name(f"{base_tokenizer_path.stem}_8k_backup.json")
    if not backup_path.is_file():
        shutil.copy2(base_tokenizer_path, backup_path)
        print(f"[BACKUP] Created backup: {backup_path.name}")
    else:
        print(f"[BACKUP] Existing backup verified: {backup_path.name}")

    tok = Tokenizer.from_file(str(base_tokenizer_path))
    initial_size = tok.get_vocab_size()
    print(f"Initial Vocabulary Size: {initial_size}")

    if initial_size >= target_vocab_size:
        print(f"Vocabulary already at or above target size ({initial_size} >= {target_vocab_size}).")
        return tok

    tokens_needed = target_vocab_size - initial_size
    print(f"Tokens needed to reach {target_vocab_size}: {tokens_needed}")

    # Gather explicit candidate tokens
    candidates = []
    seen = set()

    for t_list in [REASONING_TOKENS, CODE_TOKENS, MATH_SCIENCE_TOKENS]:
        for t in t_list:
            if t not in seen:
                seen.add(t)
                candidates.append(t)

    # Step 2: Mine high-frequency composite subwords from master corpus
    print(f"\nMining high-frequency subwords from: {corpus_path.name}...")
    word_counter = Counter()

    # Read a sample of 25MB from corpus for fast candidate extraction
    sample_limit = 25 * 1024 * 1024
    chars_read = 0

    with corpus_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            chars_read += len(line)
            # Find candidate programming tokens, words, symbols
            matches = re.findall(r"[A-Za-z0-9_]{3,15}|[^\w\s]{2,4}", line)
            for m in matches:
                word_counter[m] += 1
            if chars_read >= sample_limit:
                break

    print(f"Sampled {chars_read / (1024*1024):.1f} MB: {len(word_counter):,} unique patterns found.")

    # Filter out candidates that already tokenize as single tokens
    print("Selecting most frequent multi-token patterns...")
    existing_vocab = tok.get_vocab()

    for word, count in word_counter.most_common(20000):
        if len(candidates) >= tokens_needed + 500:
            break
        if word in seen or word in existing_vocab:
            continue

        # Check if currently tokenized into 2+ tokens
        encoding = tok.encode(word)
        if len(encoding.ids) >= 2 and count >= 5:
            seen.add(word)
            candidates.append(word)

    # Select exact needed amount
    tokens_to_add = []
    for c in candidates:
        if c not in existing_vocab:
            tokens_to_add.append(c)
            if len(tokens_to_add) == tokens_needed:
                break

    print(f"Adding {len(tokens_to_add)} new tokens to vocabulary...")
    tok.add_tokens(tokens_to_add)

    final_size = tok.get_vocab_size()
    print(f"Expanded Vocabulary Size: {final_size}")

    # Step 3: Verify backward compatibility
    # Ensure all original 8,000 token IDs retain their exact identical index mapping
    orig_tok = Tokenizer.from_file(str(backup_path))
    orig_vocab = orig_tok.get_vocab()
    new_vocab = tok.get_vocab()

    mismatches = 0
    for token_str, token_id in orig_vocab.items():
        if new_vocab.get(token_str) != token_id:
            mismatches += 1

    print(f"Checking original vocabulary preservation: {len(orig_vocab)} tokens...")
    assert mismatches == 0, f"ERROR: {mismatches} existing token IDs altered!"
    print(f"[VERIFIED] All {len(orig_vocab)} original token IDs are 100% preserved at exact indices.")

    # Test encoding of new tokens
    for test_new in ["<think>", "</think>", "def ", "return ", "console.log"]:
        enc = tok.encode(test_new)
        print(f"New token '{test_new}' -> IDs: {enc.ids}, Tokens: {enc.tokens}")

    # Save expanded tokenizer
    tok.save(str(output_path))
    print(f"Saved expanded tokenizer to: {output_path.resolve()}")
    print("=" * 65)
    return tok


if __name__ == "__main__":
    base_tok = PROJECT_ROOT / "tokenizer_bpe.json"
    corpus = PROJECT_ROOT / "data_master_combined.txt"
    expand_vocabulary(base_tok, corpus, target_vocab_size=10000)
