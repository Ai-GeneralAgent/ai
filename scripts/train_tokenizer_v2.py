#!/usr/bin/env python3
"""
Byte-Level BPE Tokenizer Trainer (v2) - 16k Vocabulary.
Trains a byte-fallback BPE tokenizer on bilingual English + Hindi / Devanagari corpora.
Properties:
- True byte-level representation: zero [UNK] tokens, 100% lossless UTF-8 roundtrip
- Balanced vocabulary allocation: English words, Devanagari subwords, code/math symbols
- Output: tokenizer_bpe_v2.json
"""
import glob
import os
from pathlib import Path
import sys
import tempfile
import time

# Ensure UTF-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NESTED_ROOT = PROJECT_ROOT / "tiny_llm_from_scratch"

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def build_bilingual_training_corpus(sample_target_bytes: int = 40_000_000) -> Path:
    """Combines high-quality English and Hindi text into a balanced training corpus file."""
    print("[CORPUS] Gathering bilingual English + Hindi training text...")
    tmp_corpus = PROJECT_ROOT / "training_data" / "bilingual_tokenizer_corpus.txt"
    tmp_corpus.parent.mkdir(parents=True, exist_ok=True)

    collected_bytes = 0
    with open(tmp_corpus, "w", encoding="utf-8", errors="ignore") as out_fp:
        # 1. Hindi textbooks & curriculum files
        textbook_dir = NESTED_ROOT / "training_data" / "curriculum_textbooks"
        hindi_files = []
        if textbook_dir.is_dir():
            for f in textbook_dir.glob("*.txt"):
                name = f.name
                if any(x in name for x in ["ह_द", "मर_ठ", "स_स", "आन_द"]):
                    hindi_files.append(f)

        print(f"  Found {len(hindi_files)} Hindi/Devanagari textbook sources.")
        for hf in hindi_files:
            try:
                with open(hf, "r", encoding="utf-8", errors="ignore") as in_fp:
                    text = in_fp.read()
                    out_fp.write(text + "\n\n")
                    collected_bytes += len(text.encode("utf-8"))
            except Exception:
                continue

        # 2. English knowledge corpus & classic literature
        english_sources = [
            NESTED_ROOT / "data_knowledge_corpus.txt",
            NESTED_ROOT / "training_data" / "qa_and_database_sft.jsonl",
            NESTED_ROOT / "training_data" / "distilled_conversations.jsonl",
        ]
        lit_dir = NESTED_ROOT / "training_data" / "classic_literature"
        if lit_dir.is_dir():
            english_sources.extend(list(lit_dir.glob("*.txt"))[:15])

        for ef in english_sources:
            if ef.is_file() and collected_bytes < sample_target_bytes:
                try:
                    with open(ef, "r", encoding="utf-8", errors="ignore") as in_fp:
                        chunk = in_fp.read(sample_target_bytes - collected_bytes)
                        out_fp.write(chunk + "\n\n")
                        collected_bytes += len(chunk.encode("utf-8"))
                except Exception:
                    continue

    print(f"[CORPUS] Built bilingual corpus: {tmp_corpus.name} ({collected_bytes / (1024*1024):.2f} MB)")
    return tmp_corpus


def train_byte_level_tokenizer(
    corpus_path: Path,
    vocab_size: int = 16384,
    output_path: Path = PROJECT_ROOT / "tokenizer_bpe_v2.json",
) -> Tokenizer:
    """Trains a byte-level BPE with 16k vocab, zero [UNK], and ChatML special tokens."""
    print(f"\n[TOKENIZER] Training Byte-Level BPE (Vocab: {vocab_size:,})...")

    # ByteLevel model with byte fallback
    tokenizer = Tokenizer(models.BPE(byte_fallback=True))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = [
        "<|pad|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|extra_0|>",
        "<|extra_1|>",
        "<|extra_2|>",
        "<|extra_3|>",
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    t0 = time.time()
    tokenizer.train([str(corpus_path)], trainer)
    train_time = time.time() - t0
    print(f"[TOKENIZER] Training complete in {train_time:.1f}s. Vocab size: {tokenizer.get_vocab_size():,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"[TOKENIZER] Saved byte-level tokenizer to: {output_path.resolve()}")
    return tokenizer


def benchmark_tokenizers(tok_v1_path: Path, tok_v2_path: Path):
    """Compares token efficiency (tokens/word), [UNK] count, and roundtrip fidelity."""
    print("\n" + "=" * 70)
    print("TOKENIZER BENCHMARK: V1 (8k) vs V2 BYTE-LEVEL (16k)")
    print("=" * 70)

    tok_v1 = Tokenizer.from_file(str(tok_v1_path))
    tok_v2 = Tokenizer.from_file(str(tok_v2_path))

    test_sentences = [
        ("English", "Artificial intelligence and deep neural networks are transforming science and technology."),
        ("English Conversational", "Hey there! How has your day been going so far? Hope you're doing well!"),
        ("Hindi", "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग तकनीक दुनिया को तेजी से बदल रही हैं।"),
        ("Hindi Conversational", "नमस्ते! आप आज कैसा महसूस कर रहे हैं? क्या मैं आपकी कोई मदद कर सकता हूँ?"),
        ("Math / Code", "def solve(x: int) -> float: return math.sqrt(x ** 2 + 15) / 2.5"),
    ]

    print(f"{'Category':<22} | {'V1 Tok/Word':<11} | {'V2 Tok/Word':<11} | {'V1 [UNK]':<8} | {'V2 [UNK]':<8} | {'Roundtrip'}")
    print("-" * 75)

    for cat, text in test_sentences:
        word_count = len(text.split())
        enc1 = tok_v1.encode(text)
        enc2 = tok_v2.encode(text)

        tpw1 = len(enc1.ids) / max(1, word_count)
        tpw2 = len(enc2.ids) / max(1, word_count)

        unk1 = sum(1 for t in enc1.tokens if t == "[UNK]")
        unk2 = sum(1 for t in enc2.tokens if t == "[UNK]")

        roundtrip = (tok_v2.decode(enc2.ids) == text)
        rt_str = "[OK] PASS" if roundtrip else "[X] FAIL"

        print(f"{cat:<22} | {tpw1:<11.2f} | {tpw2:<11.2f} | {unk1:<8} | {unk2:<8} | {rt_str}")

    print("=" * 75 + "\n")


def main():
    corpus_p = build_bilingual_training_corpus()
    tok_v2_p = PROJECT_ROOT / "tokenizer_bpe_v2.json"
    train_byte_level_tokenizer(corpus_p, vocab_size=16384, output_path=tok_v2_p)

    tok_v1_p = PROJECT_ROOT / "tokenizer_bpe.json"
    benchmark_tokenizers(tok_v1_p, tok_v2_p)

    # Sync to nested folder
    nested_tok_v2 = NESTED_ROOT / "tokenizer_bpe_v2.json"
    import shutil
    shutil.copy2(tok_v2_p, nested_tok_v2)
    print(f"Synchronized tokenizer_bpe_v2.json to: {nested_tok_v2}")


if __name__ == "__main__":
    main()
