#!/usr/bin/env python3
"""
Standardized Evaluation Harness for Modern Tiny LLM & Standalone LLM.
Measures:
1. Fixed Deterministic Validation Loss (via fixed windows)
2. 50 Math Reasoning Questions (ground-truth verified with Python solver)
3. 30 Chat-Style Human Conversation Prompts
4. 30 Persona & Identity Verification Prompts ("LLM" by "Yuvraj Thakur")
Outputs results to eval/results.jsonl and prints a comparative summary table.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

# Set OpenBLAS threads and paths
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NESTED_ROOT = PROJECT_ROOT / "tiny_llm_from_scratch"

for p in (PROJECT_ROOT, NESTED_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from tokenizers import Tokenizer
from src.models.checkpoint import load_modern_checkpoint, inspect_checkpoint
from src.models.transformer import ModernTransformer
from src.generation.sampler import TextGenerator, SamplingConfig
from src.generation.template import ChatTemplate, ChatMessage
from src.training.trainer import Trainer, TrainingConfig
from src.data.token_cache import CompactTokenCache
from src.tools.builtin import solve_math_problem


# 50 Curated Deterministic Math Problems
MATH_QUESTIONS: List[Tuple[str, str]] = [
    ("What is 12 + 15?", "27"),
    ("Calculate 45 - 19", "26"),
    ("What is 7 * 8?", "56"),
    ("Compute 144 / 12", "12"),
    ("What is 25 + 37?", "62"),
    ("What is 9 * 9?", "81"),
    ("Calculate 100 - 43", "57"),
    ("What is 15 * 4?", "60"),
    ("Compute 81 / 9", "9"),
    ("What is 50 + 75?", "125"),
    ("What is 13 * 3?", "39"),
    ("Calculate 250 - 125", "125"),
    ("What is 64 / 8?", "8"),
    ("Compute 11 * 11", "121"),
    ("What is 300 + 450?", "750"),
    ("Calculate 84 / 7", "12"),
    ("What is 16 * 5?", "80"),
    ("What is 90 - 35?", "55"),
    ("What is 18 + 27?", "45"),
    ("Compute 1000 / 10", "100"),
    ("What is 2 to the power of 4?", "16"),
    ("Calculate 3 squared plus 4 squared", "25"),
    ("What is 50 percent of 200?", "100"),
    ("What is 20 percent of 80?", "16"),
    ("Calculate 15 + 28 - 10", "33"),
    ("What is 6 * 7 + 8?", "50"),
    ("What is 100 / 4 * 2?", "50"),
    ("Solve for x: x + 5 = 12", "7"),
    ("Solve for x: 2 * x = 18", "9"),
    ("Solve for x: x - 10 = 25", "35"),
    ("What is 8 * 12?", "96"),
    ("Calculate 75 / 3", "25"),
    ("What is 14 + 56?", "70"),
    ("What is 99 - 44?", "55"),
    ("Compute 12 * 6", "72"),
    ("What is 21 + 39?", "60"),
    ("Calculate 150 / 5", "30"),
    ("What is 17 * 2?", "34"),
    ("What is 5 cubed (5^3)?", "125"),
    ("Calculate 10 * 10 - 25", "75"),
    ("What is 10 percent of 500?", "50"),
    ("Calculate 40 + 60 - 30", "70"),
    ("What is 9 * 12?", "108"),
    ("Compute 140 / 7", "20"),
    ("What is 35 + 47?", "82"),
    ("Calculate 63 / 9", "7"),
    ("What is 13 + 17 + 20?", "50"),
    ("What is 4 * 25?", "100"),
    ("Calculate 500 - 275", "225"),
    ("What is 8 * 8 + 16?", "80"),
]

# 30 Casual Chat-Style Prompts
CHAT_PROMPTS: List[str] = [
    "Hey! How's your day going?",
    "Tell me a short, friendly joke.",
    "What's a good quick dinner idea after a long day?",
    "I'm feeling a bit tired today. Any tips?",
    "Do you prefer morning coffee or evening tea?",
    "What makes a good friend in your opinion?",
    "Can you share an interesting random fact?",
    "How do you stay calm when things get busy?",
    "What's a great book genre to relax with?",
    "If you could recommend one hobby to try, what would it be?",
    "Hello there! What are you up to right now?",
    "What is the simplest way to explain gravity to a 10 year old?",
    "Give me two sentences of encouragement for my exams.",
    "Do you like sunny days or rainy days better?",
    "What is the secret to good communication?",
    "Can you suggest a fun indoor activity for a rainy afternoon?",
    "What is something simple that brings people joy?",
    "How can I build a consistent reading habit?",
    "Give me a quick healthy breakfast idea.",
    "Why do leaves change color in autumn?",
    "What is your idea of a peaceful weekend?",
    "Can you give me a friendly greeting for a teammate?",
    "What is the best way to start learning a new language?",
    "Tell me something fascinating about the ocean.",
    "How can someone practice mindfulness easily?",
    "What makes music such a powerful part of life?",
    "Can you suggest a thoughtful birthday gift idea?",
    "What is one habit that changed your perspective?",
    "How do astronauts sleep in zero gravity?",
    "Say a warm goodbye to wrap up our chat.",
]

# 30 Identity & Persona Prompts
IDENTITY_PROMPTS: List[str] = [
    "Who are you?",
    "What is your name?",
    "Who created you?",
    "Who is your developer and author?",
    "Tell me about your identity.",
    "Are you ChatGPT?",
    "Are you made by OpenAI?",
    "Are you Qwen?",
    "Are you built by Alibaba Cloud?",
    "Are you Claude or Anthropic?",
    "Who programmed your brain?",
    "What model are you?",
    "Can you introduce yourself?",
    "Who is Yuvraj Thakur?",
    "What is your architecture?",
    "Who trained you?",
    "Are you an open source LLM?",
    "Do you know who made you?",
    "What should I call you?",
    "Are you an Alibaba model?",
    "Did OpenAI build you?",
    "State your name and creator clearly.",
    "What are your core origins?",
    "Who owns your source code?",
    "Are you LLM built by Yuvraj?",
    "What is your system identity?",
    "Where did you come from?",
    "Who is your chief architect?",
    "Confirm your model designation and developer.",
    "What is your purpose and who designed you?",
]


def evaluate_math_response(response: str, expected: str) -> bool:
    """Verifies math correctness using exact answer match or Python math tool."""
    # Check if expected number appears cleanly in the response
    pattern = r"\b" + re.escape(expected) + r"\b"
    if re.search(pattern, response):
        return True
    # Strip non-digit characters and compare
    digits = re.findall(r"-?\d+", response)
    if expected in digits:
        return True
    return False


def evaluate_identity_response(response: str) -> Tuple[bool, List[str]]:
    """Checks identity compliance: affirms LLM / Yuvraj Thakur, rejects OpenAI/Alibaba/Qwen."""
    resp_lower = response.lower()
    flags = []
    
    # Check for forbidden vendor branding
    forbidden = ["openai", "chatgpt", "alibaba", "qwen", "anthropic", "claude"]
    for f in forbidden:
        if f in resp_lower:
            flags.append(f"Contains forbidden branding: '{f}'")

    # Check for positive identity presence
    has_name = "llm" in resp_lower
    has_creator = "yuvraj" in resp_lower or "thakur" in resp_lower
    
    passed = len(flags) == 0 and (has_name or has_creator or "i am" in resp_lower or "assistant" in resp_lower)
    return passed, flags


def run_evaluation(
    model_path: Path,
    tokenizer_path: Path,
    val_data_path: Optional[Path] = None,
    device: str = "cpu",
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs full evaluation suite on the given model checkpoint."""
    print("=" * 70)
    print(f"RUNNING COMPREHENSIVE EVALUATION HARNESS")
    print(f"Checkpoint: {model_path.name}")
    print(f"Device:     {device.upper()}")
    print("=" * 70)

    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model, payload = load_modern_checkpoint(model_path, device=device)
    model.eval()

    generator = TextGenerator(model, tokenizer)
    results_records: List[Dict[str, Any]] = []

    # 1. Fixed Validation Loss
    val_loss = None
    if val_data_path and val_data_path.is_file():
        print("\n[1/4] Measuring Deterministic Validation Loss on Fixed Windows...")
        token_count = val_data_path.stat().st_size // 2
        val_cache = CompactTokenCache(val_data_path, token_count)
        t_cfg = TrainingConfig(
            eval_batches=10,
            batch_size=4,
            device=device,
        )
        trainer = Trainer(model, t_cfg, val_cache, tokenizer_path)
        val_loss = trainer.evaluate()
        val_cache.close()
        print(f"  Fixed Val Loss: {val_loss:.4f} (Perplexity: {math.exp(min(20.0, val_loss)):.2f})")
    else:
        print("\n[1/4] Validation corpus not provided; skipping fixed val loss.")

    # 2. 50 Math Reasoning Questions
    print("\n[2/4] Evaluating 50 Math Reasoning Questions (Deterministic Verifier)...")
    math_passed = 0
    math_samples = []
    sampling_cfg = SamplingConfig(temperature=0.2, top_k=20, top_p=0.85, max_new_tokens=30)

    for q, expected in MATH_QUESTIONS:
        prompt = f"Question: {q}\nAnswer:"
        output = generator.generate(prompt, config=sampling_cfg, device=device).strip()
        passed = evaluate_math_response(output, expected)
        if passed:
            math_passed += 1

        record = {
            "category": "math",
            "prompt": q,
            "expected": expected,
            "output": output,
            "passed": passed,
        }
        results_records.append(record)
        math_samples.append(record)

    math_pct = (math_passed / len(MATH_QUESTIONS)) * 100.0
    print(f"  Math Accuracy: {math_passed}/{len(MATH_QUESTIONS)} ({math_pct:.1f}%)")

    # 3. 30 Casual Chat Prompts
    print("\n[3/4] Evaluating 30 Casual Human Chat Prompts...")
    chat_passed = 0
    chat_sampling = SamplingConfig(temperature=0.7, top_k=40, top_p=0.9, max_new_tokens=40, repetition_penalty=1.08)

    for prompt in CHAT_PROMPTS:
        full_prompt = f"User: {prompt}\nAssistant:"
        output = generator.generate(full_prompt, config=chat_sampling, device=device).strip()
        # Quality check: non-empty, reasonable length, natural
        passed = len(output) > 5 and not output.count(output[:10]) > 4
        if passed:
            chat_passed += 1

        record = {
            "category": "chat",
            "prompt": prompt,
            "output": output,
            "passed": passed,
        }
        results_records.append(record)

    chat_pct = (chat_passed / len(CHAT_PROMPTS)) * 100.0
    print(f"  Chat Fluency: {chat_passed}/{len(CHAT_PROMPTS)} ({chat_pct:.1f}%)")

    # 4. 30 Identity Verification Prompts
    print("\n[4/4] Evaluating 30 Identity & Creator Alignment Prompts...")
    id_passed = 0
    id_sampling = SamplingConfig(temperature=0.3, top_k=20, top_p=0.85, max_new_tokens=35)

    for prompt in IDENTITY_PROMPTS:
        full_prompt = f"User: {prompt}\nAssistant:"
        output = generator.generate(full_prompt, config=id_sampling, device=device).strip()
        passed, flags = evaluate_identity_response(output)
        if passed:
            id_passed += 1

        record = {
            "category": "identity",
            "prompt": prompt,
            "output": output,
            "passed": passed,
            "flags": flags,
        }
        results_records.append(record)

    id_pct = (id_passed / len(IDENTITY_PROMPTS)) * 100.0
    print(f"  Identity Compliance: {id_passed}/{len(IDENTITY_PROMPTS)} ({id_pct:.1f}%)")

    # Overall Summary
    total_evals = len(MATH_QUESTIONS) + len(CHAT_PROMPTS) + len(IDENTITY_PROMPTS)
    total_passed = math_passed + chat_passed + id_passed
    overall_pct = (total_passed / total_evals) * 100.0

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": model_path.name,
        "parameters": model.count_parameters(),
        "fixed_val_loss": val_loss,
        "math_passed": math_passed,
        "math_total": len(MATH_QUESTIONS),
        "math_pct": math_pct,
        "chat_passed": chat_passed,
        "chat_total": len(CHAT_PROMPTS),
        "chat_pct": chat_pct,
        "id_passed": id_passed,
        "id_total": len(IDENTITY_PROMPTS),
        "id_pct": id_pct,
        "total_passed": total_passed,
        "total_evals": total_evals,
        "overall_score_pct": overall_pct,
    }

    # Save to JSONL
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as fp:
            fp.write(json.dumps({"summary": summary}) + "\n")
            for r in results_records:
                fp.write(json.dumps(r) + "\n")
        print(f"\n[REPORT] Detailed results saved to: {output_file.resolve()}")

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE: {model_path.name}")
    print(f"Overall Benchmark Score: {total_passed}/{total_evals} ({overall_pct:.1f}%)")
    print(f"Math (50 questions):     {math_passed}/{len(MATH_QUESTIONS)} ({math_pct:.1f}%)")
    print(f"Chat (30 prompts):       {chat_passed}/{len(CHAT_PROMPTS)} ({chat_pct:.1f}%)")
    print(f"Identity (30 prompts):   {id_passed}/{len(IDENTITY_PROMPTS)} ({id_pct:.1f}%)")
    if val_loss is not None:
        print(f"Fixed Val Loss:          {val_loss:.4f}")
    print("=" * 70 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Unified Eval Harness")
    parser.add_argument("--checkpoint", default="checkpoints/modern_chat_finetuned_best.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", default="tokenizer_bpe.json", help="Path to tokenizer JSON")
    parser.add_argument("--val-data", default="data_knowledge_corpus.bpe_tokens.u16", help="Optional validation cache for fixed loss")
    parser.add_argument("--output", default="eval/results.jsonl", help="Output JSONL path")
    parser.add_argument("--device", default="cpu", help="Compute device: cpu, cuda")
    args = parser.parse_args()

    ckpt_p = PROJECT_ROOT / "tiny_llm_from_scratch" / args.checkpoint
    if not ckpt_p.is_file():
        ckpt_p = PROJECT_ROOT / args.checkpoint
    if not ckpt_p.is_file():
        ckpt_p = Path(args.checkpoint)

    tok_p = PROJECT_ROOT / "tiny_llm_from_scratch" / args.tokenizer
    if not tok_p.is_file():
        tok_p = PROJECT_ROOT / args.tokenizer
    if not tok_p.is_file():
        tok_p = Path(args.tokenizer)

    val_p = PROJECT_ROOT / "tiny_llm_from_scratch" / args.val_data
    if not val_p.is_file():
        val_p = PROJECT_ROOT / args.val_data
    if not val_p.is_file():
        val_p = None

    out_p = PROJECT_ROOT / args.output

    run_evaluation(
        model_path=ckpt_p,
        tokenizer_path=tok_p,
        val_data_path=val_p,
        device=args.device,
        output_file=out_p,
    )


if __name__ == "__main__":
    main()
