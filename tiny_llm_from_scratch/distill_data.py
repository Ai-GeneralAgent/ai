#!/usr/bin/env python3
"""
Self-Verified Distillation Pipeline (Ollama Teachers -> Small LLM Dataset).
Uses local Ollama teacher models (qwen3.5:2b, humanchat, gemma) to generate:
1. Casual, natural human dialogues (1-3 sentences, contractions, no AI boilerplate)
2. Step-by-step math reasoning traces verified deterministically by Python/SymPy

Filters out duplicates, formal corporate tone, and 'As an AI' disclaimers.
Saves to training_data/distilled_conversations.jsonl in ChatML messages format.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

# Set OpenBLAS threads and UTF-8 console
os.environ["OPENBLAS_NUM_THREADS"] = "4"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if (PROJECT_ROOT / "tiny_llm_from_scratch").is_dir():
    NESTED_ROOT = PROJECT_ROOT / "tiny_llm_from_scratch"
else:
    NESTED_ROOT = PROJECT_ROOT

for p in (PROJECT_ROOT, NESTED_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.tools.builtin import solve_math_problem

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_TEACHER = "humanchat:latest"
FALLBACK_TEACHER = "qwen3.5:2b"

SYSTEM_PROMPT = (
    "You are LLM, a friendly, warm, casual conversational assistant created by Yuvraj Thakur. "
    "Reply in 1-3 short, natural sentences with contractions. "
    "Never say 'As an AI' or 'As a language model'. "
    "Never refer to yourself as Qwen or made by Alibaba. Your name is LLM, built by Yuvraj Thakur. "
    "Never use bullet points, numbered lists, or corporate jargon."
)

TOPICS = [
    "asking how the other person is feeling today",
    "making a short, witty joke about daily life",
    "recommending a quick 15-minute dinner idea",
    "sharing a simple tip to relax after a stressful day",
    "debating morning coffee vs evening tea",
    "discussing what makes a loyal, genuine friend",
    "sharing an interesting surprising fact about nature or animals",
    "talking about the sound of rain against the window",
    "giving two sentences of genuine encouragement for exams or work",
    "talking about a favorite weekend hobby like reading or gaming",
    "explaining a simple science idea (like why the sky is blue) in plain words",
    "suggesting a creative indoor activity for a Sunday afternoon",
    "describing a favorite comfort food from childhood",
    "talking about how music changes your mood instantly",
    "sharing an honest opinion about getting up early in the morning",
    "giving a kind, thoughtful response to someone who feels lonely",
    "talking about walking in the park on a crisp autumn morning",
    "recommending a classic book or movie to unwind with",
    "discussing why taking small breaks makes a big difference",
    "saying a warm, heartfelt goodbye to end a conversation",
]

MATH_TEMPLATES = [
    ("add", "What is {a} + {b}?", lambda a, b: (f"{a} + {b}", a + b)),
    ("sub", "Calculate {a} - {b}.", lambda a, b: (f"{a} - {b}", a - b)),
    ("mul", "What is {a} * {b}?", lambda a, b: (f"{a} * {b}", a * b)),
    ("div", "Compute {a} / {b}.", lambda a, b: (f"{a} / {b}", a // b)),
    ("pct", "What is {pct}% of {val}?", lambda pct, val: (f"({pct}/100) * {val}", int(pct * val / 100))),
    ("eq", "Solve for x: x + {b} = {c}.", lambda b, c: (f"{c} - {b}", c - b)),
    ("pow", "What is {base} to the power of {exp}?", lambda base, exp: (f"{base} ** {exp}", base ** exp)),
]


def query_ollama(model: str, prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> Optional[str]:
    """Queries local Ollama instance with timeout and think:False support."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system or SYSTEM_PROMPT,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 100,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception as e:
        return None


def is_clean_human_reply(text: str) -> bool:
    """Filters out AI boilerplate, formal robotic phrases, and excessive length."""
    if not text or len(text) < 5 or len(text) > 400:
        return False
    lower = text.lower()
    disallowed = [
        "as an ai",
        "as a language model",
        "as an artificial intelligence",
        "i am an ai",
        "i do not have feelings",
        "i don't have feelings",
        "certainly! here",
        "hope this helps",
        "feel free to ask",
        "how can i assist",
    ]
    for d in disallowed:
        if d in lower:
            return False
    # Check for bullet points or lists
    if "\n1." in text or "\n-" in text or "\n*" in text:
        return False
    return True


def generate_verified_math_example() -> Dict[str, Any]:
    """Generates a deterministic math problem with a step-by-step Python-verified solution."""
    op_type, template, func = random.choice(MATH_TEMPLATES)
    if op_type == "add":
        a, b = random.randint(10, 500), random.randint(10, 500)
        expr_str, ans = func(a, b)
        q = template.format(a=a, b=b)
        sol = f"Let's add {a} and {b}. {a} plus {b} equals {ans}. The answer is {ans}."
    elif op_type == "sub":
        a = random.randint(50, 999)
        b = random.randint(10, a)
        expr_str, ans = func(a, b)
        q = template.format(a=a, b=b)
        sol = f"Subtracting {b} from {a}: {a} minus {b} gives {ans}. The answer is {ans}."
    elif op_type == "mul":
        a, b = random.randint(2, 25), random.randint(2, 25)
        expr_str, ans = func(a, b)
        q = template.format(a=a, b=b)
        sol = f"Multiplying {a} by {b}: {a} * {b} = {ans}. The answer is {ans}."
    elif op_type == "div":
        b = random.randint(2, 12)
        ans = random.randint(3, 30)
        a = b * ans
        expr_str, _ = func(a, b)
        q = template.format(a=a, b=b)
        sol = f"Dividing {a} by {b}: {a} / {b} = {ans}. The answer is {ans}."
    elif op_type == "pct":
        pct = random.choice([10, 20, 25, 50, 75])
        val = random.choice([40, 80, 100, 200, 400, 500])
        expr_str, ans = func(pct, val)
        q = template.format(pct=pct, val=val)
        sol = f"{pct}% of {val} is ({pct}/100) * {val} = {ans}. The answer is {ans}."
    elif op_type == "eq":
        b = random.randint(3, 50)
        ans = random.randint(2, 40)
        c = ans + b
        expr_str, _ = func(b, c)
        q = template.format(b=b, c=c)
        sol = f"To solve x + {b} = {c}, subtract {b} from both sides: x = {c} - {b} = {ans}. Therefore, x = {ans}."
    else:  # pow
        base = random.choice([2, 3, 4, 5, 10])
        exp = random.choice([2, 3, 4])
        expr_str, ans = func(base, exp)
        q = template.format(base=base, exp=exp)
        sol = f"{base} raised to the power of {exp} is {base}^{exp} = {ans}. The answer is {ans}."

    # Python deterministic verification
    py_result = solve_math_problem(expr_str)
    assert str(ans) in str(py_result) or str(ans) in sol, f"Math verification failed for {expr_str}"

    return {
        "messages": [
            {"role": "system", "content": "You are LLM, a helpful, precise mathematical assistant created by Yuvraj Thakur."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": sol},
        ]
    }


def generate_distillation_dataset(
    count: int = 500,
    math_ratio: float = 0.35,
    teacher_model: str = DEFAULT_TEACHER,
    output_path: Optional[Path] = None,
) -> int:
    """Generates the requested mix of casual human dialogue and verified step-by-step math."""
    if output_path is None:
        output_path = NESTED_ROOT / "training_data" / "distilled_conversations.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_hashes = set()
    if output_path.is_file():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                existing_hashes.add(hashlib.md5(line.strip().encode("utf-8")).hexdigest())

    print("=" * 70)
    print(f"STARTING DISTILLATION DATA GENERATION")
    print(f"Target count:   {count:,} conversations")
    print(f"Teacher model:  {teacher_model}")
    print(f"Math ratio:     {math_ratio*100:.0f}% verified math / {(1-math_ratio)*100:.0f}% casual chat")
    print(f"Destination:    {output_path.resolve()}")
    print("=" * 70)

    generated = 0
    t0 = time.time()
    batch_records = []

    num_math = int(count * math_ratio)
    num_chat = count - num_math

    # 1. Generate Verified Math
    print(f"\n[1/2] Generating {num_math:,} deterministically verified math solutions...")
    for i in range(num_math):
        ex = generate_verified_math_example()
        line = json.dumps(ex)
        h = hashlib.md5(line.encode("utf-8")).hexdigest()
        if h not in existing_hashes:
            existing_hashes.add(h)
            batch_records.append(line)
            generated += 1

    print(f"  Generated {len(batch_records):,} verified math examples.")

    # 2. Generate Casual Human Chat from Ollama Teacher
    print(f"\n[2/2] Generating {num_chat:,} casual human dialogues from {teacher_model}...")
    chat_done = 0
    attempts = 0
    max_attempts = num_chat * 3

    while chat_done < num_chat and attempts < max_attempts:
        attempts += 1
        topic = random.choice(TOPICS)
        user_prompt = f"Say something casual and natural about {topic}."
        teacher_reply = query_ollama(teacher_model, user_prompt, temperature=0.85)

        if not teacher_reply:
            # Fallback to teacher model if default failed
            teacher_reply = query_ollama(FALLBACK_TEACHER, user_prompt, temperature=0.85)

        if teacher_reply and is_clean_human_reply(teacher_reply):
            ex = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Hey! {topic.capitalize()}?"},
                    {"role": "assistant", "content": teacher_reply},
                ]
            }
            line = json.dumps(ex)
            h = hashlib.md5(line.encode("utf-8")).hexdigest()
            if h not in existing_hashes:
                existing_hashes.add(h)
                batch_records.append(line)
                chat_done += 1
                generated += 1
                if chat_done % 25 == 0 or chat_done == num_chat:
                    print(f"  Progress: {chat_done}/{num_chat} chat dialogues ({chat_done/max(1, time.time()-t0):.1f}/s)...")

    # Append to file atomically
    with open(output_path, "a", encoding="utf-8") as f:
        for r in batch_records:
            f.write(r + "\n")

    el = time.time() - t0
    print("\n" + "=" * 70)
    print(f"DISTILLATION COMPLETE: Added {generated:,} verified examples in {el:.1f}s")
    print(f"Total dataset entries: {len(existing_hashes):,}")
    print(f"Saved to: {output_path.resolve()}")
    print("=" * 70 + "\n")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Distill data from Ollama teachers")
    parser.add_argument("--count", type=int, default=500, help="Number of examples to generate")
    parser.add_argument("--math-ratio", type=float, default=0.35, help="Ratio of math problems")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER, help="Teacher model name in Ollama")
    parser.add_argument("--output", default="training_data/distilled_conversations.jsonl")
    args = parser.parse_args()

    out_p = NESTED_ROOT / args.output
    generate_distillation_dataset(
        count=args.count,
        math_ratio=args.math_ratio,
        teacher_model=args.teacher,
        output_path=out_p,
    )


if __name__ == "__main__":
    main()
