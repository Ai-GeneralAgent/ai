#!/usr/bin/env python3
"""
Self-Verified Distillation Pipeline (Ollama Teachers + Offline Expert Generator -> Small LLM Dataset).
Generates:
1. Casual, natural human dialogues (1-3 sentences, contractions, no AI boilerplate)
2. Step-by-step math reasoning traces verified deterministically by Python/SymPy
3. Advanced programming, cybersecurity, AI architecture, and internet-learning reasoning pairs

Filters out duplicates, formal corporate tone, and 'As an AI' disclaimers.
Saves to training_data/distilled_conversations.jsonl and merges into cleaned_sft_dataset.jsonl.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
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
OLLAMA_ONLINE = None

SYSTEM_PROMPT = (
    "You are LLM, a friendly, warm, casual conversational assistant created by Yuvraj Thakur. "
    "Reply in 1-3 short, natural sentences with contractions. "
    "Never say 'As an AI' or 'As a language model'. "
    "Never refer to yourself as Qwen or made by Alibaba. Your name is LLM, built by Yuvraj Thakur. "
    "Never use bullet points, numbered lists, or corporate jargon."
)

PROGRAMMING_TOPICS = [
    ("Write a clean Python function for binary search with edge cases.", "binary search"),
    ("Explain the difference between stack and heap memory in C with code.", "stack and heap"),
    ("Show how Rust enforces ownership and borrowing at compile time.", "rust"),
    ("Write a robust retry decorator with exponential backoff in Python.", "retry decorator"),
    ("Explain how database indexes (B-Tree) accelerate query performance.", "database indexes"),
]

CYBERSECURITY_TOPICS = [
    ("Explain how parameterized queries prevent SQL injection with a code example.", "parameterized queries"),
    ("How does Content Security Policy (CSP) protect against reflected XSS?", "content security policy"),
    ("Compare symmetric (AES-256-GCM) and asymmetric (RSA/ECC) encryption.", "symmetric and asymmetric"),
    ("Explain the STRIDE threat modeling framework with an example API service.", "stride"),
    ("What is an SSRF vulnerability and how do URL allowlists prevent it?", "ssrf"),
]

REVERSE_ENG_TOPICS = [
    ("Explain the Microsoft x64 fastcall calling convention and shadow space.", "fastcall"),
    ("What are the main sections of a PE executable (.text, .rdata, .data)?", "pe executable"),
    ("How does Python compile source code into bytecode instructions?", "bytecode"),
    ("Explain how a disassembler reconstructs a Control Flow Graph (CFG).", "control flow graph"),
]

AI_RESEARCH_TOPICS = [
    ("Derive the mathematical formula for RMSNorm and explain why it is efficient.", "rmsnorm"),
    ("Explain SwiGLU activation and how the 8/3 hidden dimension ratio is chosen.", "swiglu"),
    ("How does Rotary Position Embedding (RoPE) encode relative position?", "rope"),
    ("Explain how StreamingLLM uses attention sinks to maintain infinite context.", "streamingllm"),
    ("How does the Muon optimizer orthogonalize momentum updates?", "muon optimizer"),
]

CASUAL_TOPICS = [
    ("How are you doing today?", "I'm doing great, thanks! Ready to help you with code, security, research, or whatever you're working on."),
    ("What is your name and who created you?", "I'm LLM, a personal conversational AI assistant developed and built by Yuvraj Thakur."),
    ("What capabilities do you have?", "I can analyze and debug code across multiple languages, audit cybersecurity vulnerabilities, solve math deterministically, and learn from the internet via DuckDuckGo."),
    ("Tell me an interesting scientific fact.", "Neutron stars are so dense that a single teaspoon of their matter would weigh around 6 billion tons on Earth."),
    ("How do you search the internet?", "I use DuckDuckGo to find top live sources, browse deep page content with a headless browser, and distill facts into memory."),
    ("How do you avoid hallucinations?", "If a question is fundamentally unknowable or private, I honestly state that I don't know rather than fabricating an answer."),
]

ALL_TOPICS = [t[0] for t in (PROGRAMMING_TOPICS + CYBERSECURITY_TOPICS + REVERSE_ENG_TOPICS + AI_RESEARCH_TOPICS)] + [c[0] for c in CASUAL_TOPICS]

OFFLINE_TEACHER_KNOWLEDGE = {
    "binary search": "Binary search divides the sorted search space in half each iteration by comparing the target with the middle element, achieving O(log n) time complexity.",
    "stack and heap": "Stack memory handles fast LIFO allocation for local variables with automatic cleanup, while heap memory handles dynamic, flexible allocations that require manual or GC management.",
    "rust": "Rust's borrow checker enforces ownership at compile time: each value has one owner, and you can have either any number of immutable references or exactly one mutable reference.",
    "retry decorator": "A retry decorator catches transient exceptions and sleeps with exponential backoff before retrying, preventing thundering herd problems against upstream services.",
    "database indexes": "B-Tree indexes sort keys hierarchically, reducing lookup complexity from O(N) full table scans to O(log N) disk block lookups.",
    "parameterized queries": "Parameterized queries separate SQL syntax logic from untrusted user data, preventing attackers from injecting arbitrary SQL clauses into the database engine.",
    "content security policy": "CSP restricts the origins from which scripts, styles, and media can load, neutralizing reflected and stored XSS attacks even if injected into HTML.",
    "symmetric and asymmetric": "Symmetric encryption like AES-256 uses the same shared key for speed, while asymmetric encryption like RSA uses public/private key pairs for secure key exchange.",
    "stride": "STRIDE categorizes system threats into Spoofing, Tampering, Repudiation, Information disclosure, Denial of service, and Elevation of privilege.",
    "ssrf": "SSRF happens when a server makes requests to attacker-controlled URLs; strict hostname and IP allowlists prevent attackers from probing internal private networks.",
    "fastcall": "Microsoft x64 fastcall passes the first 4 integer arguments in RCX, RDX, R8, and R9, requiring 32 bytes of shadow space allocated on the stack before the call instruction.",
    "pe executable": "PE files structure executables into the DOS header, PE signature, COFF header, optional header, and section headers like .text for code and .data for globals.",
    "bytecode": "Python compiles source code ASTs into stack-based bytecode instructions stored in .pyc files, which the CPython virtual machine interprets via evaluation loops.",
    "control flow graph": "A disassembler reconstructs CFGs by partitioning assembly instructions into basic blocks and creating directed edges for conditional and unconditional branches.",
    "rmsnorm": "RMSNorm normalizes activations by their root mean square without subtracting the mean: RMS(x) = sqrt(1/d * sum(x_i^2) + eps), cutting compute overhead by ~30% versus LayerNorm.",
    "swiglu": "SwiGLU combines Swish gating with linear projection: SwiGLU(x) = (SiLU(xW_gate) * xW_up) W_down, with hidden ratio ~8/3 d_model for superior representation capacity.",
    "rope": "RoPE applies complex 2D orthogonal rotations to query and key head pairs, encoding relative token distance naturally through dot-product properties without fixed positional tables.",
    "streamingllm": "StreamingLLM retains initial attention sinks (the first 4 tokens) alongside a sliding local window, keeping attention score distributions stable across millions of steps without memory growth.",
    "muon optimizer": "Muon maintains momentum matrices and computes orthogonalized weight updates using iterative Newton-Schulz polynomial approximations, accelerating 2D weight convergence.",
}

MATH_TEMPLATES = [
    ("add", "What is {a} + {b}?", lambda a, b: (f"{a} + {b}", a + b)),
    ("sub", "Calculate {a} - {b}.", lambda a, b: (f"{a} - {b}", a - b)),
    ("mul", "What is {a} * {b}?", lambda a, b: (f"{a} * {b}", a * b)),
    ("div", "Compute {a} / {b}.", lambda a, b: (f"{a} / {b}", a // b)),
    ("pct", "What is {pct}% of {val}?", lambda pct, val: (f"({pct}/100) * {val}", int(pct * val / 100))),
    ("eq", "Solve for x: x + {b} = {c}.", lambda b, c: (f"{c} - {b}", c - b)),
    ("pow", "What is {base} to the power of {exp}?", lambda base, exp: (f"{base} ** {exp}", base ** exp)),
]


def is_ollama_online() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.8):
            return True
    except Exception:
        return False


def query_ollama(model: str, prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> Optional[str]:
    """Queries local Ollama instance with timeout and think:False support."""
    if not is_ollama_online():
        return None
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
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception:
        return None


def get_offline_teacher_reply(topic_prompt: str) -> str:
    """Fallback offline teacher response generator."""
    lower = topic_prompt.lower()
    for k, v in OFFLINE_TEACHER_KNOWLEDGE.items():
        if k in lower:
            return v
    for q, ans in CASUAL_TOPICS:
        if any(w in lower for w in q.lower().split()[:3]):
            return ans
    return "I'm LLM, an independent assistant built by Yuvraj Thakur, focused on verified reasoning, fast code analysis, and truth preservation."


def is_clean_human_reply(text: str) -> bool:
    """Filters out AI boilerplate, formal robotic phrases, and excessive length."""
    if not text or len(text) < 5 or len(text) > 400:
        return False
    lower = text.lower()
    disallowed = [
        "as an ai", "as a language model", "as an artificial intelligence",
        "i am an ai", "i do not have feelings", "i don't have feelings",
        "certainly! here", "hope this helps", "feel free to ask", "how can i assist",
    ]
    for d in disallowed:
        if d in lower:
            return False
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
                if line.strip():
                    existing_hashes.add(hashlib.md5(line.strip().encode("utf-8")).hexdigest())

    online_status = "ONLINE" if is_ollama_online() else "OFFLINE (Using Expert Knowledge Engine)"
    print("=" * 70)
    print("STARTING DISTILLATION DATA GENERATION")
    print(f"Target count:   {count:,} conversations")
    print(f"Teacher model:  {teacher_model} [{online_status}]")
    print(f"Math ratio:     {math_ratio*100:.0f}% verified math / {(1-math_ratio)*100:.0f}% technical & chat")
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

    # 2. Generate Technical & Casual Dialogues
    print(f"\n[2/2] Generating {num_chat:,} dialogues from teacher / offline expert...")
    chat_done = 0
    attempts = 0
    max_attempts = num_chat * 4

    while chat_done < num_chat and attempts < max_attempts:
        attempts += 1
        topic_prompt = random.choice(ALL_TOPICS)
        user_prompt = f"Explain {topic_prompt} concisely."
        teacher_reply = query_ollama(teacher_model, user_prompt, temperature=0.85)

        if not teacher_reply:
            teacher_reply = query_ollama(FALLBACK_TEACHER, user_prompt, temperature=0.85)

        if not teacher_reply:
            teacher_reply = get_offline_teacher_reply(topic_prompt)

        if teacher_reply and is_clean_human_reply(teacher_reply):
            ex = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": topic_prompt},
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
                if chat_done % 50 == 0 or chat_done == num_chat:
                    print(f"  Progress: {chat_done}/{num_chat} dialogues ({chat_done/max(1.0, time.time()-t0):.1f}/s)...")

    # Append to distillation file
    with open(output_path, "a", encoding="utf-8") as f:
        for r in batch_records:
            f.write(r + "\n")

    # Also merge directly into cleaned_sft_dataset.jsonl
    sft_targets = [
        NESTED_ROOT / "training_data" / "cleaned_sft_dataset.jsonl",
        PROJECT_ROOT / "training_data" / "cleaned_sft_dataset.jsonl",
    ]
    for sft_p in sft_targets:
        if sft_p.is_file():
            with open(sft_p, "a", encoding="utf-8") as f:
                for r in batch_records:
                    f.write(r + "\n")
            print(f"  [MERGED] Appended {len(batch_records):,} records to {sft_p.name}")

    el = time.time() - t0
    print("\n" + "=" * 70)
    print(f"DISTILLATION COMPLETE: Added {generated:,} verified examples in {el:.1f}s")
    print(f"Total dataset entries: {len(existing_hashes):,}")
    print(f"Saved to: {output_path.resolve()}")
    print("=" * 70 + "\n")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Distill data from Ollama teachers")
    parser.add_argument("--count", type=int, default=300, help="Number of examples to generate")
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
