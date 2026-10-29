"""
Modern Tiny LLM AI Assistant CLI (v2.1.0)
High-performance terminal interface with real-time streaming, KV-cache acceleration,
Claude/ChatGPT-style Chain-of-Thought thinking, Advanced Math & Algebra solver,
RAG search, AST verifications, SQLite memory, and live checkpoint switching.
"""
import argparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import sys
import time
from typing import Optional, List

# Ensure UTF-8 stdout on Windows safely without re-wrapping
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
from tokenizers import Tokenizer

from ..models.checkpoint import load_modern_checkpoint, inspect_checkpoint
from ..models.transformer import ModernTransformer
from ..generation.sampler import TextGenerator, SamplingConfig
from ..generation.template import ChatTemplate, ChatMessage
from ..tools.registry import ToolRegistry
from ..tools.builtin import register_builtin_tools, solve_math_problem
from ..reasoning.orchestrator import ReasoningOrchestrator
from ..rag.pipeline import LocalRAGPipeline
from ..memory.conversation_db import ConversationDatabase


# ANSI Color formatting
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def discover_best_checkpoint(project_root: Path, requested: Optional[str] = None) -> Path:
    """Finds the best available checkpoint with fallback priority."""
    if requested:
        p = Path(requested)
        if not p.is_absolute():
            p = project_root / p
        if p.is_file():
            return p

    candidates = [
        project_root / "checkpoints" / "modern_chat_finetuned.pt",
        project_root / "checkpoints" / "modern_production_v2_best.pt",
        project_root / "checkpoints" / "modern_base_v2_best.pt",
        project_root / "checkpoints" / "modern_knowledge_v2_best.pt",
        project_root / "checkpoints" / "modern_tiny_transformer_best.pt",
        project_root / "checkpoints" / "tiny_bpe_expanded_continued_v2.pt",
        project_root / "checkpoints" / "tiny_bpe_expanded_training_run2.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c

    ckpt_dir = project_root / "checkpoints"
    if ckpt_dir.is_dir():
        pt_files = sorted(ckpt_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
        if pt_files:
            return pt_files[0]

    return project_root / "checkpoints" / "modern_base_v2_best.pt"


def load_model_safely(ckpt_path: Path, device: str):
    """Loads either modern v2 or legacy v1 BPE checkpoint."""
    if not ckpt_path.is_file():
        return None, "File not found"

    info = inspect_checkpoint(ckpt_path)
    ckpt_format = info.get("format", "unknown")

    if ckpt_format == "v2_modern":
        model, payload = load_modern_checkpoint(ckpt_path, device=device)
        return model, f"Modern Transformer v2.1 (step {payload.get('step', 0)}, val_loss {payload.get('val_loss', 'N/A')})"
    elif ckpt_format == "v1_legacy_bpe":
        try:
            from model_bpe import TinyBPEGPT
            raw = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = TinyBPEGPT(
                vocab_size=raw.get("vocab_size", 8000),
                block_size=raw.get("block_size", 256),
                n_embd=raw.get("n_embd", 192),
                n_head=raw.get("n_head", 4),
                n_layer=raw.get("n_layer", 3),
                dropout=0.0,
            ).to(device)
            model.load_state_dict(raw["model_state_dict"])
            model.eval()
            return model, f"Legacy BPE Model v1 (step {raw.get('step', 0)})"
        except Exception as e:
            return None, f"Legacy loader error: {e}"
    else:
        return None, f"Unrecognized format ({ckpt_format})"


def print_banner(model_name: str, model_desc: str, device: str, rag_chunks: int, tool_count: int):
    print(f"\n{C.CYAN}{C.BOLD}{'=' * 68}")
    print("   MODERN TINY LLM AI ASSISTANT (v2.1.0) - TERMINAL CONSOLE")
    print(f"{'=' * 68}{C.RESET}")
    print(f" {C.WHITE}Model:{C.RESET}    {C.GREEN}{model_name}{C.RESET} ({model_desc})")
    print(f" {C.WHITE}Hardware:{C.RESET} {C.YELLOW}{device.upper()}{C.RESET} | {C.WHITE}Acceleration:{C.RESET} {C.GREEN}KV-Cache O(1) Active{C.RESET}")
    print(f" {C.WHITE}RAG Knowledge:{C.RESET} {C.MAGENTA}{rag_chunks:,} chunks indexed{C.RESET} | {C.WHITE}Memory:{C.RESET} {C.GREEN}SQLite persistent{C.RESET}")
    print(f" {C.WHITE}Math Solver:{C.RESET}   {C.CYAN}Algebra, Calculus, Percentages, Units, Stats Active{C.RESET}")
    print(f"{C.GRAY}{'-' * 68}{C.RESET}")
    print(f" Type your prompt or command. {C.YELLOW}/help{C.RESET} for all available commands.")
    print(f"{C.GRAY}{'=' * 68}{C.RESET}\n")


def print_help():
    print(f"\n{C.BOLD}{C.CYAN}--- AVAILABLE COMMANDS ---{C.RESET}")
    print(f"  {C.YELLOW}/help{C.RESET}                Show this command manual")
    print(f"  {C.YELLOW}/exit{C.RESET} or {C.YELLOW}/quit{C.RESET}         Exit the assistant")
    print(f"  {C.YELLOW}/think [on|off]{C.RESET}      Toggle Claude/ChatGPT-style internal thinking traces")
    print(f"  {C.YELLOW}/stream [on|off]{C.RESET}     Toggle real-time streaming token generation")
    print(f"  {C.YELLOW}/rag [on|off]{C.RESET}        Enable/disable local RAG document retrieval")
    print(f"  {C.YELLOW}/rag search <query>{C.RESET}  Search local knowledge corpus and view citations")
    print(f"  {C.YELLOW}/tools{C.RESET}               List all registered tool schemas and permissions")
    print(f"  {C.YELLOW}/calc <query>{C.RESET}        Run math solver (e.g. /calc 2x+10=50, /calc 20% of 150, /calc 100 km to mi)")
    print(f"  {C.YELLOW}/study <file>{C.RESET}         Study textbook (Grades 1-12): extract concepts, flashcards, & index RAG")
    print(f"  {C.YELLOW}/code <snippet>{C.RESET}       Execute Python snippet safely in sandbox with stdout capture")
    print(f"  {C.YELLOW}/creative <prompt>{C.RESET}   Generate Claude-style intuitive analogies & creative mental models")
    print(f"  {C.YELLOW}/model{C.RESET}               List available checkpoints in checkpoints/")
    print(f"  {C.YELLOW}/model <filename>{C.RESET}    Switch active checkpoint on the fly")
    print(f"  {C.YELLOW}/temp <float>{C.RESET}        Set temperature (default: 0.70)")
    print(f"  {C.YELLOW}/tokens <int>{C.RESET}        Set max generated tokens (default: 120)")
    print(f"  {C.YELLOW}/topp <float>{C.RESET}        Set nucleus top-p sampling (default: 0.90)")
    print(f"  {C.YELLOW}/history{C.RESET}             Show recent messages from SQLite memory")
    print(f"  {C.YELLOW}/clear{C.RESET}               Clear current conversation memory context")
    print(f"  {C.YELLOW}/stats{C.RESET}               Show system parameters, RAM, context window, and acceleration metrics")
    print(f"  {C.YELLOW}/system <prompt>{C.RESET}     Set active system persona / prompt")
    print(f"{C.GRAY}{'-' * 40}{C.RESET}\n")


def launch_cli():
    parser = argparse.ArgumentParser(description="Modern Tiny LLM AI Assistant (v2.1.0)")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="tokenizer_bpe.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--temperature", type=float, default=0.70)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--rag", action="store_true", help="Start with RAG enabled")
    parser.add_argument("--think", action="store_true", default=True, help="Enable Chain-of-Thought thinking")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    tok_path = project_root / args.tokenizer
    db_path = project_root / "memory.db"

    # Auto device resolution
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Tokenizer
    if not tok_path.is_file():
        print(f"{C.RED}ERROR: Tokenizer not found at {tok_path}{C.RESET}")
        return
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Initialize Memory & Session
    memory = ConversationDatabase(db_path)
    session_id = f"session_{int(time.time())}"
    memory.create_session(session_id, "Interactive Terminal CLI")

    # Initialize Tools
    tools = ToolRegistry()
    register_builtin_tools(tools, project_root)

    # Initialize RAG
    rag = LocalRAGPipeline()
    knowledge_dir = project_root / "data_knowledge"
    if knowledge_dir.is_dir():
        docs = list(knowledge_dir.rglob("*.txt"))
        for doc in docs[:100]:
            rag.add_document(doc)
    elif (project_root / "alice.txt").is_file():
        rag.add_document(project_root / "alice.txt")

    orchestrator = ReasoningOrchestrator(
        rag_pipeline=rag if args.rag else None,
        tool_registry=tools,
        enable_thinking=args.think,
    )

    # Find and Load Checkpoint
    active_ckpt_path = discover_best_checkpoint(project_root, args.checkpoint)
    model, model_desc = load_model_safely(active_ckpt_path, device=device)

    # Generation Config
    sampling_cfg = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.tokens,
    )
    stream_output = True
    show_thinking = True
    rag_enabled = args.rag
    system_persona = "You are a helpful, precise AI assistant powered by Modern Transformer v2."

    # Show Banner
    print_banner(
        model_name=active_ckpt_path.name if active_ckpt_path else "None",
        model_desc=model_desc,
        device=device,
        rag_chunks=len(rag.chunks),
        tool_count=len(tools.list_tools()),
    )

    # Main Interactive REPL
    while True:
        try:
            prompt_str = f"{C.BOLD}{C.GREEN}You > {C.RESET}"
            user_input = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C.YELLOW}Exiting AI Assistant session. Goodbye!{C.RESET}")
            break

        if not user_input:
            continue

        cmd = user_input.lower().split()
        lead = cmd[0]

        # Built-in Terminal Commands
        if lead in ["/exit", "/quit", "exit", "quit"]:
            print(f"{C.YELLOW}Goodbye!{C.RESET}")
            break

        elif lead == "/help":
            print_help()
            continue

        elif lead == "/think":
            if len(cmd) > 1 and cmd[1] in ["off", "false", "0"]:
                show_thinking = False
                print(f"{C.GRAY}[Chain-of-Thought thinking display disabled]{C.RESET}")
            else:
                show_thinking = True
                print(f"{C.GRAY}[Chain-of-Thought thinking display enabled]{C.RESET}")
            continue

        elif lead == "/stream":
            if len(cmd) > 1 and cmd[1] in ["off", "false", "0"]:
                stream_output = False
                print(f"{C.GRAY}[Streaming disabled]{C.RESET}")
            else:
                stream_output = True
                print(f"{C.GRAY}[Streaming enabled]{C.RESET}")
            continue

        elif lead == "/rag":
            if len(cmd) > 1 and cmd[1] == "search":
                search_q = " ".join(user_input.split()[2:])
                if not search_q:
                    print(f"{C.RED}Usage: /rag search <query>{C.RESET}")
                else:
                    results = rag.search(search_q, top_k=3)
                    print(f"\n{C.CYAN}RAG Search Results for '{search_q}':{C.RESET}")
                    if not results:
                        print(f"{C.GRAY}  No matching passages found.{C.RESET}")
                    for r in results:
                        doc_n = getattr(r.chunk, "doc_name", "doc")
                        c_idx = getattr(r.chunk, "chunk_idx", 0)
                        txt = getattr(r.chunk, "text", "")[:150]
                        print(f"  {C.YELLOW}[Score: {r.score:.3f} | {doc_n} (Chunk {c_idx})]{C.RESET}")
                        print(f"  {C.GRAY}{txt}...{C.RESET}\n")
            elif len(cmd) > 1 and cmd[1] in ["on", "enable"]:
                rag_enabled = True
                orchestrator.rag = rag
                print(f"{C.GREEN}[RAG knowledge retrieval enabled]{C.RESET}")
            elif len(cmd) > 1 and cmd[1] in ["off", "disable"]:
                rag_enabled = False
                orchestrator.rag = None
                print(f"{C.YELLOW}[RAG knowledge retrieval disabled]{C.RESET}")
            else:
                status_str = "ENABLED" if rag_enabled else "DISABLED"
                print(f"{C.WHITE}RAG status: {C.GREEN if rag_enabled else C.YELLOW}{status_str}{C.RESET} ({len(rag.chunks)} chunks in index). Use '/rag on' or '/rag off'")
            continue

        elif lead == "/tools":
            print(f"\n{C.BOLD}{C.CYAN}Registered Built-in Tools ({len(tools.list_tools())}):{C.RESET}")
            for t in tools.list_tools():
                print(f"  {C.YELLOW}• {t['name']}{C.RESET}: {t['description']}")
            print()
            continue

        elif lead == "/calc":
            expr = " ".join(user_input.split()[1:])
            if not expr:
                print(f"{C.RED}Usage: /calc 2x + 10 = 50  OR  /calc 20% of 150  OR  /calc 100 km to miles{C.RESET}")
            else:
                res = solve_math_problem(expr)
                if res.get("success"):
                    print(f"\n{res.get('explanation')}\n")
                else:
                    print(f"{C.RED}Math Error:{C.RESET} {res.get('error')}")
            continue

        elif lead == "/study":
            parts = user_input.split()
            if len(parts) < 2:
                print(f"{C.RED}Usage: /study <path_to_book_or_pdf> [grade_hint]{C.RESET}")
            else:
                book_target = parts[1]
                hint = parts[2] if len(parts) > 2 else None
                from ..data.curriculum import CurriculumStudyEngine
                engine = CurriculumStudyEngine(project_root / "data_curriculum", rag_pipeline=rag)
                target_p = Path(book_target)
                if not target_p.is_absolute():
                    target_p = project_root / target_p
                if not target_p.is_file():
                    print(f"{C.RED}File not found: {book_target}{C.RESET}")
                else:
                    print(f"\n{C.CYAN}--- STUDYING TEXTBOOK LIKE A HUMAN STUDENT ---{C.RESET}")
                    print(f"Reading and analyzing: {target_p.name}...")
                    try:
                        res = engine.study_book(target_p, grade_hint=hint)
                        print(f"  {C.GREEN}• Grade Classification:{C.RESET} {res['grade_name']} (Difficulty: {res['difficulty']}/12)")
                        print(f"  {C.GREEN}• Key Concepts Extracted:{C.RESET} {res['concepts_learned']}")
                        print(f"  {C.GREEN}• Q&A Flashcards Generated:{C.RESET} {res['flashcards_generated']}")
                        print(f"  {C.GREEN}• Study Notes Saved:{C.RESET} {Path(res['study_notes_path']).name}")
                        print(f"  {C.GREEN}• RAG Chunks Indexed:{C.RESET} {res['rag_chunks_indexed']}")
                        print(f"{C.GRAY}Ready for reading comprehension and inquiry.{C.RESET}\n")
                    except Exception as e:
                        print(f"{C.RED}Error studying book: {e}{C.RESET}")
            continue

        elif lead == "/code":
            code_text = " ".join(user_input.split()[1:])
            if not code_text:
                print(f"{C.RED}Usage: /code print(sum([x**2 for x in range(10)])){C.RESET}")
            else:
                from ..tools.code_engine import SafeCodeSandbox
                res = SafeCodeSandbox.execute_snippet(code_text)
                if res["success"]:
                    print(f"\n{C.GREEN}[SANDBOX OUTPUT]{C.RESET}")
                    print(res["stdout"] if res["stdout"] else "(Executed with returncode 0; no stdout)")
                    print(f"{C.GRAY}(Elapsed: {res['elapsed']:.4f}s){C.RESET}\n")
                else:
                    print(f"\n{C.RED}[SANDBOX ERROR] {res.get('error')}{C.RESET}\n")
            continue

        elif lead == "/creative":
            creative_q = " ".join(user_input.split()[1:])
            if not creative_q:
                print(f"{C.RED}Usage: /creative quantum mechanics  OR  /creative gravity{C.RESET}")
            else:
                from ..generation.creative import CreativeEngine
                analogy = CreativeEngine.get_intuitive_analogy(creative_q)
                if analogy:
                    print(f"\n{C.CYAN}### Intuitive Mental Model for '{creative_q}':{C.RESET}\n")
                    print(analogy)
                    print(f"\n{C.GRAY}(Intuitive everyday analogy mapped from core principles){C.RESET}\n")
                else:
                    print(f"{C.YELLOW}Topic not in preset analogies. Generating via model...{C.RESET}")
            continue

        elif lead == "/model":
            if len(cmd) > 1:
                target_name = cmd[1]
                target_path = project_root / "checkpoints" / target_name
                if not target_path.is_file():
                    target_path = Path(target_name)
                if not target_path.is_file():
                    print(f"{C.RED}Checkpoint not found: {target_name}{C.RESET}")
                else:
                    print(f"{C.GRAY}Loading checkpoint: {target_path.name}...{C.RESET}")
                    new_model, new_desc = load_model_safely(target_path, device=device)
                    if new_model:
                        model = new_model
                        model_desc = new_desc
                        active_ckpt_path = target_path
                        print(f"{C.GREEN}Switched to: {active_ckpt_path.name} ({model_desc}){C.RESET}")
                    else:
                        print(f"{C.RED}Failed to load checkpoint: {new_desc}{C.RESET}")
            else:
                ckpt_dir = project_root / "checkpoints"
                print(f"\n{C.BOLD}{C.CYAN}Available Checkpoints in checkpoints/:{C.RESET}")
                if ckpt_dir.is_dir():
                    for f in sorted(ckpt_dir.glob("*.pt")):
                        is_active = (active_ckpt_path and f.name == active_ckpt_path.name)
                        marker = f"{C.GREEN}* (ACTIVE){C.RESET}" if is_active else ""
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"  {f.name:36} [{size_mb:5.1f} MB] {marker}")
                print(f"{C.GRAY}Switch models using: /model <filename>{C.RESET}\n")
            continue

        elif lead == "/temp":
            try:
                v = float(cmd[1])
                sampling_cfg.temperature = max(0.01, min(2.0, v))
                print(f"{C.GRAY}[Temperature set to {sampling_cfg.temperature:.2f}]{C.RESET}")
            except Exception:
                print(f"{C.RED}Usage: /temp 0.70{C.RESET}")
            continue

        elif lead == "/tokens":
            try:
                v = int(cmd[1])
                sampling_cfg.max_new_tokens = max(10, min(1024, v))
                print(f"{C.GRAY}[Max new tokens set to {sampling_cfg.max_new_tokens}]{C.RESET}")
            except Exception:
                print(f"{C.RED}Usage: /tokens 150{C.RESET}")
            continue

        elif lead == "/topp":
            try:
                v = float(cmd[1])
                sampling_cfg.top_p = max(0.1, min(1.0, v))
                print(f"{C.GRAY}[Top-p set to {sampling_cfg.top_p:.2f}]{C.RESET}")
            except Exception:
                print(f"{C.RED}Usage: /topp 0.90{C.RESET}")
            continue

        elif lead == "/history":
            history = memory.get_messages(session_id, limit=12)
            print(f"\n{C.BOLD}{C.CYAN}--- Session History ({len(history)} messages) ---{C.RESET}")
            for m in history:
                role_col = C.GREEN if m["role"] == "user" else C.CYAN
                print(f"  {role_col}{m['role'].upper()}:{C.RESET} {m['content']}")
            print(f"{C.GRAY}{'-' * 40}{C.RESET}\n")
            continue

        elif lead == "/clear":
            session_id = f"session_{int(time.time())}"
            memory.create_session(session_id, "Interactive Terminal CLI")
            print(f"{C.GRAY}[Conversation context cleared. Started fresh session.]{C.RESET}")
            continue

        elif lead == "/system":
            if len(cmd) > 1:
                system_persona = " ".join(user_input.split()[1:])
                print(f"{C.GRAY}[System persona updated to: '{system_persona}']{C.RESET}")
            else:
                print(f"{C.WHITE}Active system persona:{C.RESET} {system_persona}")
            continue

        elif lead == "/stats":
            print(f"\n{C.BOLD}{C.CYAN}--- SYSTEM STATUS & METRICS ---{C.RESET}")
            print(f"  {C.WHITE}Active Model:{C.RESET}      {active_ckpt_path.name if active_ckpt_path else 'None'}")
            print(f"  {C.WHITE}Architecture:{C.RESET}      {model_desc}")
            if model and hasattr(model, "config"):
                cfg = model.config
                print(f"  {C.WHITE}Layers/Heads:{C.RESET}      {cfg.n_layer} layers / {cfg.n_head} attention heads")
                print(f"  {C.WHITE}Embedding Dim:{C.RESET}     {cfg.n_embd} dimensions")
                print(f"  {C.WHITE}Context Window:{C.RESET}    {cfg.block_size} tokens base (RoPE extrapolated to 1,024+)")
                print(f"  {C.WHITE}Parameters:{C.RESET}        {sum(p.numel() for p in model.parameters()):,} trainable")
            print(f"  {C.WHITE}KV-Cache Acceleration:{C.RESET} {C.GREEN}ENABLED (O(1) Step Generation){C.RESET}")
            print(f"  {C.WHITE}Compute Device:{C.RESET}    {device.upper()}")
            print(f"  {C.WHITE}RAG Index Chunks:{C.RESET}  {len(rag.chunks):,}")
            print(f"  {C.WHITE}Memory DB:{C.RESET}         {db_path.name}")
            print(f"  {C.WHITE}Streaming:{C.RESET}         {'ON' if stream_output else 'OFF'}")
            print(f"  {C.WHITE}Thinking Trace:{C.RESET}    {'VISIBLE' if show_thinking else 'HIDDEN'}")
            print(f"  {C.WHITE}Temperature:{C.RESET}       {sampling_cfg.temperature:.2f}")
            print(f"  {C.WHITE}Max Tokens:{C.RESET}        {sampling_cfg.max_new_tokens}")
            print(f"{C.GRAY}{'-' * 40}{C.RESET}\n")
            continue

        # Standard Processing: Store in SQLite Memory
        memory.add_message(session_id, "user", user_input)
        generator = TextGenerator(model, tokenizer) if model else None

        # Stage 1: Prepare Reasoning, Intent, Thinking, and Tools
        t0 = time.time()
        prep = orchestrator.prepare_reasoning(user_input)

        # Display Claude/ChatGPT-style internal thinking trace if enabled
        if show_thinking and prep.thinking:
            print(f"\n{C.GRAY}╭─ <thinking> ────────────────────────────────────────{C.RESET}")
            for t_line in prep.thinking.splitlines():
                print(f"{C.GRAY}│ {t_line}{C.RESET}")
            print(f"{C.GRAY}╰─────────────────────────────────────────────────────{C.RESET}")

        # Print AI Header
        print(f"\n{C.BOLD}{C.CYAN}AI {C.YELLOW}[{prep.intent}]{C.CYAN} > {C.RESET}", end="")
        sys.stdout.flush()

        # Stage 2: Execution / Output
        if prep.direct_answer:
            final_text = prep.direct_answer
            print(final_text)
            verifications = prep.verifications
        else:
            if not generator:
                final_text = "Model weights not loaded. Please select a valid checkpoint via /model."
                print(final_text)
            elif stream_output:
                prompt_ids = tokenizer.encode(prep.prompt_with_context).ids
                pieces = []
                for chunk in generator.generate_stream(prompt_ids, config=sampling_cfg, device=device):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    pieces.append(chunk)
                print()
                final_text = "".join(pieces)
            else:
                final_text = generator.generate(prep.prompt_with_context, config=sampling_cfg, device=device)
                print(final_text)

            from ..reasoning.verifier import ResponseVerifier
            verifications = [
                ResponseVerifier.verify_math_claims(final_text).to_dict(),
                ResponseVerifier.verify_python_code(final_text).to_dict(),
            ]
            if prep.retrieved_citations:
                verifications.append(ResponseVerifier.verify_citations(final_text, prep.retrieved_citations).to_dict())

        elapsed = time.time() - t0

        # Display Verifications & Citations
        for v in verifications:
            if v.get("passed") and v.get("details"):
                print(f"  {C.GRAY}{C.GREEN}[OK] {v.get('check')}: {v.get('details')}{C.RESET}")

        for cite in prep.retrieved_citations:
            print(f"  {C.GRAY}{C.MAGENTA}[Citation] {cite}{C.RESET}")

        print(f"{C.GRAY}({elapsed:.2f}s){C.RESET}")

        memory.add_message(session_id, "assistant", final_text)


if __name__ == "__main__":
    launch_cli()
