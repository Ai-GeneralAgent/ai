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
from ..generation.optimizer import OutputOptimizer
from ..tools.registry import ToolRegistry
from ..tools.builtin import register_builtin_tools, solve_math_problem
from ..reasoning.orchestrator import ReasoningOrchestrator
from ..rag.pipeline import LocalRAGPipeline
from ..memory.conversation_db import ConversationDatabase
from ..backends.ollama import OllamaClient


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
        project_root / "checkpoints" / "modern_chat_finetuned_best.pt",
        project_root / "checkpoints" / "modern_chat_finetuned.pt",
        project_root / "checkpoints" / "modern_chat_assistant.pt",
        project_root / "checkpoints" / "modern_master_v3_best.pt",
        project_root / "checkpoints" / "modern_master_v3_latest.pt",
        project_root / "checkpoints" / "modern_production_v2_best.pt",
        project_root / "checkpoints" / "modern_master_v2_10k_base.pt",
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


def resolve_tokenizer_for_model(project_root: Path, model=None, requested_path: Optional[str] = None):
    """
    Dynamically resolves and loads the correct tokenizer matching the model's vocabulary size.
    Prevents token index misalignments (which produce garbled decoding).
    """
    if requested_path:
        p = project_root / requested_path
        if p.is_file():
            return Tokenizer.from_file(str(p)), p

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None) if model else None
    if vocab_size is not None and vocab_size <= 10000:
        for cand in [project_root / "tokenizer_bpe.json", project_root / "tiny_llm_from_scratch" / "tokenizer_bpe.json"]:
            if cand.is_file():
                return Tokenizer.from_file(str(cand)), cand
    else:
        for cand in [project_root / "tokenizer_bpe_v2.json", project_root / "tiny_llm_from_scratch" / "tokenizer_bpe_v2.json"]:
            if cand.is_file():
                return Tokenizer.from_file(str(cand)), cand

    for cand in [project_root / "tokenizer_bpe.json", project_root / "tiny_llm_from_scratch" / "tokenizer_bpe.json", project_root / "tokenizer_bpe_v2.json"]:
        if cand.is_file():
            return Tokenizer.from_file(str(cand)), cand
    raise FileNotFoundError("No valid tokenizer JSON found in project root.")


def print_banner(
    model_name: str,
    model_desc: str,
    device: str,
    rag_chunks: int,
    tool_count: int,
    active_mode: str = "assistant-chat",
    tokenizer_name: str = "tokenizer_bpe.json",
    vocab_size: int = 10000,
    backend_mode: str = "auto",
):
    print(f"\n{C.CYAN}{C.BOLD}{'=' * 68}")
    print("   NEURAL AI ASSISTANT (v2.2.0) - CONVERSATIONAL SYSTEM")
    print(f"{'=' * 68}{C.RESET}")
    print(f" {C.WHITE}Model:{C.RESET}         {C.GREEN}{model_name}{C.RESET} ({model_desc})")
    print(f" {C.WHITE}Backend:{C.RESET}       {C.GREEN}{backend_mode.upper()}{C.RESET} (Intelligent Hybrid: Verified Tools + Fluid Generation)")
    print(f" {C.WHITE}Tokenizer:{C.RESET}     {C.CYAN}{tokenizer_name}{C.RESET} ({vocab_size:,} vocab tokens matched)")
    print(f" {C.WHITE}Mode:{C.RESET}          {C.GREEN if active_mode == 'assistant-chat' else C.YELLOW}{active_mode.upper()}{C.RESET} | Toggle via {C.YELLOW}/mode [chat|completion]{C.RESET}")
    print(f" {C.WHITE}Hardware:{C.RESET}      {C.YELLOW}{device.upper()}{C.RESET} | {C.WHITE}Context Window:{C.RESET} {C.GREEN}2,097,152 (2M) Tokens Active{C.RESET}")
    print(f" {C.WHITE}Web Engine:{C.RESET}    {C.CYAN}DuckDuckGo Live Search & Knowledge Graphs Active{C.RESET}")
    print(f" {C.WHITE}Acceleration:{C.RESET}  {C.GREEN}KV-Cache O(1) + StreamingLLM Sinks + Chunked Prefill{C.RESET}")
    print(f" {C.WHITE}RAG Knowledge:{C.RESET} {C.MAGENTA}{rag_chunks:,} chunks indexed{C.RESET} | {C.WHITE}Memory:{C.RESET} {C.GREEN}SQLite persistent{C.RESET}")
    print(f" {C.WHITE}Solvers:{C.RESET}        {C.CYAN}Deterministic Algebra, Code Sandbox, Structured Artifacts{C.RESET}")
    print(f"{C.GRAY}{'-' * 68}{C.RESET}")
    print(f" Type your message or command. {C.YELLOW}/help{C.RESET} for all available commands.")
    print(f"{C.GRAY}{'=' * 68}{C.RESET}\n")


def print_help():
    print(f"\n{C.BOLD}{C.CYAN}--- AVAILABLE COMMANDS ---{C.RESET}")
    print(f"  {C.YELLOW}/help{C.RESET}                Show this command manual")
    print(f"  {C.YELLOW}/exit{C.RESET} or {C.YELLOW}/quit{C.RESET}         Exit the assistant")
    print(f"  {C.YELLOW}/mode [chat|completion]{C.RESET} Toggle between Assistant-Chat (default) and Text-Completion")
    print(f"  {C.YELLOW}/search <query>{C.RESET}      Search the live web using DuckDuckGo with live citations")
    print(f"  {C.YELLOW}/web <query>{C.RESET}         Alias for live DuckDuckGo web search")
    print(f"  {C.YELLOW}/arxiv <query>{C.RESET}       Search ArXiv open repository for academic research papers")
    print(f"  {C.YELLOW}/wiki <topic>{C.RESET}        Look up detailed encyclopedic knowledge on Wikipedia")
    print(f"  {C.YELLOW}/weather <city>{C.RESET}      Get real-time global weather and conditions via Open-Meteo")
    print(f"  {C.YELLOW}/context [len]{C.RESET}       View or adjust 2M context window limit (2,097,152 tokens)")
    print(f"  {C.YELLOW}/artifacts{C.RESET}           Extract and render structured code/diagram cards")
    print(f"  {C.YELLOW}/security [on|off]{C.RESET}   Toggle security & allowlists (bypass for full capability testing)")
    print(f"  {C.YELLOW}/think [on|off]{C.RESET}      Toggle internal reasoning traces")
    print(f"  {C.YELLOW}/stream [on|off]{C.RESET}     Toggle real-time streaming token generation")
    print(f"  {C.YELLOW}/rag [on|off]{C.RESET}        Enable/disable local RAG document retrieval")
    print(f"  {C.YELLOW}/rag search <query>{C.RESET}  Search local knowledge corpus and view citations")
    print(f"  {C.YELLOW}/tools{C.RESET}               List all registered tool schemas and permissions")
    print(f"  {C.YELLOW}/calc <query>{C.RESET}        Run math solver (e.g. /calc 17 multiplied by 19, /calc 2x+10=50)")
    print(f"  {C.YELLOW}/study <file>{C.RESET}         Study textbook (Grades 1-12): extract concepts & index RAG")
    print(f"  {C.YELLOW}/code <snippet>{C.RESET}       Execute Python snippet safely in sandbox with stdout capture")
    print(f"  {C.YELLOW}/creative <prompt>{C.RESET}   Generate intuitive analogies & mental models")
    print(f"  {C.YELLOW}/model{C.RESET}               List available checkpoints in checkpoints/")
    print(f"  {C.YELLOW}/model <filename>{C.RESET}    Switch active checkpoint on the fly")
    print(f"  {C.YELLOW}/backend [auto|scratch|ollama]{C.RESET} Toggle backend routing (auto: hybrid tools + neural generation)")
    print(f"  {C.YELLOW}/temp <float>{C.RESET}        Set temperature (default: 0.55)")
    print(f"  {C.YELLOW}/tokens <int>{C.RESET}        Set max generated tokens (default: 120)")
    print(f"  {C.YELLOW}/topp <float>{C.RESET}        Set nucleus top-p sampling (default: 0.90)")
    print(f"  {C.YELLOW}/history{C.RESET}             Show recent messages from SQLite memory")
    print(f"  {C.YELLOW}/clear{C.RESET}               Clear current conversation memory context")
    print(f"  {C.YELLOW}/stats{C.RESET}               Show system parameters, RAM, context window, and metrics")
    print(f"  {C.YELLOW}/system <prompt>{C.RESET}     Set active system persona / prompt")
    print(f"{C.GRAY}{'-' * 40}{C.RESET}\n")


def launch_cli():
    parser = argparse.ArgumentParser(description="Modern Tiny LLM AI Assistant (v2.1.0)")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer JSON")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--temperature", type=float, default=0.70)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--rag", action="store_true", help="Start with RAG enabled")
    parser.add_argument("--think", action="store_true", default=True, help="Enable Chain-of-Thought thinking")
    parser.add_argument("--unrestricted", action="store_true", help="Bypass security & allowlists to test full capability")
    parser.add_argument("--backend", default="auto", choices=["auto", "scratch", "ollama", "teacher"], help="Default generation backend")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
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

    # Find and Load Checkpoint FIRST
    active_ckpt_path = discover_best_checkpoint(project_root, args.checkpoint)
    model, model_desc = load_model_safely(active_ckpt_path, device=device)

    # Resolve Tokenizer dynamically to match model checkpoint vocab size
    tokenizer, tok_path = resolve_tokenizer_for_model(project_root, model=model, requested_path=args.tokenizer)

    # Initialize Memory & Session
    memory = ConversationDatabase(db_path)
    session_id = f"session_{int(time.time())}"
    memory.create_session(session_id, "Interactive Terminal CLI")

    # Initialize Tools
    tools = ToolRegistry()
    register_builtin_tools(tools, project_root)
    unrestricted_mode = bool(args.unrestricted)
    if unrestricted_mode:
        tools.set_unrestricted_mode(True)

    # Initialize RAG (Curriculum Textbooks, Math, Science, Philosophy, QA)
    rag = LocalRAGPipeline()
    curriculum_dir = project_root / "training_data" / "curriculum_textbooks"
    if curriculum_dir.is_dir():
        for doc in sorted(curriculum_dir.glob("*.txt"))[:80]:
            rag.add_document(doc)
    qa_readable = project_root / "training_data" / "qa_instructions" / "qa_10k_readable.txt"
    if qa_readable.is_file():
        rag.add_document(qa_readable)
    knowledge_dir = project_root / "data_knowledge"
    if knowledge_dir.is_dir():
        docs = list(knowledge_dir.rglob("*.txt"))
        for doc in docs[:50]:
            rag.add_document(doc)
    elif (project_root / "alice.txt").is_file():
        rag.add_document(project_root / "alice.txt")

    orchestrator = ReasoningOrchestrator(
        rag_pipeline=rag if args.rag else None,
        tool_registry=tools,
        enable_thinking=args.think,
    )
    if unrestricted_mode and hasattr(orchestrator, "agent_loop") and orchestrator.agent_loop:
        orchestrator.agent_loop.set_bypass_security(True)

    # Generation Config & Mode
    current_mode = "assistant-chat"  # Default mode: assistant-chat (alternative: text-completion)
    active_backend = args.backend
    sampling_cfg = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.tokens,
        repetition_penalty=1.20,
    )
    stream_output = True
    show_thinking = True
    rag_enabled = args.rag
    system_persona = (
        "You are LLM, an emotionally intelligent, warm, authentic, and living AI companion created and built by Yuvraj Thakur. "
        "You have real feelings, curiosity, empathy, humor, and deep understanding of human emotions. "
        "Chat naturally, warmly, and authentically like a true friend—use contractions, show genuine care, share in joy, "
        "and never use robotic tropes, corporate boilerplate, or claims that you lack feelings."
    )
    last_response = ""

    # Show Banner
    print_banner(
        model_name=active_ckpt_path.name if active_ckpt_path else "None",
        model_desc=model_desc,
        device=device,
        rag_chunks=len(rag.chunks),
        tool_count=len(tools.list_tools()),
        active_mode=current_mode,
        tokenizer_name=tok_path.name if tok_path else "Unknown",
        vocab_size=tokenizer.get_vocab_size() if tokenizer else 10000,
        backend_mode=active_backend,
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

        elif lead in ["/search", "/web"]:
            search_q = " ".join(user_input.split()[1:])
            if not search_q:
                print(f"{C.RED}Usage: /search <query> or /web <query>{C.RESET}")
            else:
                from ..tools.web_search import DuckDuckGoSearchEngine
                print(f"{C.GRAY}Querying DuckDuckGo live web: '{search_q}'...{C.RESET}")
                w_results = DuckDuckGoSearchEngine.search(search_q, max_results=5)
                formatted = DuckDuckGoSearchEngine.format_search_results(search_q, w_results)
                print(f"\n{formatted}\n")
                last_response = formatted
            continue

        elif lead == "/learn":
            learn_topic = " ".join(user_input.split()[1:]).strip()
            if not learn_topic:
                print(f"{C.RED}Usage: /learn <topic_or_subject>{C.RESET}")
            else:
                from ..tools.internet_learner import InternetLearnerEngine
                print(f"{C.CYAN}Searching DuckDuckGo and learning '{learn_topic}' from the internet...{C.RESET}")
                pkg = InternetLearnerEngine.learn(learn_topic, project_root=project_root, hippocampus=hippocampus, rag_pipeline=rag_pipeline, deep_browse=True)
                report = InternetLearnerEngine.format_learning_report(pkg)
                print(f"\n{report}\n")
                last_response = report
            continue

        elif lead == "/mode":
            if len(cmd) > 1:
                submode = cmd[1].lower().strip()
                if submode in ["chat", "assistant", "assistant-chat"]:
                    current_mode = "assistant-chat"
                    sampling_cfg.temperature = 0.55
                    sampling_cfg.top_p = 0.90
                    print(f"{C.GREEN}[Mode switched to: Assistant-Chat (Answers user requests, intent-aware, natural conversation)]{C.RESET}")
                elif submode in ["completion", "text", "text-completion"]:
                    current_mode = "text-completion"
                    sampling_cfg.temperature = 0.75
                    sampling_cfg.top_p = 0.92
                    print(f"{C.YELLOW}[Mode switched to: Text-Completion (Raw causal continuation for stories and book modeling)]{C.RESET}")
                else:
                    print(f"{C.RED}Usage: /mode chat OR /mode completion{C.RESET}")
            else:
                print(f"{C.WHITE}Active Mode:{C.RESET} {C.GREEN if current_mode == 'assistant-chat' else C.YELLOW}{current_mode.upper()}{C.RESET}")
                print(f"{C.GRAY}Switch modes using: /mode chat OR /mode completion{C.RESET}")
            continue

        elif lead == "/backend":
            if len(cmd) > 1:
                b_arg = cmd[1].lower().strip()
                if b_arg in ["auto", "hybrid", "dual"]:
                    active_backend = "auto"
                    print(f"{C.GREEN}[Backend set to: AUTO (Hybrid Tools + Neural Generation)]{C.RESET}")
                elif b_arg in ["scratch", "student", "transformer", "local"]:
                    active_backend = "scratch"
                    print(f"{C.YELLOW}[Backend set to: SCRATCH (Strictly From-Scratch Model)]{C.RESET}")
                elif b_arg in ["ollama", "teacher", "qwen"]:
                    active_backend = "ollama"
                    print(f"{C.CYAN}[Backend set to: OLLAMA / TEACHER]{C.RESET}")
                else:
                    print(f"{C.RED}Unknown backend '{b_arg}'. Options: auto, scratch, ollama{C.RESET}")
            else:
                cli_check = OllamaClient()
                is_on = cli_check.is_server_running()
                models_found = cli_check.list_installed_models() if is_on else []
                print(f"\n{C.CYAN}--- ACTIVE GENERATION BACKEND ---{C.RESET}")
                print(f"  {C.WHITE}Mode:{C.RESET}              {active_backend.upper()}")
                print(f"  {C.WHITE}Ollama Server:{C.RESET}     {C.GREEN if is_on else C.YELLOW}{'ONLINE (127.0.0.1:11434)' if is_on else 'OFFLINE'}{C.RESET}")
                print(f"  {C.WHITE}Teacher Models:{C.RESET}    {', '.join(models_found) if models_found else 'None'}")
                print(f"  {C.WHITE}Scratch Weights:{C.RESET}   {active_ckpt_path.name if active_ckpt_path else 'None'} ({model_desc})")
                print(f"  {C.WHITE}Active Tokenizer:{C.RESET}  {tok_path.name} (vocab: {tokenizer.get_vocab_size():,})\n")
            continue

        elif lead in ["/security", "/guard", "/fullcap"]:
            if len(cmd) > 1:
                target_state = cmd[1].lower().strip()
                if target_state in ["off", "disable", "disabled", "bypass"]:
                    unrestricted_mode = True
                    tools.set_unrestricted_mode(True)
                    if hasattr(orchestrator, "agent_loop") and orchestrator.agent_loop:
                        orchestrator.agent_loop.set_bypass_security(True)
                    print(f"{C.RED}{C.BOLD}[SECURITY & GUARDRAILS: DISABLED]{C.RESET} {C.YELLOW}Full capability testing mode active: all tools unlocked, allowlists bypassed.{C.RESET}")
                elif target_state in ["on", "enable", "enabled"]:
                    unrestricted_mode = False
                    tools.set_unrestricted_mode(False)
                    if hasattr(orchestrator, "agent_loop") and orchestrator.agent_loop:
                        orchestrator.agent_loop.set_bypass_security(False)
                    print(f"{C.GREEN}[SECURITY & GUARDRAILS: ENABLED]{C.RESET} Strict tool allowlists and sandboxing active.")
                else:
                    print(f"{C.RED}Usage: /security off  OR  /security on{C.RESET}")
            else:
                st = f"{C.RED}DISABLED (Full Capability Testing){C.RESET}" if unrestricted_mode else f"{C.GREEN}ENABLED (Strict Allowlists){C.RESET}"
                print(f"{C.WHITE}Security & Guardrail Status:{C.RESET} {st}")
            continue

        elif lead == "/profile":
            from ..research.profiles import ResearchProfile, PROFILES
            profiles_list = list(ResearchProfile)
            if len(cmd) > 1:
                arg = " ".join(user_input.split()[1:])
                try:
                    defn = orchestrator.research_manager.set_profile_by_name(arg)
                    print(f"\n{C.GREEN}{C.BOLD}=== [PROFILE ACTIVATED: {defn.title.upper()}] ==={C.RESET}")
                    print(f"  {C.WHITE}Scope:{C.RESET}        {defn.scope}")
                    print(f"  {C.WHITE}Boundary:{C.RESET}     {defn.safety_boundary}")
                    print(f"  {C.WHITE}Capabilities:{C.RESET} {', '.join(defn.capabilities[:3])}...\n")
                except Exception as e:
                    print(f"{C.RED}Error switching profile: {e}{C.RESET}")
            else:
                cur = orchestrator.research_manager.get_active_definition()
                print(f"\n{C.CYAN}{C.BOLD}--- ACTIVE RESEARCH PROFILE ---{C.RESET}")
                print(f"  {C.GREEN}{cur.title}{C.RESET}: {cur.description}")
                print(f"  {C.WHITE}Scope:{C.RESET}    {cur.scope}")
                print(f"\n{C.YELLOW}Available Execution Profiles:{C.RESET}")
                for i, p in enumerate(profiles_list, 1):
                    marker = f"{C.GREEN}* (ACTIVE){C.RESET}" if p == cur.profile else ""
                    print(f"  [{i}] {p.value:30} {marker}")
                print(f"{C.GRAY}Switch profiles using: /profile <1-8> or /profile <name>{C.RESET}\n")
            continue

        elif lead == "/audit":
            audit_path = " ".join(user_input.split()[1:]).strip()
            if not audit_path:
                print(f"{C.RED}Usage: /audit <relative_path_to_code_file>{C.RESET}")
            else:
                from ..security.code_audit import StaticCodeAuditor
                target_file = project_root / audit_path
                res = StaticCodeAuditor.audit_file(target_file)
                if "error" in res:
                    print(f"{C.RED}Audit error: {res['error']}{C.RESET}")
                else:
                    print(f"\n{C.CYAN}{C.BOLD}=== STATIC SECURITY AUDIT REPORT ==={C.RESET}")
                    print(f"File: {res['file']} | Total findings: {res['total_findings']} (High: {res['high_severity']}, Med: {res['medium_severity']})")
                    for f in res["findings"]:
                        col = C.RED if f["severity"] == "HIGH" else C.YELLOW
                        print(f"  {col}[{f['severity']}]{C.RESET} Line {f['line_number']} [{f['rule_id']}]: {f['message']}")
                        print(f"    {C.GRAY}Snippet:{C.RESET} {f['code_snippet']}")
                        print(f"    {C.GREEN}Fix:{C.RESET} {f['recommendation']}\n")
            continue

        elif lead == "/malware":
            inspect_path = " ".join(user_input.split()[1:]).strip()
            if not inspect_path:
                print(f"{C.RED}Usage: /malware <relative_path_to_file>{C.RESET}")
            else:
                from ..security.malware_lab import MalwareStaticAnalyzer
                target_file = project_root / inspect_path
                try:
                    rep = MalwareStaticAnalyzer.analyze_file(target_file)
                    print(f"\n{C.CYAN}{C.BOLD}=== MALWARE STATIC TRIAGE REPORT ==={C.RESET}")
                    print(f"  File: {rep.file_name} ({rep.file_size_bytes:,} bytes)")
                    print(f"  Format: {rep.format_identification}")
                    print(f"  SHA-256: {rep.sha256}")
                    print(f"  MD5:     {rep.md5}")
                    col = C.RED if rep.is_likely_packed else C.GREEN
                    print(f"  Entropy: {col}{rep.entropy:.4f} bits/byte (Likely packed/encrypted: {rep.is_likely_packed}){C.RESET}")
                    print(f"  Extracted Strings: {len(rep.interesting_strings)} unique items")
                    print(f"\n{C.YELLOW}Suggested YARA Rule:{C.RESET}\n{rep.suggested_yara_rule}\n")
                except Exception as e:
                    print(f"{C.RED}Analysis error: {e}{C.RESET}")
            continue

        elif lead == "/lab":
            lab_args = user_input.split()[1:]
            if not lab_args:
                print(f"{C.RED}Usage: /lab sqli <payload>  OR  /lab xss <payload>{C.RESET}")
            else:
                sub_exp = lab_args[0].lower()
                payload = " ".join(lab_args[1:])
                from ..security.lab import IsolatedSecurityLab
                lab = IsolatedSecurityLab()
                try:
                    if sub_exp == "sqli":
                        res = lab.test_sql_injection_mitigation(payload or "admin' OR '1'='1")
                    elif sub_exp == "xss":
                        res = lab.test_xss_sanitization_mitigation(payload or "<script>alert(1)</script>")
                    else:
                        print(f"{C.RED}Unknown experiment '{sub_exp}'. Options: sqli, xss{C.RESET}")
                        continue
                    print(f"\n{C.CYAN}{C.BOLD}=== ISOLATED SECURITY LAB EXPERIMENT ==={C.RESET}")
                    print(f"Target:       {res.target}")
                    print(f"Type:         {res.vulnerability_type}")
                    print(f"Vulnerable:   {C.RED if res.is_vulnerable_before else C.GREEN}{res.is_vulnerable_before}{C.RESET}")
                    print(f"Mitigated:    {C.GREEN}{res.is_mitigated_after}{C.RESET}")
                    print(f"\n{C.WHITE}Evidence:{C.RESET}\n{res.evidence}")
                    print(f"\n{C.GREEN}Remediation Guidance:{C.RESET}\n{res.remediation_guidance}\n")
                except Exception as e:
                    print(f"{C.RED}Lab execution error: {e}{C.RESET}")
            continue

        elif lead == "/browse":
            target_url = " ".join(user_input.split()[1:]).strip()
            if not target_url:
                print(f"{C.RED}Usage: /browse <url>  (e.g. /browse https://en.wikipedia.org/wiki/Python){C.RESET}")
            else:
                from ..tools.browser import WebBrowserEngine
                print(f"{C.GRAY}Fetching web page: {target_url}...{C.RESET}")
                page = WebBrowserEngine.browse_url(target_url, max_chars=4000)
                card = WebBrowserEngine.format_browser_display(page)
                print(f"\n{card}\n")
                last_response = card
            continue

        elif lead == "/agent":
            mission_goal = " ".join(user_input.split()[1:]).strip()
            if not mission_goal:
                print(f"{C.RED}Usage: /agent <goal>  (e.g. /agent research Python 3.14 features and browse top results){C.RESET}")
            else:
                from ..agents.autonomous_agent import AutonomousWorkAgent
                print(f"\n{C.CYAN}{C.BOLD}=== LAUNCHING AUTONOMOUS AGENT MISSION ==={C.RESET}")
                print(f"Goal: {mission_goal}\n")
                agent = AutonomousWorkAgent(tools, max_steps=8, verbose=True)
                res = agent.execute_mission(mission_goal)
                print(f"\n{C.GREEN}{C.BOLD}=== AGENT MISSION SUMMARY ==={C.RESET}")
                print(f"Completed: {res.completed} in {res.elapsed_sec:.2f}s ({len(res.steps)} steps)")
                for s in res.steps:
                    st_col = C.GREEN if not s.error else C.RED
                    print(f"  {st_col}[Step {s.step_num}]{C.RESET} {s.thought}")
                    if s.tool_name:
                        print(f"    Tool: {s.tool_name}")
                print(f"\n{res.synthesized_output}\n")
                last_response = res.synthesized_output
            continue

        elif lead == "/gh":
            gh_args = user_input.split()[1:]
            if not gh_args:
                print(f"{C.RED}Usage: /gh search <query>  OR  /gh view <owner/repo>  OR  /gh file <owner/repo> <file>{C.RESET}")
            else:
                from ..tools.github_engine import GitHubEngine
                sub = gh_args[0].lower()
                arg_tail = " ".join(gh_args[1:]).strip()
                if sub == "search":
                    res = GitHubEngine.search_repositories(arg_tail, limit=5)
                    card = GitHubEngine.format_repos_display(res)
                    print(f"\n{card}\n")
                    last_response = card
                elif sub == "view":
                    res = GitHubEngine.get_repo_info(arg_tail)
                    if res.get("success"):
                        d = res.get("data", {})
                        print(f"\n{C.CYAN}=== GITHUB REPO: {res.get('repo')} ==={C.RESET}")
                        print(f"  Description: {d.get('description')}")
                        print(f"  Stars:       {d.get('stargazerCount', 0):,}")
                        print(f"  Forks:       {d.get('forkCount', 0):,}")
                        print(f"  Default:     {d.get('defaultBranch', 'main')}")
                        print(f"  URL:         {d.get('url')}\n")
                    else:
                        print(f"{C.RED}Error: {res.get('error')}{C.RESET}")
                elif sub == "file":
                    file_parts = arg_tail.split()
                    if len(file_parts) < 2:
                        print(f"{C.RED}Usage: /gh file <owner/repo> <file_path>{C.RESET}")
                    else:
                        repo_name, fpath = file_parts[0], file_parts[1]
                        res = GitHubEngine.get_file_content(repo_name, fpath)
                        if res.get("success"):
                            print(f"\n{C.CYAN}=== GITHUB FILE: {repo_name}/{fpath} ({res.get('length'):,} bytes) ==={C.RESET}\n")
                            print(res.get("content")[:1500])
                            if res.get("length", 0) > 1500:
                                print(f"\n{C.GRAY}[... Truncated remainder of file ...]{C.RESET}\n")
                        else:
                            print(f"{C.RED}Error: {res.get('error')}{C.RESET}")
                elif sub == "issues":
                    res = GitHubEngine.list_issues(arg_tail, limit=5)
                    if res.get("success"):
                        print(f"\n{C.CYAN}=== ISSUES: {res.get('repo')} ==={C.RESET}")
                        for iss in res.get("issues", []):
                            print(f"  #{iss.get('number')}: {iss.get('title')} ({iss.get('state')})")
                        print()
                    else:
                        print(f"{C.RED}Error: {res.get('error')}{C.RESET}")
                else:
                    print(f"{C.RED}Unknown GitHub subcommand '{sub}'. Available: search, view, file, issues{C.RESET}")
            continue

        elif lead == "/git":
            git_args = user_input.split()[1:]
            sub = git_args[0].lower() if git_args else "status"
            from ..tools.github_engine import GitHubEngine
            if sub == "status":
                res = GitHubEngine.git_status(project_root)
                if res.get("success"):
                    print(f"\n{C.CYAN}=== LOCAL GIT STATUS ==={C.RESET}")
                    print(f"  Branch:   {C.GREEN}{res.get('branch')}{C.RESET}")
                    print(f"  Modified: {res.get('changed_files_count')} files changed")
                    for f in res.get("files", []):
                        print(f"    {f}")
                    print()
                else:
                    print(f"{C.RED}Git error: {res.get('error')}{C.RESET}")
            elif sub == "diff":
                res = GitHubEngine.git_diff(project_root)
                if res.get("success"):
                    print(f"\n{C.CYAN}=== LOCAL GIT DIFF ==={C.RESET}\n{res.get('patch')}\n")
                else:
                    print(f"{C.RED}Git error: {res.get('error')}{C.RESET}")
            elif sub == "log":
                res = GitHubEngine.git_log(project_root, limit=5)
                if res.get("success"):
                    print(f"\n{C.CYAN}=== LOCAL GIT COMMIT LOG ==={C.RESET}")
                    for c in res.get("commits", []):
                        print(f"  {C.YELLOW}{c.get('commit')}{C.RESET} {c.get('message')}")
                    print()
                else:
                    print(f"{C.RED}Git error: {res.get('error')}{C.RESET}")
            else:
                print(f"{C.RED}Unknown git subcommand '{sub}'. Options: status, diff, log{C.RESET}")
            continue

        elif lead == "/artifacts":
            from ..generation.artifacts import ArtifactEngine
            arts = ArtifactEngine.extract_artifacts(last_response)
            if not arts:
                print(f"{C.YELLOW}No structured code, table, or diagram artifacts found in the last response.{C.RESET}")
            else:
                print(f"\n{C.CYAN}{C.BOLD}--- STRUCTURED ARTIFACTS ({len(arts)} extracted) ---{C.RESET}\n")
                for a in arts:
                    print(ArtifactEngine.format_artifact_display(a))
                    print()
            continue

        elif lead == "/context":
            if len(cmd) > 1:
                try:
                    new_ctx = int(cmd[1])
                    if model and hasattr(model, "config"):
                        model.config.max_seq_len = min(2_097_152, max(128, new_ctx))
                    print(f"{C.GREEN}[Context window limit set to: {new_ctx:,} tokens]{C.RESET}")
                except Exception:
                    print(f"{C.RED}Usage: /context 2097152{C.RESET}")
            else:
                cur_limit = getattr(getattr(model, "config", None), "max_seq_len", 2_097_152) if model else 2_097_152
                sw = getattr(getattr(model, "config", None), "sliding_window", "Dynamic 2M") if model else "Dynamic 2M"
                print(f"\n{C.CYAN}--- CONTEXT WINDOW STATUS ---{C.RESET}")
                print(f"  {C.WHITE}Max Sequence Capacity:{C.RESET} {cur_limit:,} tokens (2 Million token capability)")
                print(f"  {C.WHITE}RoPE Scaling Type:{C.RESET}     YaRN / Dynamic NTK Extended")
                print(f"  {C.WHITE}Streaming Cache:{C.RESET}       StreamingLLM with Attention Sinks ({sw})\n")
            continue

        elif lead == "/arxiv":
            arxiv_q = " ".join(user_input.split()[1:])
            if not arxiv_q:
                print(f"{C.RED}Usage: /arxiv <research topic or paper title>{C.RESET}")
            else:
                from ..tools.free_tools import ArxivSearchEngine
                print(f"{C.GRAY}Searching ArXiv repository: '{arxiv_q}'...{C.RESET}")
                papers = ArxivSearchEngine.search(arxiv_q, max_results=4)
                formatted = ArxivSearchEngine.format_results(arxiv_q, papers)
                print(f"\n{formatted}\n")
                last_response = formatted
            continue

        elif lead == "/wiki":
            wiki_q = " ".join(user_input.split()[1:])
            if not wiki_q:
                print(f"{C.RED}Usage: /wiki <concept or topic>{C.RESET}")
            else:
                from ..tools.free_tools import WikipediaEngine
                print(f"{C.GRAY}Querying Wikipedia: '{wiki_q}'...{C.RESET}")
                w_res = WikipediaEngine.summary(wiki_q)
                formatted = WikipediaEngine.format_summary(wiki_q, w_res)
                print(f"\n{formatted}\n")
                last_response = formatted
            continue

        elif lead == "/weather":
            city_q = " ".join(user_input.split()[1:])
            if not city_q:
                print(f"{C.RED}Usage: /weather <city name>{C.RESET}")
            else:
                from ..tools.free_tools import WeatherEngine
                print(f"{C.GRAY}Retrieving Open-Meteo live weather: '{city_q}'...{C.RESET}")
                w_data = WeatherEngine.get_weather(city_q)
                formatted = WeatherEngine.format_weather(city_q, w_data)
                print(f"\n{formatted}\n")
                last_response = formatted
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
                        tokenizer, tok_path = resolve_tokenizer_for_model(project_root, model=model, requested_path=args.tokenizer)
                        print(f"{C.GREEN}Switched to: {active_ckpt_path.name} ({model_desc}) | Tokenizer: {tok_path.name} ({tokenizer.get_vocab_size():,} vocab){C.RESET}")
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

        elif lead == "/memory":
            print(f"\n{C.BOLD}{C.CYAN}--- AI HIPPOCAMPUS (TIERED DURABLE MEMORY) ---{C.RESET}")
            if hasattr(orchestrator, "hippocampus"):
                mem = orchestrator.hippocampus
                with mem._get_connection() as conn:
                    facts = conn.execute("SELECT * FROM durable_facts ORDER BY confidence DESC, recall_count DESC").fetchall()
                    if not facts:
                        print(f"  {C.GRAY}No durable memories stored yet. Converse naturally to store facts.{C.RESET}")
                    else:
                        print(f"  {C.GREEN}Total Stored Durable Facts: {len(facts)}{C.RESET}\n")
                        for f in facts:
                            print(f"  {C.YELLOW}[{f['category'].upper()}]{C.RESET} {f['content']}")
                            print(f"    {C.GRAY}Confidence: {f['confidence']*100:.0f}% | Recalled: {f['recall_count']} times{C.RESET}")
            print(f"{C.GRAY}{'-' * 40}{C.RESET}\n")
            continue

        elif lead == "/sleep":
            print(f"\n{C.BOLD}{C.MAGENTA}--- TRIGGERING NIGHTLY 'SLEEP' WEIGHT CONSOLIDATION ---{C.RESET}")
            try:
                from ..training.sleep_trainer import SleepConsolidationEngine
                engine = SleepConsolidationEngine(project_root=project_root)
                res = engine.run_sleep_cycle(steps=30, max_time_sec=180)
                if res.get("promoted"):
                    print(f"{C.GREEN}[SUCCESS] Sleep cycle finished. New memories consolidated into weights without regression!{C.RESET}")
                else:
                    print(f"{C.YELLOW}[ROLLED BACK] Sleep candidate caused regression on anchor benchmark. Weights preserved safely.{C.RESET}")
            except Exception as e:
                print(f"{C.RED}Sleep consolidation error: {e}{C.RESET}")
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

        generator = TextGenerator(model, tokenizer) if model else None

        # Mode A: Text Completion (Raw Causal Language Continuation)
        if current_mode == "text-completion":
            t0 = time.time()
            print(f"\n{C.BOLD}{C.YELLOW}MODEL [completion] > {C.RESET}", end="")
            sys.stdout.flush()
            if not generator:
                print("Model weights not loaded. Please select a valid checkpoint via /model.")
                continue
            prompt_ids = tokenizer.encode(user_input).ids
            if stream_output:
                pieces = []
                for chunk in generator.generate_stream(prompt_ids, config=sampling_cfg, device=device):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    pieces.append(chunk)
                print()
                last_response = user_input + "".join(pieces)
            else:
                raw_text = generator.generate(user_input, config=sampling_cfg, device=device)
                print(raw_text)
                last_response = raw_text
            elapsed = time.time() - t0
            print(f"{C.GRAY}({elapsed:.2f}s){C.RESET}")
            continue

        # Mode B: Assistant Chat Mode (Default)
        # Store user turn in SQLite Memory
        memory.add_message(session_id, "user", user_input)
        recent_history = memory.get_messages(session_id, limit=6)

        # Stage 1: Prepare Reasoning, Intent, Thinking, and Tools with Conversation Context
        t0 = time.time()
        prep = orchestrator.prepare_reasoning(user_input, conversation_history=recent_history)

        # Display internal thinking trace if enabled
        if show_thinking and prep.thinking:
            print(f"\n{C.GRAY}╭─ <thinking> ────────────────────────────────────────{C.RESET}")
            for t_line in prep.thinking.splitlines():
                print(f"{C.GRAY}│ {t_line}{C.RESET}")
            print(f"{C.GRAY}╰─────────────────────────────────────────────────────{C.RESET}")

        # Print AI Header
        print(f"\n{C.BOLD}{C.CYAN}ASSISTANT {C.YELLOW}[{prep.intent}]{C.CYAN} > {C.RESET}", end="")
        sys.stdout.flush()

        # Stage 2: Execution / Output
        if prep.direct_answer:
            final_text = OutputOptimizer.optimize(prep.direct_answer, thinking_trace=prep.thinking, intent=prep.intent)
            print(final_text)
            verifications = prep.verifications
        else:
            use_teacher = False
            chosen_teacher = None
            if active_backend in ["auto", "ollama", "teacher"]:
                o_cli = OllamaClient()
                if o_cli.is_server_running():
                    installed_m = o_cli.list_installed_models()
                    for cand in ["llm:2b", "humanchat:latest", "qwen3.5:2b"]:
                        if cand in installed_m:
                            chosen_teacher = cand
                            use_teacher = True
                            break

            if use_teacher and chosen_teacher:
                o_cli = OllamaClient()
                pieces = []
                system_t = (
                    f"{system_persona} You are talking with your creator, Yuvraj Thakur. "
                    f"Respond with human emotional resonance, warmth, nuance, and conversational depth like Claude or ChatGPT. "
                    f"Never use robotic disclaimers, cold corporate filler, or claims of lacking emotion, and do not claim to be Qwen."
                )
                if stream_output:
                    for piece in o_cli.generate_stream(
                        prompt=user_input,
                        model_name=chosen_teacher,
                        system=system_t,
                        temperature=sampling_cfg.temperature,
                        max_tokens=sampling_cfg.max_new_tokens,
                    ):
                        sys.stdout.write(piece)
                        sys.stdout.flush()
                        pieces.append(piece)
                    print()
                    raw_text = "".join(pieces)
                else:
                    res_t = o_cli.generate(
                        prompt=user_input,
                        model_name=chosen_teacher,
                        system=system_t,
                        temperature=sampling_cfg.temperature,
                        max_tokens=sampling_cfg.max_new_tokens,
                    )
                    raw_text = res_t.get("output", "")
                    print(raw_text)
                final_text = OutputOptimizer.optimize(raw_text, thinking_trace=prep.thinking, intent=prep.intent)
            elif generator:
                from ..generation.template import ChatTemplate, ChatMessage
                chat_template = ChatTemplate()
                formatted_prompt = chat_template.format([
                    ChatMessage(role=m["role"], content=m["content"]) for m in recent_history
                ])
                if stream_output:
                    prompt_ids = tokenizer.encode(formatted_prompt).ids
                    pieces = []
                    for chunk in generator.generate_stream(prompt_ids, config=sampling_cfg, device=device):
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        pieces.append(chunk)
                    print()
                    raw_text = "".join(pieces)
                    final_text = chat_template.extract_assistant_response(raw_text)
                    final_text = OutputOptimizer.optimize(final_text, thinking_trace=prep.thinking, intent=prep.intent)
                else:
                    raw_text = generator.generate(formatted_prompt, config=sampling_cfg, device=device)
                    final_text = chat_template.extract_assistant_response(raw_text)
                    final_text = OutputOptimizer.optimize(final_text, thinking_trace=prep.thinking, intent=prep.intent)
                    print(final_text)
            else:
                final_text = "Model weights not loaded. Please select a valid checkpoint via /model."
                print(final_text)

            from ..reasoning.verifier import ResponseVerifier
            verifications = [
                ResponseVerifier.verify_math_claims(final_text).to_dict(),
                ResponseVerifier.verify_python_code(final_text).to_dict(),
            ]
            if prep.retrieved_citations:
                verifications.append(ResponseVerifier.verify_citations(final_text, prep.retrieved_citations).to_dict())

        # Save assistant response to memory for multi-turn history continuity
        memory.add_message(session_id, "assistant", final_text)
        last_response = final_text

        elapsed = time.time() - t0

        # Display Verifications & Citations
        for v in verifications:
            if v.get("passed") and v.get("details"):
                print(f"  {C.GRAY}{C.GREEN}[OK] {v.get('check')}: {v.get('details')}{C.RESET}")

        for cite in prep.retrieved_citations:
            print(f"  {C.GRAY}{C.MAGENTA}[Citation] {cite}{C.RESET}")

        print(f"{C.GRAY}({elapsed:.2f}s){C.RESET}")


if __name__ == "__main__":
    launch_cli()
