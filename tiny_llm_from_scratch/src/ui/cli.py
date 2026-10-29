import argparse
import os
from pathlib import Path
import sys
import time

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
from tokenizers import Tokenizer

from ..models.checkpoint import load_modern_checkpoint, inspect_checkpoint
from ..models.transformer import ModernTransformer
from ..generation.sampler import TextGenerator, SamplingConfig
from ..generation.template import ChatTemplate, ChatMessage
from ..tools.registry import ToolRegistry
from ..tools.builtin import register_builtin_tools
from ..reasoning.orchestrator import ReasoningOrchestrator
from ..rag.pipeline import LocalRAGPipeline
from ..memory.conversation_db import ConversationDatabase


def launch_cli():
    parser = argparse.ArgumentParser(description="Modern Tiny LLM AI Assistant (v2.0.0)")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_bpe_expanded_continued_v2.pt")
    parser.add_argument("--tokenizer", default="tokenizer_bpe.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--rag", action="store_true", help="Enable document retrieval")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    ckpt_path = project_root / args.checkpoint
    tok_path = project_root / args.tokenizer
    db_path = project_root / "memory.db"

    print("=" * 65)
    print("[AI ASSISTANT] MODERN TINY LLM AI ASSISTANT (v2.0.0)")
    print("=" * 65)

    device = "cpu"
    print(f"Device: {device} | Checkpoint: {ckpt_path.name}")

    # Load Tokenizer
    if not tok_path.is_file():
        print(f"ERROR: Tokenizer not found at {tok_path}")
        return
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Initialize Memory, RAG, Tools, and Reasoning
    memory = ConversationDatabase(db_path)
    session_id = f"session_{int(time.time())}"
    memory.create_session(session_id, "Interactive CLI Session")

    tools = ToolRegistry()
    register_builtin_tools(tools, project_root)

    rag = LocalRAGPipeline()
    knowledge_dir = project_root / "data_knowledge"
    if knowledge_dir.is_dir():
        docs_to_index = list(knowledge_dir.rglob("*.txt"))
        for doc in docs_to_index[:100]:
            rag.add_document(doc)
        print(f"RAG Index: indexed {len(rag.chunks):,} chunks from {len(docs_to_index[:100])} knowledge documents.")
    elif (project_root / "alice.txt").is_file():
        sample_doc = project_root / "alice.txt"
        rag.add_document(sample_doc)
        print(f"RAG Index: loaded {len(rag.chunks)} chunks from {sample_doc.name}")

    orchestrator = ReasoningOrchestrator(rag_pipeline=rag if args.rag else None, tool_registry=tools)

    # Checkpoint Inspection & Loading
    model = None
    info = inspect_checkpoint(ckpt_path) if ckpt_path.is_file() else {}
    ckpt_format = info.get("format", "unknown")

    if ckpt_format == "v2_modern":
        model, _ = load_modern_checkpoint(ckpt_path, device=device)
        print("Loaded Modern Transformer (v2.0.0)")
    elif ckpt_format == "v1_legacy_bpe":
        print(f"Loading Legacy BPE Transformer ({info.get('vocab_size')} vocab, step {info.get('step')})...")
        from model_bpe import TinyBPEGPT
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = TinyBPEGPT(
            vocab_size=ckpt_data["vocab_size"],
            block_size=ckpt_data["block_size"],
            n_embd=ckpt_data["n_embd"],
            n_head=ckpt_data["n_head"],
            n_layer=ckpt_data["n_layer"],
            dropout=ckpt_data.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.eval()
    else:
        print(f"Warning: Checkpoint {ckpt_path.name} not found or unrecognized. Running in tool-only mode.")

    generator = TextGenerator(model, tokenizer) if model else None
    sampling_cfg = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=120,
    )

    print("\nCommands: /exit to quit | /rag [on|off] | /temp <val> | /history | /tools")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session...")
            break

        if not user_input:
            continue
        if user_input.lower() in ["/exit", "exit", "quit"]:
            break
        elif user_input.startswith("/rag"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() == "on":
                orchestrator.rag = rag
                print("[RAG enabled]")
            else:
                orchestrator.rag = None
                print("[RAG disabled]")
            continue
        elif user_input.startswith("/temp"):
            try:
                val = float(user_input.split()[1])
                sampling_cfg.temperature = val
                print(f"[Temperature set to {val}]")
            except Exception:
                print("Usage: /temp 0.7")
            continue
        elif user_input.lower() == "/tools":
            for t in tools.list_tools():
                print(f"  - {t['name']}: {t['description']}")
            continue
        elif user_input.lower() == "/history":
            for m in memory.get_messages(session_id, limit=10):
                print(f"[{m['role'].upper()}]: {m['content']}")
            continue

        memory.add_message(session_id, "user", user_input)

        # Generate through reasoning orchestrator
        def run_generation(prompt: str) -> str:
            if not generator:
                return "Model backend unavailable."
            return generator.generate(prompt, config=sampling_cfg)

        summary = orchestrator.orchestrate(user_input, generator_fn=run_generation)

        print(f"\nAI [{summary.intent}] > {summary.final_answer}")
        memory.add_message(session_id, "assistant", summary.final_answer)


if __name__ == "__main__":
    launch_cli()
