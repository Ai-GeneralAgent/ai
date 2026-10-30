#!/usr/bin/env python3
# Modified: renamed and adapted for this project by Yuvraj Thakur.
"""
Standalone LLM Text Generation & Interactive Chat CLI
Features:
- Streaming autoregressive generation
- Temperature, top-p, top-k, repetition penalty
- ChatML multi-turn conversation memory
"""
import argparse
from pathlib import Path
import sys
import time

# Ensure UTF-8 stdout on Windows safely without crashing on subwords/emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

if __package__ is None or __package__ == "":
    pkg_dir = Path(__file__).resolve().parent
    root_dir = pkg_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from llm.configuration_llm import LLMConfig
    from llm.modeling_llm import LLMForCausalLM
    from llm.tokenizer import LLMTokenizer
    from llm.convert_weights import load_llm_checkpoint
else:
    from .configuration_llm import LLMConfig
    from .modeling_llm import LLMForCausalLM
    from .tokenizer import LLMTokenizer
    from .convert_weights import load_llm_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Standalone LLM Generation & Chat")
    parser.add_argument("--checkpoint", default="checkpoints/modern_chat_finetuned_best.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer JSON")
    parser.add_argument("--prompt", default=None, help="One-shot prompt to generate from")
    parser.add_argument("--raw", action="store_true", help="Raw unconstrained generation bypassing ChatML template and persona")
    parser.add_argument("--max-tokens", type=int, default=120, help="Max new tokens")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--rep-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda)")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    print(f"[LLM] Running on device: {device.upper()}")

    # Resolve paths relative to tiny_llm_from_scratch if relative
    root = Path(__file__).resolve().parent.parent
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / "tiny_llm_from_scratch" / ckpt_path
        if not ckpt_path.is_file():
            ckpt_path = root / args.checkpoint

    if args.tokenizer:
        tok_path = Path(args.tokenizer)
        if not tok_path.is_absolute():
            tok_path = root / "tiny_llm_from_scratch" / tok_path
            if not tok_path.is_file():
                tok_path = root / args.tokenizer
    else:
        use_v2 = True
        if ckpt_path.is_file():
            try:
                ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state = ckpt_meta.get("model_state_dict", ckpt_meta)
                emb = state.get("model.embed_tokens.weight", state.get("tok_embeddings.weight"))
                if emb is not None and emb.shape[0] == 10000:
                    use_v2 = False
            except Exception:
                pass

        if not use_v2 and (root / "tokenizer_bpe.json").is_file():
            tok_path = root / "tokenizer_bpe.json"
        elif (root / "tokenizer_bpe_v2.json").is_file():
            tok_path = root / "tokenizer_bpe_v2.json"
        elif (root / "tiny_llm_from_scratch" / "tokenizer_bpe_v2.json").is_file():
            tok_path = root / "tiny_llm_from_scratch" / "tokenizer_bpe_v2.json"
        else:
            tok_path = root / "tokenizer_bpe.json"

    tokenizer = LLMTokenizer(tok_path)
    print(f"[TOKENIZER] Loaded tokenizer ({tokenizer.vocab_size:,} vocabulary)")

    config = LLMConfig.preset_tiny(vocab_size=tokenizer.vocab_size)
    if ckpt_path.is_file():
        print(f"[MODEL] Loading weights from: {ckpt_path.name}")
        model = load_llm_checkpoint(ckpt_path, config=config, device=device)
    else:
        print("[MODEL] Checkpoint not found; initializing model from config...")
        model = LLMForCausalLM(config).to(device)

    # 1. One-shot Generation
    if args.prompt:
        if args.raw:
            print("[MODE] RAW UNCONSTRAINED GENERATION (Full Capability - No System Wrappers)")
            prompt_text = args.prompt
        else:
            messages = [
                {"role": "system", "content": "Your name is LLM. Built by Yuvraj Thakur. Reply friendly and short."},
                {"role": "user", "content": args.prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)

        print("\n" + "=" * 60)
        print(f"Prompt: {args.prompt}")
        print("=" * 60)
        print("LLM > ", end="", flush=True)

        def stream_tok(tok_id):
            txt = tokenizer.decode([tok_id])
            sys.stdout.write(txt)
            sys.stdout.flush()

        model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            stop_token_ids=[tokenizer.eos_token_id],
            stream_callback=stream_tok,
        )
        print("\n")
        return

    # 2. Interactive Terminal Chat Loop
    print("\n" + "=" * 60)
    print("  STANDALONE LLM CHAT — Built by Yuvraj Thakur")
    print("  Type '/exit' to quit.")
    print("=" * 60)

    messages = [{"role": "system", "content": "Your name is LLM, built by Yuvraj Thakur. Friendly, short, casual chat."}]

    while True:
        try:
            user_in = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_in:
            continue
        if user_in.lower() in ["/exit", "/quit", "exit"]:
            print("Catch you later!")
            break

        messages.append({"role": "user", "content": user_in})
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)

        print(f"\nLLM > ", end="", flush=True)
        gen_tokens = []

        def stream_tok(tok_id):
            gen_tokens.append(tok_id)
            txt = tokenizer.decode([tok_id])
            sys.stdout.write(txt)
            sys.stdout.flush()

        model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            stop_token_ids=[tokenizer.eos_token_id],
            stream_callback=stream_tok,
        )
        print()
        assistant_reply = tokenizer.decode(gen_tokens, skip_special=True)
        messages.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()

