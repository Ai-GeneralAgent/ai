import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from model_bpe import TinyBPEGPT


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with the trained BPE Transformer."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tiny_bpe_gpt.pt",
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", default="Artificial intelligence")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--repeat-last", type=int, default=64)
    args = parser.parse_args()

    if args.tokens < 0:
        parser.error("--tokens must be >= 0")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.repetition_penalty < 1:
        parser.error("--repetition-penalty must be >= 1")
    if args.repeat_last < 0:
        parser.error("--repeat-last must be >= 0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(args.checkpoint)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}"
        )

    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )

    tokenizer_path = args.tokenizer or checkpoint.get(
        "tokenizer_path", "tokenizer_bpe.json"
    )
    if not Path(tokenizer_path).is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {Path(tokenizer_path).resolve()}"
        )

    tokenizer = Tokenizer.from_file(tokenizer_path)

    model = TinyBPEGPT(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens.")

    # Keep only the most recent context tokens if the prompt is too long.
    idx = torch.tensor(
        [prompt_ids[-model.block_size:]],
        dtype=torch.long,
        device=device,
    )
    generated = []

    with torch.inference_mode():
        for _ in range(args.tokens):
            context = idx[:, -model.block_size:]
            logits, _ = model(context)
            logits = logits[:, -1, :] / args.temperature

            # Penalize tokens that appeared recently.
            if args.repetition_penalty > 1 and args.repeat_last > 0:
                recent = idx[0, -args.repeat_last:].unique()
                for token_id in recent.tolist():
                    score = logits[0, token_id]
                    if score < 0:
                        logits[0, token_id] = score * args.repetition_penalty
                    else:
                        logits[0, token_id] = score / args.repetition_penalty

            # Top-k filtering.
            if 0 < args.top_k < logits.size(-1):
                threshold = torch.topk(logits, args.top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Nucleus (top-p) filtering.
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True, dim=-1
                )
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                remove = cumulative_probs > args.top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False

                sorted_logits = sorted_logits.masked_fill(
                    remove, float("-inf")
                )
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, sorted_indices, sorted_logits)
                logits = filtered

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            generated.append(next_token.item())

    print("\n--- Generated text ---\n")
    print(tokenizer.decode(prompt_ids))
    print(tokenizer.decode(generated))
    print("\n--- End ---")


if __name__ == "__main__":
    main()
