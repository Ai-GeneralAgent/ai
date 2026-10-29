import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from model_bpe import TinyBPEGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="checkpoints/tiny_bpe_gpt.pt"
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", default="Artificial intelligence")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(args.checkpoint)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}. Train the model first."
        )

    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )

    tokenizer_path = (
        args.tokenizer
        or checkpoint.get("tokenizer_path", "tokenizer_bpe.json")
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

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=max(0, args.tokens),
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
