import argparse
import json
import math
import os
import random
import time

import torch

from model import TinyGPT


def main():
    parser = argparse.ArgumentParser(description="Train a tiny GPT model from randomly initialized weights.")
    parser.add_argument("--data", default="data.txt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("CPU mode: training may be slow. Try --steps 300 for a quick test.")

    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text) < 100:
        raise ValueError("Dataset is too small. Add more plain text to data.txt.")

    # Character tokenizer: every distinct character gets an integer ID.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    data = torch.tensor(encode(text), dtype=torch.long)
    if len(data) < args.block_size + 2:
        raise ValueError("Dataset must contain more characters or use a smaller --block-size.")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    if len(val_data) <= args.block_size:
        # For tiny demo corpora, use a held-out slice large enough to evaluate.
        split = max(args.block_size + 1, int(0.8 * len(data)))
        train_data, val_data = data[:split], data[split:]

    def get_batch(split_name):
        source = train_data if split_name == "train" else val_data
        max_start = len(source) - args.block_size - 1
        if max_start <= 0:
            source = train_data
            max_start = len(source) - args.block_size - 1
        ix = torch.randint(max_start + 1, (args.batch_size,))
        x = torch.stack([source[i:i + args.block_size] for i in ix])
        y = torch.stack([source[i + 1:i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = TinyGPT(
        vocab_size=len(chars),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Characters in vocabulary: {len(chars)}")
    print(f"Training characters: {len(train_data):,}; validation characters: {len(val_data):,}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training from random initialization...")

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        result = {}
        for name in ("train", "val"):
            losses = []
            for _ in range(5):
                xb, yb = get_batch(name)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            result[name] = sum(losses) / len(losses)
        model.train()
        return result

    start = time.time()
    model.train()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss()
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        if step == args.steps:
            break
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": len(chars), "block_size": args.block_size,
            "n_embd": args.n_embd, "n_head": args.n_head, "n_layer": args.n_layer,
        },
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "training_steps": args.steps,
    }
    torch.save(checkpoint, "checkpoints/tiny_gpt.pt")
    with open("checkpoints/vocab.json", "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, f, ensure_ascii=False, indent=2)
    print(f"Saved: checkpoints/tiny_gpt.pt")
    print(f"Elapsed: {(time.time() - start):.1f} seconds")
    prompt = "The "
    prompt_ids = [stoi.get(c, 0) for c in prompt]
    model.eval()
    sample = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=250)[0].tolist()
    print("\n--- Sample generated text ---")
    print("".join(itos[i] for i in sample))


if __name__ == "__main__":
    main()
