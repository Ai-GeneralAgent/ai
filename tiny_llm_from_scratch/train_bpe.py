import argparse
import math
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from model_bpe import TinyBPEGPT


def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(
        0, len(data) - block_size - 1, (batch_size,)
    )
    x = torch.stack([
        data[int(i):int(i) + block_size] for i in starts
    ])
    y = torch.stack([
        data[int(i) + 1:int(i) + block_size + 1] for i in starts
    ])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, train_data, val_data, batch_size,
             block_size, device, eval_batches):
    model.eval()
    result = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data),
    ]:
        losses = []

        for _ in range(eval_batches):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())

        result[name] = sum(losses) / len(losses)

    model.train()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_merged_updated.txt")
    parser.add_argument("--tokenizer", default="tokenizer_bpe.json")
    parser.add_argument("--out", default="checkpoints/tiny_bpe_gpt.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(6)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer}"
        )

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}"
        )

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print("Vocabulary size:", vocab_size)

    print("Reading dataset...")
    text = Path(args.data).read_text(
        encoding="utf-8", errors="replace"
    )
    print("Characters:", f"{len(text):,}")

    print("Encoding text into BPE tokens...")
    ids = tokenizer.encode(text).ids
    del text

    if len(ids) < args.block_size * 2 + 2:
        raise ValueError("Dataset is too small for this context length.")

    tokens = torch.tensor(ids, dtype=torch.long)
    del ids

    split = int(len(tokens) * 0.90)
    train_data = tokens[:split]
    val_data = tokens[split:]

    print("Total tokens:", f"{len(tokens):,}")
    print("Training tokens:", f"{len(train_data):,}")
    print("Validation tokens:", f"{len(val_data):,}")

    model = TinyBPEGPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)

    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    started = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(
            train_data, args.batch_size, args.block_size, device
        )

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            scores = evaluate(
                model, train_data, val_data,
                args.batch_size, args.block_size,
                device, args.eval_batches,
            )

            elapsed = (time.time() - started) / 60
            print(
                f"Step {step}/{args.steps} | "
                f"train loss {scores['train']:.4f} | "
                f"validation loss {scores['val']:.4f} | "
                f"time {elapsed:.1f} min"
            )

            if scores["val"] < best_val:
                best_val = scores["val"]

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "block_size": args.block_size,
                    "n_embd": args.n_embd,
                    "n_head": args.n_head,
                    "n_layer": args.n_layer,
                    "dropout": args.dropout,
                    "tokenizer_path": args.tokenizer,
                    "step": step,
                    "val_loss": best_val,
                }, output)

                print("Saved best checkpoint:", output)

    print("Training complete.")
    print("Best validation loss:", round(best_val, 4))


if __name__ == "__main__":
    main()
