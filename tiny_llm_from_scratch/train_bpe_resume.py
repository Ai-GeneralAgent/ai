import argparse
import json
import math
import random
import time
from array import array
from pathlib import Path

import torch
from tokenizers import Tokenizer
from model_bpe import TinyBPEGPT


def prepare_token_cache(data_path, tokenizer_path, tokenizer):
    if array("H").itemsize != 2:
        raise RuntimeError("This script requires 2-byte unsigned integers.")

    cache_path = data_path.with_suffix(".bpe_tokens.u16")
    metadata_path = Path(str(cache_path) + ".json")

    data_stat = data_path.stat()
    tok_stat = tokenizer_path.stat()
    signature = {
        "data_size": data_stat.st_size,
        "data_mtime_ns": data_stat.st_mtime_ns,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_size": tok_stat.st_size,
        "tokenizer_mtime_ns": tok_stat.st_mtime_ns,
        "vocab_size": tokenizer.get_vocab_size(),
    }

    if metadata_path.is_file() and cache_path.is_file():
        try:
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                all(saved.get(k) == v for k, v in signature.items())
                and cache_path.stat().st_size % 2 == 0
                and saved.get("token_count") == cache_path.stat().st_size // 2
            ):
                print("Using cached BPE tokens:", cache_path)
                print(f"Cached tokens: {saved['token_count']:,}")
                return cache_path, saved["token_count"]
        except (OSError, ValueError, TypeError):
            pass

    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("Vocabulary is too large for this compact cache.")

    temp_path = Path(str(cache_path) + ".tmp")
    print("Building token cache in chunks. This may take a while...")
    print("Cache file:", cache_path.resolve())

    total_chars = 0
    total_tokens = 0
    next_report = 5_000_000
    chunk_limit = 250_000
    pieces = []
    piece_chars = 0

    try:
        with data_path.open("r", encoding="utf-8-sig", errors="replace") as src:
            with temp_path.open("wb") as out:
                for line in src:
                    pieces.append(line)
                    piece_chars += len(line)
                    total_chars += len(line)

                    if piece_chars >= chunk_limit:
                        ids = tokenizer.encode("".join(pieces)).ids
                        values = array("H", ids)
                        values.tofile(out)
                        total_tokens += len(values)
                        pieces.clear()
                        piece_chars = 0

                    if total_chars >= next_report:
                        print(
                            f"Read {total_chars:,} characters; "
                            f"encoded {total_tokens:,} tokens so far."
                        )
                        next_report += 5_000_000

                if pieces:
                    ids = tokenizer.encode("".join(pieces)).ids
                    values = array("H", ids)
                    values.tofile(out)
                    total_tokens += len(values)

                out.flush()

        temp_path.replace(cache_path)
        metadata = dict(signature)
        metadata["token_count"] = total_tokens
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    print(f"Dataset characters: {total_chars:,}")
    print(f"Total cached BPE tokens: {total_tokens:,}")
    print("Token cache created successfully.")
    return cache_path, total_tokens


def get_batch(file_handle, begin, end, batch_size, block_size, device):
    if end - begin <= block_size:
        raise ValueError("A dataset split is too small for this context length.")

    xs = []
    ys = []
    bytes_per_window = (block_size + 1) * 2

    for _ in range(batch_size):
        start = random.randrange(begin, end - block_size)
        file_handle.seek(start * 2)
        raw = file_handle.read(bytes_per_window)

        values = array("H")
        values.frombytes(raw)

        if len(values) != block_size + 1:
            raise IOError("Could not read a complete token window from cache.")

        window = torch.tensor(values.tolist(), dtype=torch.long)
        xs.append(window[:-1])
        ys.append(window[1:])

    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, fh, train_end, total_tokens, args, block_size, device):
    model.eval()
    scores = {}

    for name, begin, end in (
        ("train", 0, train_end),
        ("val", train_end, total_tokens),
    ):
        losses = []

        for _ in range(args.eval_batches):
            x, y = get_batch(
                fh, begin, end, args.batch_size, block_size, device
            )
            _, loss = model(x, y)
            losses.append(loss.item())

        scores[name] = sum(losses) / len(losses)

    model.train()
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Continue BPE training using a compact disk token cache."
    )
    parser.add_argument("--resume", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data_merged_expanded.txt")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if min(args.steps, args.eval_interval, args.eval_batches, args.batch_size) < 1:
        parser.error("Steps, intervals, batches, and batch size must be positive.")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")

    resume_path = Path(args.resume)
    output_path = Path(args.out)
    data_path = Path(args.data)

    if not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path.resolve()}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path.resolve()}\n"
            "Choose a different --out filename to protect it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(6, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("Device:", device)
    print("Loading checkpoint:", resume_path.resolve())
    checkpoint = torch.load(
        resume_path, map_location=device, weights_only=False
    )

    tokenizer_path = Path(
        args.tokenizer or checkpoint.get("tokenizer_path", "tokenizer_bpe.json")
    )
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path.resolve()}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = checkpoint["vocab_size"]

    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary {tokenizer.get_vocab_size()} does not "
            f"match checkpoint vocabulary {vocab_size}."
        )

    block_size = checkpoint["block_size"]
    model = TinyBPEGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cache_path, total_tokens = prepare_token_cache(
        data_path, tokenizer_path, tokenizer
    )
    if total_tokens <= block_size + 1:
        raise ValueError("Too few tokens to train.")

    train_end = int(total_tokens * 0.9)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Training tokens: {train_end:,}")
    print(f"Validation tokens: {total_tokens - train_end:,}")
    print("Starting checkpoint step:", checkpoint.get("step", 0))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate
    )
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Restored optimizer state.")
    else:
        print("Original checkpoint has no optimizer state; using a fresh optimizer.")

    initial_step = int(checkpoint.get("step", 0))
    best_val = float(checkpoint.get("val_loss", math.inf))
    started = time.time()

    def save_checkpoint(step, val_loss):
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_embd": checkpoint["n_embd"],
            "n_head": checkpoint["n_head"],
            "n_layer": checkpoint["n_layer"],
            "dropout": checkpoint.get("dropout", 0.1),
            "tokenizer_path": str(tokenizer_path),
            "step": step,
            "val_loss": val_loss,
        }, output_path)

    with cache_path.open("rb") as fh:
        scores = evaluate(
            model, fh, train_end, total_tokens, args, block_size, device
        )
        print(
            f"Initial evaluation | train {scores['train']:.4f} "
            f"| validation {scores['val']:.4f}"
        )

        # Save a separate recoverable checkpoint before additional training.
        save_checkpoint(initial_step, best_val)
        print("Saved starting checkpoint:", output_path)

        model.train()
        for local_step in range(1, args.steps + 1):
            x, y = get_batch(
                fh, 0, train_end, args.batch_size, block_size, device
            )
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if local_step % args.eval_interval == 0 or local_step == args.steps:
                scores = evaluate(
                    model, fh, train_end, total_tokens,
                    args, block_size, device
                )
                global_step = initial_step + local_step
                elapsed = (time.time() - started) / 60

                print(
                    f"Step {global_step} (+{local_step}/{args.steps}) | "
                    f"train {scores['train']:.4f} | "
                    f"validation {scores['val']:.4f} | "
                    f"time {elapsed:.1f} min"
                )

                if math.isfinite(scores["val"]) and scores["val"] < best_val:
                    best_val = scores["val"]
                    save_checkpoint(global_step, best_val)
                    print("Saved improved checkpoint:", output_path)

    print("Training complete.")
    print("Best validation loss recorded:", round(best_val, 4))
    print("Output checkpoint:", output_path.resolve())


if __name__ == "__main__":
    main()
