import argparse
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from model import TinyGPT


def normalize_text(text: str) -> str:
    """Normalize line endings but preserve punctuation and Unicode symbols."""
    import unicodedata
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    return unicodedata.normalize("NFC", text)


def normalize_stoi(raw):
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint is missing a valid 'stoi' character-to-ID mapping.")
    mapping = {str(ch): int(idx) for ch, idx in raw.items()}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Checkpoint vocabulary IDs are not a continuous range starting at 0.")
    return mapping


def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    results = {}
    model.eval()
    with torch.no_grad():
        for split_name, split_data in (("train", train_data), ("val", val_data)):
            losses = []
            if len(split_data) <= block_size + 1:
                results[split_name] = float("nan")
                continue
            for _ in range(eval_iters):
                starts = torch.randint(0, len(split_data) - block_size - 1, (batch_size,))
                x = torch.stack([split_data[i:i + block_size] for i in starts.tolist()]).to(device)
                y = torch.stack([split_data[i + 1:i + block_size + 1] for i in starts.tolist()]).to(device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                losses.append(loss.item())
            results[split_name] = sum(losses) / max(1, len(losses))
    model.train()
    return results


def load_weights_by_character(model, old_state, old_stoi, new_stoi):
    """Load matching weights and remap vocabulary rows by character.

    This allows a dataset to add new punctuation/symbols without discarding all
    learned weights. Newly added character rows retain their random initialization.
    """
    current = model.state_dict()
    old_vocab_size = len(old_stoi)
    new_vocab_size = len(new_stoi)
    copied = 0
    remapped = []
    missing_or_changed = []
    vocab_name_hints = ("token", "tok_emb", "embed", "lm_head", "vocab", "output_head")

    with torch.no_grad():
        for name, target in current.items():
            source = old_state.get(name)
            if source is None:
                missing_or_changed.append(name)
                continue
            if not torch.is_tensor(source):
                missing_or_changed.append(name)
                continue
            source = source.to(dtype=target.dtype, device=target.device)
            same_trailing_dims = (
                source.ndim >= 1 and target.ndim == source.ndim
                and tuple(source.shape[1:]) == tuple(target.shape[1:])
            )
            looks_vocab_related = any(hint in name.lower() for hint in vocab_name_hints)
            is_vocab_rows = (
                same_trailing_dims
                and source.shape[0] == old_vocab_size
                and target.shape[0] == new_vocab_size
                and (looks_vocab_related or old_vocab_size != new_vocab_size)
            )

            if is_vocab_rows:
                for char, old_id in old_stoi.items():
                    new_id = new_stoi.get(char)
                    if new_id is not None and 0 <= old_id < source.shape[0] and 0 <= new_id < target.shape[0]:
                        target[new_id].copy_(source[old_id])
                remapped.append(name)
            elif source.shape == target.shape:
                target.copy_(source)
                copied += 1
            else:
                missing_or_changed.append(
                    f"{name} (checkpoint {tuple(source.shape)} -> model {tuple(target.shape)})"
                )

    model.load_state_dict(current)
    if missing_or_changed:
        print("Note: these parameters were left at their initialized values:")
        for item in missing_or_changed[:12]:
            print("  -", item)
        if len(missing_or_changed) > 12:
            print(f"  ... and {len(missing_or_changed) - 12} more")
    if remapped:
        print("Vocabulary-shaped weights remapped by character:")
        for name in remapped:
            print("  -", name)
    print(f"Loaded {copied} same-shape parameter tensors.")


def decode_ids(ids, itos):
    return "".join(itos.get(int(i), "�") for i in ids)


def main():
    parser = argparse.ArgumentParser(description="Train or continue a small character-level GPT.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint.")
    parser.add_argument("--steps", type=int, default=1000, help="Additional training steps (when resuming).")
    parser.add_argument("--block-size", type=int, default=None, help="Context length. On resume, it must match the checkpoint.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--data", default="data.txt", help="UTF-8 text corpus path; default: data.txt")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.batch_size < 1 or args.eval_interval < 1:
        parser.error("--batch-size and --eval-interval must be positive")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset: {data_path.resolve()}")
    text = normalize_text(data_path.read_text(encoding="utf-8-sig"))
    if len(text) < 100:
        raise ValueError("Dataset is very small. Add book text before training.")

    # Dynamic character vocabulary: every character actually present in the corpus
    # is assigned an ID, including apostrophes, quotes, em dashes, ellipses, symbols,
    # accented characters, tabs, and newlines.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    print(f"Characters in dataset: {len(text):,}")
    print(f"Unique characters in vocabulary: {len(chars):,}")
    punctuation = [ch for ch in chars if __import__("unicodedata").category(ch)[0] in ("P", "S")]
    if punctuation:
        preview = " ".join(repr(ch) for ch in punctuation[:80])
        print(f"Punctuation/symbol examples ({min(80, len(punctuation))} shown): {preview}")

    split_at = int(len(text) * 0.90)
    all_ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    train_data = all_ids[:split_at]
    val_data = all_ids[split_at:]

    checkpoint_path = Path(args.checkpoint)
    old_ckpt = None
    old_stoi = None
    previous_steps = 0
    optimizer_state = None

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume was requested, but checkpoint was not found: {checkpoint_path}")
        # Only load checkpoints you created or trust.
        old_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" not in old_ckpt or "config" not in old_ckpt or "stoi" not in old_ckpt:
            raise ValueError("Checkpoint must contain model_state_dict, config, and stoi.")
        old_stoi = normalize_stoi(old_ckpt["stoi"])
        config = dict(old_ckpt["config"])
        old_block_size = int(config.get("block_size", 96))
        if args.block_size is not None and args.block_size != old_block_size:
            raise ValueError(
                f"This checkpoint uses block_size={old_block_size}, but you requested {args.block_size}. "
                "Keep the same context length for this resume run."
            )
        config["block_size"] = old_block_size
        config["vocab_size"] = len(stoi)
        model = TinyGPT(**config).to(device)
        load_weights_by_character(model, old_ckpt["model_state_dict"], old_stoi, stoi)
        previous_steps = int(old_ckpt.get("training_steps", 0))
        optimizer_state = old_ckpt.get("optimizer_state_dict")
        vocab_changed = old_stoi != stoi
        if vocab_changed:
            print(f"Vocabulary changed from {len(old_stoi)} to {len(stoi)} characters.")
            print("Existing character weights were transferred by character; new characters start with initialized weights.")
            print("Optimizer state will be reset because vocabulary-sized parameters changed.")
            optimizer_state = None
        else:
            print("Vocabulary mapping matches the checkpoint.")
        print("Loaded existing model weights.")
    else:
        block_size = args.block_size if args.block_size is not None else 96
        config = {
            "vocab_size": len(stoi),
            "block_size": block_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
        }
        model = TinyGPT(**config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Restored optimizer state.")
        except (ValueError, RuntimeError) as exc:
            print(f"Could not restore optimizer state ({exc}); starting a fresh optimizer.")
    elif args.resume:
        print("Starting a fresh optimizer.")

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Training characters: {len(train_data):,}")
    print(f"Validation characters: {len(val_data):,}")
    print(f"Parameters: {parameters:,}")
    print(f"Previously completed steps: {previous_steps}")
    print(f"Additional steps requested: {args.steps}")

    def get_batch(split):
        source = train_data if split == "train" else val_data
        if len(source) <= config["block_size"] + 1:
            raise ValueError(f"Not enough {split} data for block_size={config['block_size']}.")
        starts = torch.randint(0, len(source) - config["block_size"] - 1, (args.batch_size,))
        x = torch.stack([source[i:i + config["block_size"]] for i in starts.tolist()]).to(device)
        y = torch.stack([source[i + 1:i + config["block_size"] + 1] for i in starts.tolist()]).to(device)
        return x, y

    def save_checkpoint(step_value):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "stoi": stoi,
            # Save string keys for compatibility with generate.py's int(key) conversion.
            "itos": {str(i): ch for i, ch in itos.items()},
            "training_steps": int(step_value),
            "dataset_path": str(data_path),
            "dataset_characters": len(text),
        }
        temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(payload, temp_path)
        temp_path.replace(checkpoint_path)

    # Preserve the starting model before this run modifies the main checkpoint.
    if checkpoint_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_backup_{timestamp}{checkpoint_path.suffix}")
        shutil.copy2(checkpoint_path, backup_path)
        print(f"Starting checkpoint backed up to: {backup_path}")

    model.train()
    start_time = time.time()
    end_step = previous_steps + args.steps
    for local_step in range(args.steps):
        global_step = previous_steps + local_step
        if global_step % args.eval_interval == 0 or local_step == args.steps - 1:
            losses = estimate_loss(model, train_data, val_data, config["block_size"], args.batch_size, device)
            print(
                f"Step {global_step:6d}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            # Save at each evaluation so progress is recoverable if the run is interrupted.
            save_checkpoint(global_step)
        xb, yb = get_batch("train")
        logits, _ = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Save the final completed-step count (the last loop's global_step is end_step - 1).
    save_checkpoint(end_step)
    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f} seconds.")
    print(f"Total training steps: {end_step}")
    print(f"Saved checkpoint: {checkpoint_path}")

    model.eval()
    prompt = "The "
    prompt_ids = [stoi.get(ch, stoi.get(" ", 0)) for ch in prompt]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=300, temperature=0.7, top_k=min(20, len(stoi)))
    print("\n--- Generated text sample ---\n")
    print(decode_ids(generated[0].tolist(), itos))


if __name__ == "__main__":
    main()
