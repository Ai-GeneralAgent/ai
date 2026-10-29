"""Fine-tune TinyGPT on short User/Assistant examples while mixing in base text."""
import argparse
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from model import TinyGPT


def norm_stoi(raw):
    return {str(ch): int(idx) for ch, idx in raw.items()}


def load_examples(path):
    raw = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
    examples = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 2 or not lines[0].startswith("User: ") or not lines[1].startswith("Assistant: "):
            continue
        examples.append("\n".join(lines[:2]) + "\n\n")
    if len(examples) < 10:
        raise ValueError(f"Only {len(examples)} usable examples found in {path}; need at least 10.")
    return examples


def encode(text, stoi, label):
    missing = sorted(set(text) - set(stoi))
    if missing:
        shown = " ".join(repr(ch) for ch in missing[:30])
        raise ValueError(f"{label} contains characters not in checkpoint vocabulary: {shown}. Use ASCII text or expand vocabulary deliberately.")
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def make_batch(source, block, batch_size, device):
    if len(source) <= block + 1:
        raise ValueError("Training/validation text is too short for this checkpoint's context length.")
    starts = torch.randint(0, len(source) - block - 1, (batch_size,))
    x = torch.stack([source[int(i):int(i) + block] for i in starts])
    y = torch.stack([source[int(i) + 1:int(i) + block + 1] for i in starts])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TinyGPT for short text Q&A and writing prompts.")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpt_vocab_test.pt")
    parser.add_argument("--examples", default="qa_seed.txt")
    parser.add_argument("--base-data", default="data_merged.txt")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--qa-ratio", type=float, default=0.75, help="Fraction of training batch rows sampled from chat examples")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_interval < 1:
        parser.error("steps, batch-size, and eval-interval must be positive")
    if not 0.1 <= args.qa_ratio <= 1.0:
        parser.error("qa-ratio must be between 0.1 and 1.0")

    torch.manual_seed(2026)
    random.seed(2026)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}. This script never creates a model from scratch.")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key in ("config", "model_state_dict", "stoi"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint lacks required field: {key}")
    config = dict(ckpt["config"])
    stoi = norm_stoi(ckpt["stoi"])
    if int(config.get("vocab_size", len(stoi))) != len(stoi):
        raise ValueError("Checkpoint config and vocabulary disagree.")
    examples = load_examples(args.examples)
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    qa_train_text = "".join(examples[:split])
    qa_val_text = "".join(examples[split:])
    if len(qa_val_text) < int(config["block_size"]) + 2:
        # Keep enough material for random validation windows on small starter datasets.
        qa_val_text = "".join(examples[-max(3, len(examples)//10):])
    qa_train = encode(qa_train_text, stoi, "Chat examples")
    qa_val = encode(qa_val_text, stoi, "Validation examples")

    base_path = Path(args.base_data)
    base_train = None
    if base_path.is_file():
        base_text = base_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        base_ids = encode(base_text, stoi, "Base corpus")
        if len(base_ids) > int(config["block_size"]) + 2:
            cut = int(len(base_ids) * 0.9)
            base_train = base_ids[:cut]
        print(f"Base corpus: {len(base_ids):,} chars; batches will mix in general prose to reduce forgetting.")
    else:
        print(f"WARNING: base corpus {base_path} not found. Fine-tuning on Q&A alone can make general text generation worse.")

    model = TinyGPT(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    block = int(config["block_size"])
    top_k = args.qa_ratio
    print(f"Examples: {len(examples)} | Q&A training chars: {len(qa_train):,} | validation chars: {len(qa_val):,}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} | context: {block} characters | device: {device}")
    print(f"Target steps: {args.steps} | Q&A batch ratio: {args.qa_ratio:.0%}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = checkpoint_path.with_name(f"{checkpoint_path.stem}_before_chat_{timestamp}{checkpoint_path.suffix}")
    shutil.copy2(checkpoint_path, backup)
    print(f"Backup created: {backup}")

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for _ in range(12):
            x, y = make_batch(qa_val, block, args.batch_size, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        model.train()
        return sum(losses) / len(losses)

    started = time.time()
    for step in range(1, args.steps + 1):
        rows_x, rows_y = [], []
        for _ in range(args.batch_size):
            use_qa = base_train is None or random.random() < args.qa_ratio
            source = qa_train if use_qa else base_train
            x, y = make_batch(source, block, 1, device)
            rows_x.append(x[0])
            rows_y.append(y[0])
        xb, yb = torch.stack(rows_x), torch.stack(rows_y)
        model.train()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            vloss = val_loss()
            print(f"Step {step:5d}/{args.steps}: train loss {loss.item():.4f} | Q&A val loss {vloss:.4f}")
            payload = dict(ckpt)
            payload.update({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "stoi": stoi,
                "training_steps": int(ckpt.get("training_steps", 0)) + step,
                "chat_finetune_steps": step,
                "chat_examples_path": str(Path(args.examples)),
            })
            if "itos" not in payload:
                payload["itos"] = {str(i): ch for ch, i in stoi.items()}
            tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(payload, tmp)
            tmp.replace(checkpoint_path)
    print(f"Done in {time.time() - started:.1f}s. Updated checkpoint: {checkpoint_path}")
    print("Try: python chat.py --checkpoint " + str(checkpoint_path))


if __name__ == "__main__":
    main()
