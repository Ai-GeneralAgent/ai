"""Interactive text-only chat for the TinyGPT character-level model."""
import argparse
import torch
from model import TinyGPT


def normalize_stoi(raw):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Checkpoint is missing its character vocabulary ('stoi').")
    return {str(ch): int(idx) for ch, idx in raw.items()}


def normalize_itos(raw, stoi):
    if isinstance(raw, dict):
        try:
            values = {int(k): str(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            values = {idx: ch for ch, idx in stoi.items()}
    elif isinstance(raw, list):
        values = {i: str(ch) for i, ch in enumerate(raw)}
    else:
        values = {idx: ch for ch, idx in stoi.items()}
    expected = {idx: ch for ch, idx in stoi.items()}
    return values if len(values) == len(expected) and all(values.get(i) == ch for i, ch in expected.items()) else expected


def main():
    parser = argparse.ArgumentParser(description="Chat with your TinyGPT model using text only.")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpt_vocab_test.pt")
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()
    if args.temperature <= 0 or args.top_k < 0 or args.tokens < 1:
        parser.error("temperature and tokens must be positive; top-k must be zero or more")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}\nRun from your project folder and check the --checkpoint path.")
    for key in ("config", "model_state_dict", "stoi"):
        if key not in ckpt:
            raise SystemExit(f"Checkpoint is missing required field: {key}")
    config = dict(ckpt["config"])
    stoi = normalize_stoi(ckpt["stoi"])
    itos = normalize_itos(ckpt.get("itos"), stoi)
    if int(config.get("vocab_size", len(stoi))) != len(stoi):
        raise SystemExit("Checkpoint vocabulary size does not match its config. Do not edit the checkpoint manually.")

    model = TinyGPT(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    top_k = None if args.top_k == 0 else min(args.top_k, len(stoi))

    print("TinyGPT text chat (no speech). Type /exit to quit. This small model may give incorrect or incomplete answers.")
    print(f"Checkpoint: {args.checkpoint} | device: {device} | context: {config.get('block_size')} characters")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            break
        prompt = f"User: {question}\nAssistant:"
        unknown = sorted(set(ch for ch in prompt if ch not in stoi))
        if unknown:
            print("Model vocabulary does not contain these character(s):", " ".join(repr(c) for c in unknown))
            continue
        input_ids = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=args.tokens, temperature=args.temperature, top_k=top_k)
        decoded = "".join(itos.get(int(i), "") for i in output[0].tolist())
        answer = decoded[len(prompt):]
        # The seed/training format uses one-line answers. Stop at a likely turn boundary.
        answer = answer.split("\n", 1)[0].split("User:", 1)[0].strip()
        print("AI:", answer if answer else "(No answer generated. Train on the chat examples first.)")


if __name__ == "__main__":
    main()
