
"""Generate text using a TinyGPT checkpoint and its saved vocabulary."""

import argparse
from pathlib import Path

import torch
from model import TinyGPT


def normalize_stoi(raw):
    """Normalize a character -> token ID vocabulary."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "Checkpoint does not contain a valid character vocabulary ('stoi')."
        )

    try:
        result = {str(ch): int(idx) for ch, idx in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint 'stoi' must map characters to integer token IDs."
        ) from exc

    if len(set(result.values())) != len(result):
        raise ValueError("Checkpoint 'stoi' contains duplicate token IDs.")

    return result


def normalize_itos(raw, stoi):
    """Normalize token ID -> character, accepting common saved formats."""

    if isinstance(raw, list):
        result = {i: str(ch) for i, ch in enumerate(raw)}

    elif isinstance(raw, dict) and raw:
        # Common format: {"0": "\n", "1": " ", ...}
        try:
            result = {int(k): str(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            # Also accept a character -> ID dictionary if saved by mistake.
            try:
                result = {int(idx): str(ch) for ch, idx in raw.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Checkpoint 'itos' has an unsupported vocabulary format."
                ) from exc
    else:
        result = {idx: ch for ch, idx in stoi.items()}

    # The saved stoi mapping is the source of truth.
    expected = {idx: ch for ch, idx in stoi.items()}

    if (
        len(result) != len(stoi)
        or any(result.get(idx) != ch for idx, ch in expected.items())
    ):
        result = expected

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a TinyGPT checkpoint."
    )

    parser.add_argument(
        "--prompt",
        default="The ",
        help="Starting text for generation",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=300,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature; lower is less random",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Limit sampling to the top K tokens; use 0 to disable",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tiny_gpt.pt",
        help="Path to the saved model checkpoint",
    )

    args = parser.parse_args()

    if args.tokens < 0:
        parser.error("--tokens must be zero or greater")

    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")

    if args.top_k < 0:
        parser.error("--top-k must be zero or greater")

    ckpt_path = Path(args.checkpoint)
    project_root = Path(__file__).resolve().parent
    if not ckpt_path.is_file():
        if (project_root / ckpt_path).is_file():
            ckpt_path = project_root / ckpt_path
        elif (project_root / "checkpoints" / ckpt_path).is_file():
            ckpt_path = project_root / "checkpoints" / ckpt_path
        else:
            parser.error(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Only load checkpoints you created or trust.
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(ckpt, dict):
        raise ValueError(
            "Checkpoint must be a dictionary saved by train.py."
        )

    required_keys = ("config", "model_state_dict", "stoi")

    if not all(key in ckpt for key in required_keys):
        if "arch_version" in ckpt or ("config" in ckpt and "stoi" not in ckpt):
            raise ValueError(
                f"Checkpoint '{ckpt_path.name}' is a Modern/BPE checkpoint without character 'stoi'. "
                "Please use 'generate_bpe.py' or 'chat.py' to generate with this model."
            )
        raise ValueError(
            "Checkpoint is missing one of: config, model_state_dict, or stoi."
        )

    config = dict(ckpt["config"])
    stoi = normalize_stoi(ckpt["stoi"])
    itos = normalize_itos(ckpt.get("itos"), stoi)

    config_vocab_size = int(
        config.get("vocab_size", len(stoi))
    )

    if config_vocab_size != len(stoi):
        raise ValueError(
            f"Checkpoint config has vocab_size={config_vocab_size}, "
            f"but stoi has {len(stoi)} entries. "
            "The checkpoint may be inconsistent; do not edit it manually."
        )

    # Check whether the prompt contains unsupported characters.
    unknown = sorted(
        set(ch for ch in args.prompt if ch not in stoi)
    )

    if unknown:
        shown = " ".join(repr(ch) for ch in unknown)

        raise SystemExit(
            "Your prompt contains character(s) not in this checkpoint's "
            f"vocabulary: {shown}\n"
            "Add text containing those characters to the dataset, "
            "expand the vocabulary, and resume training."
        )

    # Build and load the model.
    model = TinyGPT(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Encode the prompt.
    ids = [stoi[ch] for ch in args.prompt]

    if not ids:
        # Prefer a space, then fall back to the first known token.
        ids = (
            [stoi[" "]]
            if " " in stoi
            else [min(stoi.values())]
        )

    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    top_k = (
        None
        if args.top_k == 0
        else min(args.top_k, len(stoi))
    )

    # Generate text without calculating gradients.
    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=top_k,
        )

    generated_text = "".join(
        itos.get(int(token_id), "�")
        for token_id in output[0].tolist()
    )

    print("\n--- Generated text ---\n")
    print(generated_text)


if __name__ == "__main__":
    main()
