from pathlib import Path
import torch

source = Path("checkpoints/tiny_gpt_vocab_test.pt")
destination = Path("checkpoints/tiny_gpt_context512.pt")
new_block_size = 512

if not source.is_file():
    raise SystemExit(f"Checkpoint not found: {source}")

checkpoint = torch.load(source, map_location="cpu", weights_only=False)

if "config" not in checkpoint or "model_state_dict" not in checkpoint:
    raise SystemExit("Checkpoint is missing config or model_state_dict.")

config = dict(checkpoint["config"])
old_block_size = int(config.get("block_size", 0))

if old_block_size <= 0:
    raise SystemExit("Could not determine the old context length.")

if old_block_size >= new_block_size:
    raise SystemExit(
        f"Existing context is {old_block_size}; "
        f"target context is {new_block_size}."
    )

state = dict(checkpoint["model_state_dict"])

# Locate the learned positional embedding matrix.
position_keys = [
    key for key, value in state.items()
    if isinstance(value, torch.Tensor)
    and value.ndim == 2
    and value.shape[0] == old_block_size
    and any(term in key.lower() for term in (
        "position", "pos_emb", "pos_embed", "positional"
    ))
]

if not position_keys:
    print("Could not safely identify the positional embedding.")
    print("State-dictionary keys:")
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}")
    raise SystemExit(
        "No changes saved. Inspect model.py before resizing."
    )

for key in position_keys:
    old_weight = state[key]
    expanded = torch.empty(
        (new_block_size, *old_weight.shape[1:]),
        dtype=old_weight.dtype
    )

    # Initialize new positions, preserving learned old positions.
    if old_weight.is_floating_point():
        torch.nn.init.normal_(expanded, mean=0.0, std=0.02)
    else:
        expanded.zero_()

    expanded[:old_block_size].copy_(old_weight)
    state[key] = expanded
    print(
        f"Expanded {key}: {tuple(old_weight.shape)} "
        f"-> {tuple(expanded.shape)}"
    )

config["block_size"] = new_block_size
checkpoint["config"] = config
checkpoint["model_state_dict"] = state

# The optimizer's saved tensors may have old parameter shapes.
# Start with a fresh optimizer while preserving model weights.
checkpoint.pop("optimizer_state_dict", None)

torch.save(checkpoint, destination)

print(f"Old context: {old_block_size} characters")
print(f"New context: {new_block_size} characters")
print(f"Resized checkpoint saved to: {destination}")
print("Original checkpoint was not modified.")
