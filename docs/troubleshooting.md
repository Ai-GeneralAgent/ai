# Troubleshooting Guide: Modern Tiny LLM & Autonomous Agent

This guide documents the root causes, diagnostic steps, and concrete engineering fixes for common issues on Windows, CPU hardware (including Snapdragon X / ARM64), PyTorch runtime, and tokenization.

---

### 1. OpenBLAS Memory Allocation Retry Failure on Windows CPU

#### Symptom:
```text
OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
```

#### Root Cause:
On Windows, especially with multi-core processors (e.g. Snapdragon X / ARM64 with 8-12 cores or Intel/AMD with hyperthreading), OpenBLAS and Intel MKL spawn an unconstrained thread pool for matrix multiplications ($W_q, W_k, W_v$). When multiple tensor operations or sub-processes execute simultaneously, OpenBLAS runs out of thread memory slots.

#### Fix:
Clamp OpenBLAS, OMP, and MKL threads to 4 *before* importing `torch`:
```python
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
In PowerShell, this is automated in `launch.ps1`, `train.ps1`, and `evaluate.ps1`.

---

### 2. Windows Console Unicode / Closed Stream Error

#### Symptom:
```text
ValueError: I/O operation on closed file
UnicodeEncodeError: 'charmap' codec can't encode character ...
```

#### Root Cause:
Re-wrapping `sys.stdout.buffer` multiple times using `io.TextIOWrapper` closes the underlying Windows console file descriptor. Additionally, default Windows codepages (cp1252 / charmap) reject mathematical symbols (such as $\Delta$, $\pi$, or superscript $^2$).

#### Fix:
Use non-closing `sys.stdout.reconfigure`:
```python
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
```

---

### 3. Checkpoint Format Version Compatibility

#### Formats Supported:
* **`v2_modern`**: ModernTransformer architecture (RMSNorm, RoPE, SwiGLU, KV-Cache). Saved with architecture config dictionary, optimizer state, scheduler state, step count, and validation loss.
* **`v1_legacy_bpe`**: Legacy `TinyBPEGPT` model with learned absolute positional embeddings and Layernorm.
* **`v0_legacy_char`**: Character-level baseline models with `stoi` character dictionaries.

#### Inspection Command:
```bash
python -c "from src.models.checkpoint import inspect_checkpoint; print(inspect_checkpoint('checkpoints/modern_production_v2_best.pt'))"
```

---

### 4. Preventing Catastrophic Forgetting During SFT

#### Symptom:
After fine-tuning on a small instruction dataset (e.g. 50 chat pairs), the model forgets basic English vocabulary or generates garbled text when asked general questions.

#### Fix:
Use the **Experience Replay Buffer** built into `train_chat_sft.py`:
```bash
python tiny_llm_from_scratch/train_chat_sft.py --data instruction_examples.jsonl --replay-ratio 0.20 --steps 100
```
This mixes 20% tokens from the base pretraining knowledge corpus with every instruction batch, preserving foundational language representations while aligning responses.

---

### 5. Tokenizer & Vocabulary Mismatches

#### Symptom:
`RuntimeError: shape '[B, T, 8000]' is invalid for input of size ...`

#### Fix:
Ensure that `tokenizer_bpe.json` (8,000 vocab size) is used consistently. Always pass `vocab_size=tokenizer.get_vocab_size()` to `ModelConfig.preset_small()`.

