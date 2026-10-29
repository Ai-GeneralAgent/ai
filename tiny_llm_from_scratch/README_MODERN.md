# Modern Tiny LLM AI System (v2.0.0)

A modular, testable, locally runnable Transformer architecture and autonomous agent framework designed for CPU-only training and local inference.

---

## 1. System Architecture Overview

```
tiny_llm_from_scratch/
├── src/
│   ├── models/            # Versioned Transformer (v2.0), RMSNorm, RoPE, SwiGLU, LoRA
│   │   ├── config.py      # Dataclass-driven configurations (tiny, small, standard)
│   │   ├── transformer.py # Modern decoder-only transformer with SDPA attention
│   │   ├── lora.py        # Low-rank adapters (rank r, alpha, freeze base weights)
│   │   └── checkpoint.py  # Atomic serialization, metadata, versioned loaders
│   ├── data/              # Bounded-memory data pipeline & compact token cache
│   │   ├── manifest.py    # Document provenance, domain labels, split tracking
│   │   ├── pipeline.py    # Extraction (TXT, MD, JSONL, PDF), normalization, dedup
│   │   └── token_cache.py # Compact uint16 array, memmap zero-RAM random batch sampling
│   ├── training/          # Full training & resume pipeline
│   │   ├── trainer.py     # AdamW, cosine warmup, grad clipping, atomic checkpointing
│   │   ├── monitor.py     # Metrics logger (JSONL), perplexity, steps/s, overfitting check
│   │   └── finetune.py    # Supervised instruction fine-tuning with prompt loss masking
│   ├── reasoning/         # Structured reasoning, planning, and verification
│   │   ├── orchestrator.py# Intent classification, complexity estimation, lifecycle
│   │   ├── planner.py     # Task planning (goals, constraints, subtasks)
│   │   └── verifier.py    # Independent checks: AST code syntax, math, citations
│   ├── tools/             # Typed tool framework & bounded agent loop
│   │   ├── registry.py    # Tool schema validation, permission checks, timeouts
│   │   ├── builtin.py     # AST-safe calculator, datetime, file reader, code search
│   │   └── agent_loop.py  # Cycle detection, max iterations, error containment
│   ├── rag/               # Deterministic local knowledge retrieval
│   │   └── pipeline.py    # Word chunking, inverted index, BM25 scoring, citations
│   ├── memory/            # Local conversation store
│   │   └── conversation_db.py # SQLite sessions, message history, user preferences
│   ├── generation/        # Advanced generation engine
│   │   ├── sampler.py     # Streaming, top-p, top-k, repetition & frequency penalties
│   │   └── template.py    # Versioned chat templates (User, Assistant, System)
│   ├── evaluation/        # Objective benchmark suite
│   │   └── benchmark.py   # Versioned tasks (fluency, instruction, math, code, etc.)
│   └── ui/
│       └── cli.py         # Terminal chat interface with RAG, tools, and memory
├── tests/
│   └── test_system.py     # 24 mandatory system unit & integration tests
├── run_chat.bat           # Launch interactive chat assistant
├── run_train.bat          # Launch training from scratch
├── run_tests.bat          # Execute test suite
├── run_benchmark.bat      # Run side-by-side empirical benchmark
└── compare_models.py      # Compare Legacy v1 vs Modern v2 checkpoints
```

---

## 2. Key Upgrades

### Modern Transformer vs Legacy Architecture
| Feature | Legacy BPE Model (v1) | Modern Model (v2) | Impact |
| :--- | :--- | :--- | :--- |
| **Normalization** | Post/Pre LayerNorm | **Pre RMSNorm** | Faster execution, gradient stability |
| **Positional Embedding** | Learned Absolute Pos (`nn.Embedding`) | **Rotary Position Embedding (RoPE)** | Better relative position extrapolation |
| **Feed-Forward Network** | Standard GELU Linear MLP | **SwiGLU MLP** | Superior parameter efficiency |
| **Attention** | `nn.MultiheadAttention` | **PyTorch SDPA with Causal Mask** | Lower memory footprint, optimized CPU kernel |
| **Embeddings** | Un-tied / tied | **Weight-Tied (`tok_emb` = `lm_head`)** | 2× faster representation learning |
| **LoRA Adapters** | None | **Configurable LoRA on $Q, V, W_1, W_3$** | Parameter-efficient fine-tuning |

### Checkpoint Compatibility
* **Backward Compatible**: The system detects legacy checkpoints (`v1_legacy_bpe` and `v0_legacy_char`) and loads them safely using legacy model definitions (`TinyBPEGPT` and `TinyGPT`).
* **Format Protection**: Every modern checkpoint (`v2_modern`) records full architectural parameters, vocabulary size, tokenizer fingerprints, step numbers, and optimizer states.

---

## 3. Quickstart & Usage

### 1. Launch Interactive Chat Interface
```bat
run_chat.bat
```
Inside the chat, interact normally or use commands:
* `/rag on` or `/rag off` — Toggle local document retrieval
* `/temp 0.6` — Adjust generation temperature
* `/tools` — List all registered safe tools
* `/history` — Review conversation memory from SQLite
* `/exit` — Quit

### 2. Run Automated System Tests (24 Mandatory Tests)
```bat
run_tests.bat
```
*(Runs test suite covering model construction, backpropagation, atomic save/load, tokenizer compatibility, chunked tokenization, cache recovery, document retrieval, AST math verification, tool schema validation, agent loop cycle detection, and SQLite persistence).*

### 3. Run Side-by-Side Model Benchmark
```bat
run_benchmark.bat
```
Evaluates both Legacy (`v1`) and Modern (`v2`) models against standardized tasks:
* English Fluency
* Instruction Following
* Mathematics
* Code Generation
* Summarization
* Tool Selection

### 4. Train from Scratch
```bat
run_train.bat
```
Trains a Modern Transformer with cosine learning rate warmup, AdamW optimizer, and gradient clipping on local CPU.

---

## 4. Hardware Budget & Optimization Profiles
Optimized for local 8-core CPU, Snapdragon X / Qualcomm Oryon, and ~15.6 GB RAM:
* **Tiny Profile** (~1.4M params): Ideal for rapid iteration, unit tests, and instant CPU inference.
* **Small Profile** (~2.8M params): Matches existing checkpoint capacity with significantly better parameter efficiency via SwiGLU.
* **Standard Profile** (~9.5M params): Maximum capacity for local CPU multi-threading.

