# Modern AI Transformer & Autonomous Agent (from Scratch)

A clean, modular, production-grade Transformer language model and autonomous reasoning agent built entirely in PyTorch from scratch. Designed for efficient training and inference across **CPU, CUDA GPU, and Apple Silicon (MPS)**.

---

## 🚀 Key Features

* **Modern Decoder-Only Transformer (v2.0)**:
  * **Pre-RMSNorm**: Numerical gradient stability and faster execution.
  * **Rotary Position Embeddings (RoPE)**: Dynamic relative position encodings with context extrapolation.
  * **SwiGLU MLP**: State-of-the-art activation function for superior parameter efficiency.
  * **PyTorch SDPA Attention**: Scaled Dot-Product Attention with causal masking.
  * **Weight-Tied Embeddings**: Input token embeddings and output projection share weights for 2× faster convergence.
  * **LoRA Adapters**: Configurable low-rank adaptation ($Q, V, W_1, W_3$) for lightweight fine-tuning.
* **Deterministic Local Reasoning Engine**:
  * **Goal & Task Planner**: Structured planning with complexity estimation.
  * **Independent AST Verifiers**: Real-time syntax checking, safe mathematical validation, and citation tracking.
* **Bounded Agent Loop & Tool Registry**:
  * Typed tool schemas with execution timeouts, permission sandboxing, and cycle detection.
  * Built-in safe tools: Calculator, system datetime, path-bounded file reader, code search.
* **Local RAG (Retrieval-Augmented Generation)**:
  * Inverted index with BM25 ranking, word chunking, and automatic `[doc#chunk_X]` citation attribution.
* **Persistent SQLite Conversation Memory**:
  * Cross-session chat storage, structured message roles, and user preference persistence.
* **Compact Bounded-Memory Data Pipeline**:
  * High-speed BPE tokenization (>300,000 tokens/sec).
  * 2-byte unsigned integer (`uint16`) disk cache with `np.memmap` zero-RAM random batch sampling.

---

## 💻 Multi-Device Quickstart

This repository is designed to run seamlessly on any platform without modification.

### Option 1: Google Colab / Cloud GPU (Fastest)

Open a Google Colab notebook (or SSH into a GPU instance) and run:

```bash
# 1. Clone repository
!git clone https://github.com/Ai-GeneralAgent/ai.git
%cd ai

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Verify system tests (24/24 tests)
!python tiny_llm_from_scratch/tests/test_system.py

# 4. Train with automatic GPU acceleration (CUDA)
!python tiny_llm_from_scratch/train_modern.py --data tiny_llm_from_scratch/alice.txt --steps 1000 --batch-size 16
```

### Option 2: Linux / macOS

```bash
# 1. Clone repository
git clone https://github.com/Ai-GeneralAgent/ai.git
cd ai

# 2. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run all tests
bash tiny_llm_from_scratch/run_tests.sh

# 4. Start training
bash tiny_llm_from_scratch/run_train.sh --data alice.txt --steps 500

# 5. Launch interactive chat
bash tiny_llm_from_scratch/run_chat.sh
```

### Option 3: Windows (Command Prompt or PowerShell)

```cmd
:: 1. Clone repository
git clone https://github.com/Ai-GeneralAgent/ai.git
cd ai

:: 2. Setup virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: 3. Double-click or run batch scripts:
tiny_llm_from_scratch\run_tests.bat
tiny_llm_from_scratch\run_train.bat
tiny_llm_from_scratch\run_chat.bat
```

---

## 🔄 Training & Resuming on Another Device

### 1. Training from Scratch with Any Dataset
The system will automatically tokenize your text into memory-mapped uint16 files and train:
```bash
python tiny_llm_from_scratch/train_modern.py --data path/to/your_corpus.txt --steps 2000 --batch-size 8
```
*(If `--data` is omitted or not found, it automatically falls back to the bundled `alice.txt` dataset).*

### 2. Resuming Training
To resume training from an existing checkpoint on any device:
```bash
python tiny_llm_from_scratch/train_modern.py --resume checkpoints/modern_base_v2_best.pt --data your_corpus.txt --steps 5000
```
On Windows, you can simply run:
```cmd
tiny_llm_from_scratch\resume_train.bat
```

### 3. Hardware Auto-Detection
The training engine auto-detects hardware via `--device auto`:
* **CUDA**: Automatically enabled when an NVIDIA GPU is present.
* **MPS**: Automatically enabled on Apple Silicon Macs (M1/M2/M3/M4).
* **CPU**: Multithreaded execution optimized for Snapdragon X, Intel, and AMD processors.

---

## 🧪 System Verification

The repository includes a comprehensive 24-test suite covering every architectural module:

```bash
python tiny_llm_from_scratch/tests/test_system.py
```

* Tests covered:
  * Forward pass & backward loss backpropagation
  * Pre-RMSNorm, RoPE rotary position embeddings & SwiGLU MLP
  * Atomic `.tmp` checkpoint saving, loading & corruption protection
  * Tokenizer fingerprint validation & BPE round-trip encoding
  * Bounded-memory chunked token cache & memory-mapped batch sampling
  * BM25 document indexing, term scoring & citation verification
  * AST safe calculator validation & syntax verification
  * Tool registry schema validation, permissions & execution timeouts
  * Agent loop cycle detection & iteration cap protection
  * SQLite persistent conversation database & message history

---

## 📊 Architecture Comparison

| Component | Legacy Baseline (v1) | Modern Architecture (v2) | Benefits |
| :--- | :--- | :--- | :--- |
| **Normalization** | LayerNorm | **Pre-RMSNorm** | Faster forward/backward pass, gradient stability |
| **Positional Encoding** | Absolute Embeddings | **Rotary Embeddings (RoPE)** | Preserves relative token distance |
| **Activation / MLP** | Standard GELU Linear | **SwiGLU** | Superior parameter-to-loss efficiency |
| **Attention** | `nn.MultiheadAttention` | **PyTorch SDPA Causal** | Optimized kernel, lower peak VRAM/RAM |
| **Weight Tying** | Untied | **Tied (`tok_emb` = `lm_head`)** | 2× faster representation learning |
| **Adapters** | None | **Modular LoRA** | Parameter-efficient fine-tuning without full retrain |
| **Reasoning** | Unstructured output | **Planner + Verifier + Safe Tools** | Factuality checks, AST code checks |
| **Memory** | Ephemeral | **SQLite Persistence** | Cross-session context and user preferences |

---

## 📁 Repository Structure

```
ai/
├── requirements.txt                   # Universal dependencies (torch, tokenizers, numpy, etc.)
├── README.md                          # Project documentation and cross-device guide
└── tiny_llm_from_scratch/
    ├── src/
    │   ├── models/                    # ModernTransformer, RoPE, RMSNorm, SwiGLU, LoRA
    │   ├── data/                      # Bounded-memory chunked token cache & pipeline
    │   ├── training/                  # Trainer, CosineWarmupScheduler, TrainingMonitor
    │   ├── reasoning/                 # Task orchestrator, planner, AST verifier
    │   ├── tools/                     # Typed tool registry, sandboxed agent loop
    │   ├── rag/                       # BM25 knowledge retriever & citation tracker
    │   ├── memory/                    # SQLite conversation & session store
    │   ├── generation/                # Sampler, top-p, top-k, streaming generator
    │   ├── evaluation/                # 6-task standardized model benchmark
    │   └── ui/                        # Interactive CLI assistant
    ├── tests/
    │   └── test_system.py             # 24 mandatory unit & integration tests
    ├── tokenizer_bpe.json             # Trained BPE tokenizer (8,000 vocab)
    ├── alice.txt                      # Bundled starter training text (174 KB)
    ├── train_modern.py                # Cross-device training script (CPU/CUDA/MPS)
    ├── compare_models.py              # Side-by-side empirical benchmark
    ├── run_train.sh / run_train.bat   # 1-click training scripts
    ├── run_chat.sh / run_chat.bat     # 1-click chat assistant scripts
    └── run_tests.sh / run_tests.bat   # 1-click test runners
```

---

## 📜 License
MIT License. Created for open-source AI development and education.

