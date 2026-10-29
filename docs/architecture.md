# Technical Architecture: Modern Tiny LLM & Autonomous Agent (v2.2.0)

## 1. System Topology Overview

```
                               ┌──────────────────────────────────────────────┐
                               │       User Interface (Terminal CLI / Web)     │
                               └──────────────────────┬───────────────────────┘
                                                      │
                       ┌──────────────────────────────▼──────────────────────────────┐
                       │           Reasoning Orchestrator (Chain-of-Thought)          │
                       │   Intent Classifier -> Complexity -> Subtask Task Planner    │
                       └──────────────┬───────────────────────────────┬───────────────┘
                                      │                               │
             ┌────────────────────────▼────────┐     ┌────────────────▼────────────────┐
             │   Deterministic Tool Execution  │     │   Knowledge Retrieval (RAG)     │
             │   - Safe Python AST Sandbox     │     │   - Inverted BM25 Chunk Index   │
             │   - Advanced Math & Algebra     │     │   - Document Chunk Extraction   │
             │   - Curriculum Study Engine     │     │   - Automatic Citation Tracker  │
             │   - System Datetime / Inspection│     └────────────────┬────────────────┘
             └────────────────┬────────────────┘                      │
                              │                                       │
                       ┌──────▼───────────────────────────────────────▼──────────────┐
                       │          Backend Manager & Autoregressive Decoding           │
                       │   - From-Scratch ModernTransformer (KV-Cache O(1) step)      │
                       │   - Optional Local Ollama Backend (Teacher / Pretrained)     │
                       │   - Independent Response Verifier (AST, Math, Citations)     │
                       └──────────────────────────────┬───────────────────────────────┘
                                                      │
                               ┌──────────────────────▼───────────────────────┐
                               │   Persistent SQLite Conversation Memory DB   │
                               └──────────────────────────────────────────────┘
```

---

## 2. Transformer Core Architecture (`src/models/transformer.py`)

* **Decoder-Only Causal Transformer**: $L$ stacked Transformer blocks with tied input/output embedding matrices.
* **Pre-RMSNorm**: Root-Mean-Square layer normalization with learnable gain $\gamma \in \mathbb{R}^d$ and numerical epsilon $\epsilon = 10^{-6}$, eliminating mean computation overhead:
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
* **Rotary Position Embeddings (RoPE)**: Relative position encoding applied directly to query and key head vectors in 2D coordinate pairs. Dynamic frequency extrapolation scales $\theta$ when sequence length exceeds base block size, allowing context expansion to 1,024+ tokens.
* **SwiGLU Feed-Forward Network**:
  $$\text{SwiGLU}(x) = \left( \text{Swish}(x W_1) \otimes (x W_3) \right) W_2$$
* **Key-Value (KV) Cache**: Caches past keys and values across attention layers, converting autoregressive token generation from $O(N^2)$ to $O(1)$ constant time per step.
* **LoRA Adapters (`src/models/lora.py`)**: Parameter-efficient low-rank decomposition ($W = W_0 + \frac{\alpha}{r} B A$) for lightweight fine-tuning.

---

## 3. Training & Alignment Engine (`src/training/`)

* **Trainer (`trainer.py`)**:
  * Decoupled AdamW parameter groups (weight decay applied strictly to 2D matrices; zero decay on 1D normalization vectors and biases).
  * Cosine Warmup Scheduler with minimum LR floor ($0.1 \times \text{LR}_{\max}$).
  * Loss Spike Detector & Numerical NaN/Inf Protection.
  * Atomic checkpoint serialization (`.tmp` write followed by atomic `replace()`) with SHA-256 fingerprinting.
* **Supervised Fine-Tuning (`finetune.py`)**:
  * Completion-Only Loss Masking (prompt tokens assigned index `-100`).
  * Experience Replay Buffer (blends base pretraining corpus tokens to prevent catastrophic forgetting).
* **Direct Preference Optimization (`dpo.py`)**:
  * Closed-form preference loss on pairwise $(x, y_{\text{chosen}}, y_{\text{rejected}})$ completions with reference policy KL penalty:
  $$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

---

## 4. Multi-Backend & Teacher Model Architecture (`src/backends/`)

* **Separation of Concerns**:
  * `scratch_from_scratch`: The pure in-house ModernTransformer trained from scratch.
  * `pretrained_local_ollama`: Optional local pretrained models running via Ollama.
  * `teacher_distillation`: Teacher models used strictly to synthesize candidate instruction datasets and provide critiques.
* **Zero Silent Substitutions**: Every generated output carries explicit origin attribution labels.

