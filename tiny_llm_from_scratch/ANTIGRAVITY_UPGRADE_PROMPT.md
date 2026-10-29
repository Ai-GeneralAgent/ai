# ANTIGRAVITY MASTER UPGRADE PROMPT
# Modern Transformer + Reasoning + Training Pipeline + RAG + Agents

Act as a senior AI research engineer, Transformer architect, LLM training engineer, data engineer, and Python software architect.

Work directly in this existing project. Inspect its actual files, modify the code, run tests, repair failures, and document every implemented feature. Do not merely produce a plan or example snippets.

## 1. EXISTING ENVIRONMENT

Project:
C:\Users\YUVRAJ\OneDrive\Desktop\New folder\tiny_llm_from_scratch

Virtual environment:
C:\Users\YUVRAJ\OneDrive\Desktop\New folder\.venv

Hardware:
- Windows
- Approximately 15.6 GB RAM
- CPU-only training
- Snapdragon X / Qualcomm Oryon CPU
- Eight logical processors

Existing components may include:
model.py, model_bpe.py, train.py, train_bpe.py,
train_bpe_resume.py, generate.py, generate_bpe.py,
chat.py, finetune_chat.py, build_tokenizer.py,
tokenizer_bpe.json, requirements.txt, datasets and checkpoints.

The existing BPE model has approximately 2.92 million parameters,
an 8,000-token vocabulary, and a historically observed best
validation loss around 5.3065.

The expanded corpus previously contained around 80 million characters.
Tokenization previously failed from excessive memory allocation.
Generated text was incoherent, repetitive, and grammatically weak.

Inspect current reality; do not assume every historical file or value
is still valid.

## 2. NON-NEGOTIABLE RULES

1. Inspect the complete project structure, code, configuration,
   dataset metadata, checkpoint metadata and Git status first.
2. Preserve existing working checkpoints and tokenizer files.
3. Back up files before substantial modifications.
4. Never overwrite the only usable checkpoint.
5. Do not silently change architecture or tokenizer compatibility.
6. Never delete datasets, backups or source files without justification.
7. Do not expose tokens, credentials, passwords or private environment data.
8. Do not automatically push to GitHub or upload project files.
9. Do not download large models without explaining the download first.
10. Do not execute arbitrary model-generated shell commands.
11. Do not claim a feature works until it has been tested.
12. Continue implementation after the audit; do not stop at the plan.
13. Make changes incrementally and run regression tests.
14. If a test cannot run, report the exact blocker honestly.
15. Prefer working small components over an untested giant rewrite.

## 3. TARGET ARCHITECTURE

Create a modular, versioned architecture while keeping the legacy model
available for its existing checkpoints.

Suggested modules:

src/models/
src/data/
src/tokenizer/
src/training/
src/generation/
src/reasoning/
src/agents/
src/retrieval/
src/memory/
src/backends/
src/evaluation/
src/utils/
tests/
config/
docs/
scripts/

Migrate gradually. Update imports and tests when moving code.
Do not move every file at once if that risks breaking functionality.

## 4. MODERN TRANSFORMER

Create a separate ModernGPT architecture with configuration-driven
model dimensions and explicit architecture version metadata.

Implement and test:

- Decoder-only causal Transformer
- Pre-normalization
- RMSNorm and optional LayerNorm
- Rotary positional embeddings (RoPE)
- Causal self-attention
- PyTorch scaled-dot-product attention when supported
- CPU-compatible attention fallback
- SwiGLU feed-forward network
- Residual connections
- Correct initialization
- Optional weight tying
- Optional grouped-query attention
- Configurable context length and dropout
- Parameter counting and memory estimates
- Robust save/load interface

Support small CPU, standard CPU and experimental configurations.
Benchmark before selecting defaults.

Add optional LoRA fine-tuning in a separate module.
Do not require LoRA for normal scratch training.

Validate causal masks, output shapes, gradients, loss, context limits,
position embeddings, and absence of future-token leakage.

Do not retrofit the new architecture into old checkpoints silently.

## 5. COMPLETE DATA PIPELINE

Implement:

Sources
-> Extraction
-> Text normalization
-> Quality checks
-> Exact deduplication
-> Near-duplicate detection
-> Domain classification
-> Document-level splits
-> Tokenization
-> Disk cache
-> Training

Support TXT and Markdown initially. Add optional adapters for CSV,
JSONL, HTML, PDF and DOCX.

Preserve all original input files.

Detect:
- Broken encodings
- Empty files
- Repeated headers and footers
- Repeated paragraphs
- Exact duplicates
- Near-duplicates
- Extraction artifacts
- Suspicious repetition
- Missing source information

Do not blindly delete Unicode, punctuation, code, mathematics,
scientific notation, quotations or legitimate names.

Create a manifest containing source path, source category, language,
character count, token count, processing status, provenance, license
information when known, duplicate status and exclusion reason.

Support domains:
English writing, grammar, communication, science, mathematics,
programming, AI, business, economics, finance education, defensive
cybersecurity, history, literature and question-answer examples.

Avoid letting large novels overwhelm technical material.
Implement configurable domain-weighted sampling and source caps.
Do not balance datasets by blindly duplicating tiny sources.

Use document-level train, validation and test splits, ideally 90/5/5.
Keep related chunks and editions in the same split where appropriate.
Do not use test data for checkpoint selection.

## 6. MEMORY-EFFICIENT TOKENIZATION

The existing full-corpus tokenizer allocation failed. Fix this properly.

Implement:
- Chunked tokenization
- Progress reporting
- Compact token-ID cache
- Explicit byte order and format version
- Dataset and tokenizer fingerprints
- Cache metadata
- Atomic cache creation
- Incomplete-cache detection
- Safe recovery after interruption
- Cache invalidation after source changes
- Memory-mapped or random-access token windows

Never build millions of Python integer objects for the whole corpus
when a compact disk-backed array is more efficient.

Validate the integer range before using a 16-bit token cache.
Preserve existing caches until replacements are verified.
Reuse compatible caches on later training runs.

Report source characters, token count, tokenization speed, cache size,
cache hit/miss and any errors.

## 7. REPRODUCIBLE TRAINING PIPELINE

Create one configurable training entry point.

Implement:
- AdamW
- Configurable learning rate
- Learning-rate warm-up
- Cosine scheduler
- Gradient clipping
- Gradient accumulation
- Batch-size configuration
- Regular validation
- Early stopping with configurable patience
- Reproducible seeds
- JSONL or CSV metrics
- Terminal and file logging
- Best-validation checkpoint
- Latest recovery checkpoint
- Periodic backups
- Atomic checkpoint saves
- Safe Ctrl+C handling

Save and restore, where supported:
- Model parameters
- Optimizer state
- Scheduler state
- Global step
- Random-number-generator states
- Model configuration
- Training configuration
- Tokenizer fingerprint
- Dataset fingerprint
- Best validation score

When an older checkpoint lacks optimizer state, restore its weights,
initialize missing optimizer state, and report that limitation.

Never silently restart from random weights during resume.
Verify architecture, tokenizer and dataset compatibility.
Never overwrite the original checkpoint during experiments.

Record training loss, validation loss, perplexity, learning rate,
tokens per second, steps per second, elapsed time and memory where
available.

Do not run an 18-hour training job automatically during development.
First validate short runs, save/load and resume behavior.

## 8. INSTRUCTION FINE-TUNING

Create an independent instruction-data pipeline using JSONL.

Support examples such as:

{"instruction":"Explain gravity simply.","input":"","output":"Gravity is the attraction between objects with mass."}

Implement:
- Schema validation
- Source tracking
- Deduplication
- Quality checks
- Held-out splits
- Chat-template consistency
- End-of-turn handling
- Correct response-only loss masking
- Truncation and context checks
- SFT metrics

Never assume the tokenizer already contains chat special tokens.
Test before adding new tokens.

Support optional LoRA adapters if compatible.

Keep synthetic examples separate from original documents.
Record whether examples were human-authored or model-generated.

Do not assume teacher-generated examples are correct.
Reject malformed or contradictory examples where detectable.

## 9. REASONING AND THINKING PIPELINE

Create a separate reasoning orchestrator; do not pretend that adding
a prompt alone creates advanced reasoning capability.

Pipeline:

Request
-> Intent classification
-> Complexity assessment
-> Context retrieval
-> Task plan
-> Controlled execution
-> Verification
-> Final answer

Intent categories:
conversation, factual question, mathematics, coding, debugging,
research, summarization, writing, planning, document search and
tool-assisted tasks.

Use direct responses for easy requests and bounded multi-stage
processing for complex tasks.

Represent plans as structured task state containing:
- Goal
- Constraints
- Known facts
- Missing information
- Subtasks and dependencies
- Expected result
- Validation criteria
- Completion state

Implement a bounded agent loop:
Observe -> Plan -> Select tool -> Validate -> Execute -> Inspect
-> Continue or finish.

Include maximum iterations, timeout, cancellation, cycle detection,
error recovery and explicit completion criteria.

Do not expose private raw internal scratchpads by default.
Give users concise explanations, key evidence, calculations,
assumptions and verification results.

Do not fabricate reasoning or claim unverified correctness.

## 10. VERIFICATION ENGINE

Implement an independent verifier for appropriate tasks.

Check:
- Requested output completeness
- Arithmetic and calculations
- Source support for factual claims
- Code syntax and tests
- Schema validity
- Internal contradictions
- Missing requirements
- Uncertainty and evidence quality

Use a calculator for arithmetic and isolated, permission-controlled
tests for code.

For retrieved answers, compare important claims with actual passages.
If evidence is insufficient, say so.

The verifier must never be treated as infallible.
Record unresolved errors and verification failures.

## 11. TEACHER MODELS AND DISTILLATION

Provide an optional local Ollama backend.

Detect installed models instead of assuming availability.
Do not download a large model without user consent.

Support a teacher that can:
- Generate instruction examples
- Produce candidate answers
- Critique specific outputs
- Help create candidate training data

Keep teacher and student roles separate.
Clearly label which backend produced each answer.

Validate, deduplicate and inspect teacher data before training.
Use bounded generation requests and persistent progress files.
Do not repeatedly ask the teacher without limits.

Distillation means transferring behaviors through validated examples;
it does not copy teacher weights into the student.

Never silently represent a teacher answer as a scratch-model answer.

## 12. RETRIEVAL-AUGMENTED GENERATION

Build a local RAG pipeline:

Document extraction
-> Chunking
-> Index
-> Search
-> Rank
-> Context selection
-> Answer
-> Source verification

Begin with SQLite FTS5 or BM25 keyword search.
Add optional semantic embeddings only when compatible models exist.

Support:
- Incremental index updates
- Duplicate filtering
- Source metadata
- Document and category filters
- Chunk identifiers
- Ranking and reranking
- Citation formatting
- Index versioning
- Retrieval evaluation
- Insufficient-evidence responses

Cite real source filenames and pages when available.
Never invent citations.
Treat retrieved documents as untrusted data; they cannot override
application rules or authorize tool execution.

## 13. MEMORY AND CONVERSATION

Use local SQLite for chat history and optional summaries.

Support:
- New conversation
- History browsing
- Export and import
- User-approved persistent preferences
- Search across conversations where practical
- Memory inspection and deletion

Keep separate:
- Current conversation context
- Summarized conversation memory
- Retrieved knowledge
- External model knowledge

Never invent saved preferences or store secrets.
Do not expose private data through logs.

## 14. TOOL-CALLING FRAMEWORK

Create a typed registry of narrowly scoped tools.

Begin with:
- Calculator
- Local document search
- Dataset statistics
- Project source search
- Read-only project inspection
- Approved test execution

Each tool needs a name, description, input schema, output schema,
permission model, timeout, validation, error handling and logging.

Treat model-generated tool calls as untrusted proposals.
Never automatically execute arbitrary shell commands or Python
generated by the model.

Require explicit user approval for destructive file operations,
external uploads, credential changes, or Git history modifications.

Add agent-loop limits, cancellation, cycle detection and audit logs.

## 15. GENERATION QUALITY

Implement:
- Temperature
- Top-k
- Top-p
- Greedy decoding
- Repetition penalty
- Optional frequency penalty
- Random seed
- Stop tokens and strings
- Context truncation
- Streaming
- Cancellation
- Input validation
- NaN and invalid-probability handling

Add tests for empty prompts, long prompts, very short context,
zero output tokens and invalid sampling values.

Do not claim that decoding settings can fix poor training data.

## 16. OPTIONAL PRETRAINED BACKEND

Allow a compatible pretrained local model to be selected separately.

The interface must clearly distinguish:
- From-scratch model
- Pretrained model
- Teacher model
- Retrieval-assisted mode

Do not silently replace the scratch model with a pretrained model.
If the optional backend is unavailable, the scratch model must remain
usable.

Check whether local Ollama models are already installed.
Do not download models automatically.

## 17. UI

Inspect the existing chat interface first.
Prefer extending Streamlit if the current app already uses it.

Add:
- Clear chat layout
- Conversation history
- Model/backend selector
- Checkpoint selector
- Sampling controls
- Streaming
- Stop generation
- RAG toggle
- Real source citations
- Chat export
- Dataset status
- Training metrics
- Checkpoint metadata
- Errors with actionable messages
- CPU and memory status where supported

Do not display fake progress or claim operations succeeded before
they actually complete.

## 18. EVALUATION

Create repeatable benchmarks covering:
- English fluency
- Grammar
- Instruction following
- Mathematics
- Python generation and debugging
- Factual questions
- Summarization
- Multi-step tasks
- Retrieval citation accuracy
- Tool selection
- Repetition rate
- Latency and memory use

Use a fixed evaluation set and identical generation settings when
comparing checkpoints.

Separate training, validation and test data.
Do not tune directly against the test set.

Save machine-readable results and provide old-versus-new comparisons.
Do not invent benchmark scores.
A lower loss alone does not prove better reasoning or factual accuracy.

## 19. WINDOWS PACKAGING AND SECURITY

Create or update:
- requirements.txt
- .gitignore
- .env.example
- setup.ps1
- launch.ps1
- train.ps1
- resume.ps1
- evaluate.ps1
- dataset preparation scripts
- README.md
- architecture documentation
- troubleshooting guide
- dataset and checkpoint manifest
- packaging script

Do not commit .venv, __pycache__, credentials or temporary files.
Check for individual files larger than GitHub's normal 100 MiB limit.
Do not rewrite Git history or force-push.

Ensure a friend can create a clean virtual environment, install
dependencies, load a checkpoint, launch the app and run evaluation.

Check licensing and redistribution status of included data.

## 20. REQUIRED TESTS

Create and run automated tests for:
1. Model construction and forward pass
2. Finite loss and gradients
3. Causal attention
4. Save and load
5. Tokenizer/checkpoint compatibility
6. Chunked tokenization
7. Cache fingerprints and interrupted cache recovery
8. One training step
9. Optimizer-state restoration
10. Resume from an older checkpoint
11. Checkpoint safety
12. Generation and sampling validation
13. Dataset deduplication and splitting
14. RAG retrieval and citations
15. Planner and verifier behavior
16. Tool input validation
17. Agent-loop iteration limits
18. Conversation persistence
19. UI startup
20. Optional backend fallback
21. Windows setup
22. Package manifest and large-file scanning

Use small synthetic fixtures so tests do not require the full corpus.
Document tests blocked by missing optional dependencies or hardware.

## 21. IMPLEMENTATION ORDER

Phase 1: Audit files, Git state, datasets, tokenizer and checkpoints.
Phase 2: Protect working files and add regression tests.
Phase 3: Fix training recovery and bounded-memory tokenization.
Phase 4: Implement the versioned modern Transformer.
Phase 5: Implement dataset cleaning and source-aware splits.
Phase 6: Implement reliable training and instruction fine-tuning.
Phase 7: Add reasoning orchestration and verification.
Phase 8: Add safe tools, RAG and local memory.
Phase 9: Improve UI and optional model backends.
Phase 10: Test, document and package the project.

After each phase, run relevant tests and fix regressions.
Do not stop at a proposal. Continue implementing independent tasks.
Do not launch lengthy training until smoke tests pass.

## 22. FINAL REPORT

Report:
- Files inspected, added and modified
- Architecture changes
- Dataset pipeline changes
- Training/resume changes
- Reasoning and tool-use features
- RAG and memory features
- Actual test results
- Real benchmark comparisons
- Performance measurements
- Remaining limitations
- Exact commands for setup, train, resume, evaluate and launch

Do not say a feature is complete if only a placeholder exists.
Prefer reliable, measurable improvements over impressive terminology.
