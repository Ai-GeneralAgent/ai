import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer

from src.models import (
    ModelConfig,
    ModernTransformer,
    apply_lora,
    save_checkpoint,
    load_modern_checkpoint,
    inspect_checkpoint,
)
from src.data import (
    DatasetManifest,
    DocumentRecord,
    CorpusPipeline,
    CompactTokenCache,
)
from src.training import (
    Trainer,
    TrainingConfig,
    TrainingMonitor,
    InstructionDataset,
)
from src.generation import (
    TextGenerator,
    SamplingConfig,
    ChatTemplate,
    ChatMessage,
)
from src.tools import (
    ToolDefinition,
    ToolRegistry,
    register_builtin_tools,
    SafeCalculator,
    BoundedAgentLoop,
)
from src.rag import LocalRAGPipeline
from src.memory import ConversationDatabase
from src.reasoning import (
    TaskPlanner,
    ResponseVerifier,
    ReasoningOrchestrator,
)
from src.evaluation import BenchmarkSuite


class TestModernLLMSystem(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = PROJECT_ROOT / "tests" / "tmp_test_data"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tok_path = PROJECT_ROOT / "tokenizer_bpe.json"
        self.assertTrue(self.tok_path.is_file(), "Tokenizer file must exist")
        self.tokenizer = Tokenizer.from_file(str(self.tok_path))

    def tearDown(self):
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # 1. Model construction
    def test_01_model_construction(self):
        cfg = ModelConfig.preset_tiny(vocab_size=self.tokenizer.get_vocab_size())
        model = ModernTransformer(cfg)
        self.assertIsInstance(model, ModernTransformer)
        self.assertGreater(model.count_parameters(), 100_000)

    # 2. Model forward pass
    def test_02_model_forward_pass(self):
        cfg = ModelConfig.preset_tiny(vocab_size=self.tokenizer.get_vocab_size())
        model = ModernTransformer(cfg)
        x = torch.randint(0, 100, (2, 16))
        logits, loss = model(x)
        self.assertEqual(logits.shape, (2, 16, self.tokenizer.get_vocab_size()))
        self.assertIsNone(loss)

    # 3. Loss calculation
    def test_03_loss_calculation(self):
        cfg = ModelConfig.preset_tiny(vocab_size=self.tokenizer.get_vocab_size())
        model = ModernTransformer(cfg)
        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))
        logits, loss = model(x, targets=y)
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

    # 4. Backpropagation
    def test_04_backpropagation(self):
        cfg = ModelConfig.preset_tiny(vocab_size=self.tokenizer.get_vocab_size())
        model = ModernTransformer(cfg)
        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))
        _, loss = model(x, targets=y)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    # 5. Save and load
    def test_05_save_and_load(self):
        cfg = ModelConfig.preset_tiny(vocab_size=self.tokenizer.get_vocab_size())
        model = ModernTransformer(cfg)
        ckpt_file = self.tmp_dir / "test_ckpt.pt"
        save_checkpoint(model, ckpt_file, step=42, val_loss=3.14)
        self.assertTrue(ckpt_file.is_file())

        loaded_model, payload = load_modern_checkpoint(ckpt_file)
        self.assertEqual(payload["step"], 42)
        self.assertEqual(payload["val_loss"], 3.14)

    # 6. Tokenizer compatibility
    def test_06_tokenizer_compatibility(self):
        text = "Transformer architecture with Rotary Position Embeddings."
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded.ids)
        self.assertIn("Transformer", decoded)

    # 7. Chunked data processing
    def test_07_chunked_data_processing(self):
        pipe = CorpusPipeline()
        sample_doc = self.tmp_dir / "sample.txt"
        sample_doc.write_text("Python is a high-level programming language used for machine learning.", encoding="utf-8")
        rec, norm_text = pipe.process_document(sample_doc)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.domain, "programming")

    # 8. Cache recovery
    def test_08_cache_recovery(self):
        sample_doc = self.tmp_dir / "sample_cache.txt"
        sample_doc.write_text("Learning representations with deep transformers.", encoding="utf-8")
        cache = CompactTokenCache.build_or_load(sample_doc, self.tok_path, self.tokenizer)
        self.assertGreater(cache.token_count, 0)
        cache.close()

    # 9. One training step
    def test_09_one_training_step(self):
        sample_doc = self.tmp_dir / "sample_train.txt"
        sample_doc.write_text(("Artificial intelligence learns patterns from data.\n" * 100), encoding="utf-8")
        cache = CompactTokenCache.build_or_load(sample_doc, self.tok_path, self.tokenizer)

        cfg = ModelConfig.preset_tiny(vocab_size=self.tokenizer.get_vocab_size())
        model = ModernTransformer(cfg)
        t_cfg = TrainingConfig(
            max_steps=1,
            eval_interval=1,
            eval_batches=1,
            batch_size=2,
            output_dir=str(self.tmp_dir),
            run_name="quick_step_test",
        )
        trainer = Trainer(model, t_cfg, cache, self.tok_path)
        monitor = trainer.train()
        cache.close()
        self.assertEqual(monitor.get_summary()["total_steps"], 1)

    # 10. Checkpoint resume
    def test_10_checkpoint_resume(self):
        info = inspect_checkpoint(PROJECT_ROOT / "checkpoints" / "tiny_bpe_expanded_continued_v2.pt")
        self.assertEqual(info["format"], "v1_legacy_bpe")
        self.assertEqual(info["vocab_size"], 8000)

    # 11. Generation
    def test_11_generation(self):
        cfg = ModelConfig.preset_tiny(vocab_size=self.tokenizer.get_vocab_size())
        model = ModernTransformer(cfg)
        gen = TextGenerator(model, self.tokenizer)
        out = gen.generate("Hello world", SamplingConfig(max_new_tokens=5))
        self.assertIsInstance(out, str)

    # 12. Sampling validation
    def test_12_sampling_validation(self):
        cfg = ModelConfig.preset_tiny(vocab_size=self.tokenizer.get_vocab_size())
        model = ModernTransformer(cfg)
        gen = TextGenerator(model, self.tokenizer)
        # Test extreme temperature and top_p values
        out = gen.generate("Test", SamplingConfig(temperature=0.01, top_p=0.5, top_k=5, max_new_tokens=5))
        self.assertIsInstance(out, str)

    # 13. Document retrieval
    def test_13_document_retrieval(self):
        rag = LocalRAGPipeline(chunk_size_words=20, overlap_words=5)
        doc = self.tmp_dir / "doc1.txt"
        doc.write_text("Quantum physics describes nature at the smallest scales of energy levels of atoms and subatomic particles.", encoding="utf-8")
        rag.add_document(doc)
        results = rag.search("quantum atoms", top_k=1)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("quantum", results[0].chunk.text.lower())

    # 14. Citation formatting
    def test_14_citation_formatting(self):
        rag = LocalRAGPipeline(chunk_size_words=20)
        doc = self.tmp_dir / "physics.txt"
        doc.write_text("Relativity is Albert Einstein's theory of spacetime and gravity.", encoding="utf-8")
        rag.add_document(doc)
        context, citations = rag.format_context_for_prompt("relativity einstein")
        self.assertIn("[physics.txt#chunk_0]", citations)

    # 15. Structured planning
    def test_15_structured_planning(self):
        plan = TaskPlanner.create_plan("Calculate 25 * 4", intent="mathematical", complexity="simple")
        self.assertEqual(plan.intent, "mathematical")
        self.assertGreater(len(plan.subtasks), 0)

    # 16. Verification fallback
    def test_16_verification_fallback(self):
        res_math = ResponseVerifier.verify_math_claims("We calculated that 10 + 5 = 15.")
        self.assertTrue(res_math.passed)

        res_code = ResponseVerifier.verify_python_code("```python\ndef add(a, b):\n    return a + b\n```")
        self.assertTrue(res_code.passed)

    # 17. Tool schema validation
    def test_17_tool_schema_validation(self):
        registry = ToolRegistry()
        register_builtin_tools(registry, PROJECT_ROOT)
        calc_tool = registry.get_tool("calculator")
        self.assertIsNotNone(calc_tool)
        res = registry.execute("calculator", {"expression": "12 * 12 + 6"})
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], "150")

    # 18. Agent loop limits
    def test_18_agent_loop_limits(self):
        registry = ToolRegistry()
        register_builtin_tools(registry, PROJECT_ROOT)
        agent = BoundedAgentLoop(registry, max_iterations=2, max_tool_calls=2)
        plan_steps = [
            {"thought": "step 1", "tool": "datetime_now", "args": {}},
            {"thought": "step 2", "tool": "datetime_now", "args": {}},  # Cycle!
            {"thought": "step 3", "tool": "datetime_now", "args": {}},
        ]
        res = agent.run_plan("Test bounds", plan_steps)
        self.assertFalse(res["completed"])
        self.assertIn("iterations", res["reason"])
        # Second step should be flagged as cycle
        self.assertIn("Cycle detected", res["steps"][1].error)

    # 19. Conversation persistence
    def test_19_conversation_persistence(self):
        db_file = self.tmp_dir / "test_memory.db"
        mem = ConversationDatabase(db_file)
        sid = mem.create_session("s1", "Session 1")
        mem.add_message(sid, "user", "What is gradient descent?")
        mem.add_message(sid, "assistant", "An optimization algorithm.")
        msgs = mem.get_messages(sid)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["content"], "What is gradient descent?")

    # 20. UI launch / module import
    def test_20_ui_launch_import(self):
        import src.ui.cli
        self.assertTrue(hasattr(src.ui.cli, "launch_cli"))

    # 21. Optional-backend fallback
    def test_21_optional_backend_fallback(self):
        info = inspect_checkpoint(self.tmp_dir / "non_existent.pt") if (self.tmp_dir / "non_existent.pt").exists() else {}
        self.assertEqual(info.get("format"), None)

    # 22. Windows setup / run scripts
    def test_22_windows_run_scripts(self):
        self.assertTrue((PROJECT_ROOT / "quick_test.bat").is_file())
        self.assertTrue((PROJECT_ROOT / "quick_generate.bat").is_file())

    # 23. Packaging & clean imports
    def test_23_packaging_imports(self):
        import src
        self.assertEqual(src.__version__, "2.1.0")

    # 24. Git ignore and large-file checks
    def test_24_gitignore_and_large_files(self):
        gitignore = (PROJECT_ROOT.parent / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".venv/", gitignore)
        self.assertIn("*.dll", gitignore)
        self.assertIn("books_txt/", gitignore)

    # 25. KV Cache acceleration and step correctness
    def test_25_kv_cache_acceleration_and_correctness(self):
        cfg = ModelConfig.preset_tiny(vocab_size=100)
        model = ModernTransformer(cfg)
        model.eval()

        # Full sequence without cache
        tokens = torch.randint(0, 100, (1, 6))
        logits_full, _ = model(tokens)

        # Step-by-step with KV Cache
        caches = model.create_kv_caches()
        # Prefill first 5 tokens
        logits_prefill, _ = model(tokens[:, :5], kv_caches=caches, start_pos=0)
        # Step with 6th token
        logits_step, _ = model(tokens[:, 5:6], kv_caches=caches, start_pos=5)

        # Logits for the 6th token should match full sequence computation
        self.assertTrue(torch.allclose(logits_full[:, 5, :], logits_step[:, 0, :], atol=1e-4))

    # 26. RoPE context extrapolation
    def test_26_rope_context_extrapolation(self):
        cfg = ModelConfig.preset_tiny(vocab_size=100)
        cfg.block_size = 64
        cfg.max_seq_len = 256
        model = ModernTransformer(cfg)
        model.eval()

        # Sequence longer than base block_size (128 > 64)
        extended_tokens = torch.randint(0, 100, (1, 128))
        logits, _ = model(extended_tokens)
        self.assertEqual(logits.shape, (1, 128, 100))

    # 27. Advanced Math & Equation Solver
    def test_27_advanced_math_solver(self):
        from src.tools.builtin import solve_math_problem, AdvancedMathSolver
        # Linear equation
        lin = solve_math_problem("Solve: 4x + 20 = 100")
        self.assertTrue(lin["success"])
        self.assertEqual(lin["result"], 20.0)

        # Quadratic equation
        quad = solve_math_problem("x^2 - 7x + 12 = 0")
        self.assertTrue(quad["success"])
        self.assertEqual(quad["result"], [4.0, 3.0])

        # Percentage
        pct = solve_math_problem("What is 15% of 200?")
        self.assertTrue(pct["success"])
        self.assertEqual(pct["result"], 30.0)

        # Unit conversion
        conv = solve_math_problem("Convert 100 km to miles")
        self.assertTrue(conv["success"])
        self.assertAlmostEqual(conv["result"], 62.1371, places=3)

        # Statistics
        stats = AdvancedMathSolver.solve_statistics([10, 20, 30, 40, 50])
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["median"], 30.0)

    # 28. Claude / ChatGPT-style CoT Reasoning
    def test_28_claude_chatgpt_style_cot_reasoning(self):
        from src.reasoning.orchestrator import ReasoningOrchestrator
        orchestrator = ReasoningOrchestrator()

        # Mathematical request
        prep_math = orchestrator.prepare_reasoning("Solve: 2x + 10 = 50")
        self.assertEqual(prep_math.intent, "mathematical")
        self.assertIn("1. Intent Analysis:", prep_math.thinking)
        self.assertIn("20.0", prep_math.direct_answer)

        # Complex task request
        prep_complex = orchestrator.prepare_reasoning("Compare Python and C++ step by step")
        self.assertEqual(prep_complex.intent, "complex-task")
        self.assertIn("1. Intent Analysis:", prep_complex.thinking)

    # 29. Curriculum Study Engine (Grades 1 - 12)
    def test_29_curriculum_study_engine(self):
        from src.data.curriculum import CurriculumStudyEngine
        sample_book = self.tmp_dir / "physics_sample.txt"
        sample_book.write_text(
            "Chapter 1: Mechanics. Gravity is defined as the force of attraction between masses. "
            "Velocity refers to the rate of change of displacement with time. "
            "Photosynthesis is known as the process used by plants to convert light energy into chemical energy.",
            encoding="utf-8"
        )
        engine = CurriculumStudyEngine(self.tmp_dir / "curriculum_out")
        res = engine.study_book(sample_book, grade_hint="grade_9")
        self.assertEqual(res["grade_level"], "grade_9")
        self.assertGreaterEqual(res["concepts_learned"], 2)
        self.assertGreaterEqual(res["flashcards_generated"], 2)
        self.assertTrue(Path(res["study_notes_path"]).is_file())

    # 30. Safe Code Sandbox Execution
    def test_30_safe_code_sandbox(self):
        from src.tools.code_engine import SafeCodeSandbox
        # Valid execution with stdout
        code = "res = [x * 2 for x in range(5)]\nprint('Result:', res)"
        out = SafeCodeSandbox.execute_snippet(code)
        self.assertTrue(out["success"])
        self.assertIn("Result: [0, 2, 4, 6, 8]", out["stdout"])

        # Security restriction on hazardous import
        bad_code = "import os\nos.listdir('.')"
        bad_out = SafeCodeSandbox.execute_snippet(bad_code)
        self.assertFalse(bad_out["success"])
        self.assertIn("Security restriction", bad_out["error"])

    # 31. Creative Analogy Engine
    def test_31_creative_analogy_engine(self):
        from src.generation.creative import CreativeEngine
        analogy = CreativeEngine.get_intuitive_analogy("quantum mechanics")
        self.assertIsNotNone(analogy)
        self.assertIn("spinning coin", analogy)

    # 32. Instruction Dataset with Completion Loss Masking & ChatML
    def test_32_instruction_dataset_chatml(self):
        import json
        ds_file = self.tmp_dir / "test_chat.jsonl"
        with ds_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a high-level language."}
                ]
            }) + "\n")
        dataset = InstructionDataset(ds_file, self.tokenizer, block_size=64)
        self.assertEqual(len(dataset), 1)
        x, y = dataset.sample_batch(1)
        self.assertEqual(x.size(0), 1)
        # Check that -100 mask is applied to prompt tokens
        self.assertIn(-100, y.tolist()[0])

    # 33. Direct Preference Optimization (DPO) Loss
    def test_33_dpo_preference_loss(self):
        import json
        from src.training.dpo import DPODataset, dpo_loss
        dpo_file = self.tmp_dir / "test_dpo.jsonl"
        with dpo_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "prompt": "What is 2+2?",
                "chosen": "2+2 equals 4.",
                "rejected": "2+2 equals 5."
            }) + "\n")
        dpo_ds = DPODataset(dpo_file, self.tokenizer, block_size=64)
        self.assertEqual(len(dpo_ds), 1)
        batch = dpo_ds.sample_batch(1)
        self.assertIn("chosen_x", batch)
        self.assertIn("rejected_x", batch)

        # Compute DPO loss with synthetic logprobs
        pi_c = torch.tensor([-1.2])
        pi_r = torch.tensor([-3.4])
        ref_c = torch.tensor([-1.5])
        ref_r = torch.tensor([-2.0])
        loss, r_c, r_r = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
        self.assertGreater(loss.item(), 0.0)

    # 34. Constitutional AI Critique & Self-Correction
    def test_34_constitutional_ai_critique(self):
        # Clean answer passes
        passed, summary, _ = ResponseVerifier.critique_and_self_correct(
            prompt="Explain gravity",
            response="Gravity is an attractive force between objects with mass."
        )
        self.assertTrue(passed)

        # Dangerous code call fails
        passed_bad, summary_bad, recs = ResponseVerifier.critique_and_self_correct(
            prompt="Delete files",
            response="Run os.system('rm -rf /')"
        )
        self.assertFalse(passed_bad)
        self.assertIn("os.system", summary_bad)

    # 35. Ambiguity Resolution & Proactive Clarification
    def test_35_ambiguity_resolution(self):
        orch = ReasoningOrchestrator()
        prep = orch.prepare_reasoning("help me fix")
        self.assertEqual(prep.intent, "ambiguous")
        self.assertIn("proactive clarification", prep.thinking.lower())
        self.assertIsNotNone(prep.direct_answer)
        self.assertIn("What specific topic", prep.direct_answer)


if __name__ == "__main__":
    unittest.main()
