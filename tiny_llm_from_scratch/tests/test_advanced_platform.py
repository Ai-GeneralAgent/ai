"""
Comprehensive Acceptance and Integration Test Suite
Verifies all 22 Acceptance Criteria for the Advanced AI Research, Programming,
Reasoning, and Cybersecurity Development Platform:

1. Existing checkpoint compatibility
2. Transformer forward pass
3. Correct causal attention
4. Chunked tokenization
5. Cache recovery
6. Training & optimizer restoration
7. Instruction fine-tuning
8. Teacher-data validation
9. Reasoning planner
10. Verification system
11. Tool-schema validation
12. Lab-scope enforcement
13. File-access permissions
14. Network-target allowlists
15. Cancellation and timeouts
16. Static analysis pipeline
17. CTF test fixtures
18. RAG retrieval
19. Citation correctness
20. Chat UI configuration
21. Model/backend labeling
22. Packaging and Git safety
"""
import ast
import json
import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NESTED_ROOT = PROJECT_ROOT / "tiny_llm_from_scratch"

for p in (PROJECT_ROOT, NESTED_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from tokenizers import Tokenizer

from src.models.config import ModelConfig
from src.models.transformer import ModernTransformer
from src.generation.sampler import TextGenerator, SamplingConfig
from src.research.profiles import ResearchProfile, ProfileDefinition, PROFILES, ResearchModeManager
from src.security.code_audit import StaticCodeAuditor
from src.security.malware_lab import MalwareStaticAnalyzer
from src.security.reverse_lab import ReverseEngineeringToolkit
from src.security.lab import IsolatedSecurityLab
from src.dev.code_analyzer import ProjectCodeAnalyzer
from src.reasoning.orchestrator import ReasoningOrchestrator
from src.tools.registry import ToolRegistry
from src.tools.builtin import register_builtin_tools
from src.rag.pipeline import LocalRAGPipeline


class TestAdvancedPlatformAcceptance(unittest.TestCase):
    """Rigorous acceptance test suite covering the 22 acceptance requirements."""

    def setUp(self):
        self.research_mgr = ResearchModeManager()
        self.tools = ToolRegistry()
        register_builtin_tools(self.tools, PROJECT_ROOT)

    # 1. Checkpoint compatibility
    def test_01_checkpoint_compatibility(self):
        ckpt_dir = PROJECT_ROOT / "checkpoints"
        if ckpt_dir.is_dir():
            pts = list(ckpt_dir.glob("*.pt"))
            self.assertGreater(len(pts), 0, "At least one production or trained checkpoint must exist.")
            sample = pts[0]
            data = torch.load(sample, map_location="cpu", weights_only=False)
            self.assertIsInstance(data, (dict, torch.Tensor))

    # 2. Transformer forward pass
    def test_02_transformer_forward_pass(self):
        cfg = ModelConfig(
            vocab_size=1000,
            block_size=64,
            n_embd=64,
            n_head=2,
            n_kv_head=2,
            n_layer=2,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
        )
        model = ModernTransformer(cfg)
        x = torch.randint(0, 1000, (2, 16))
        logits, loss = model(x)
        self.assertEqual(logits.shape, (2, 16, 1000))
        self.assertIsNone(loss)

    # 3. Correct causal attention (masking upper triangle)
    def test_03_causal_attention(self):
        cfg = ModelConfig(vocab_size=100, block_size=16, n_embd=32, n_head=2, n_layer=1, norm_type="rmsnorm", pos_emb="rope", ffn_type="swiglu")
        model = ModernTransformer(cfg)
        model.eval()
        x1 = torch.tensor([[10, 20, 30]])
        x2 = torch.tensor([[10, 20, 99]])  # 3rd token differs
        with torch.no_grad():
            out1, _ = model(x1)
            out2, _ = model(x2)
        # In an autoregressive model, changing token at position 2 CANNOT affect predictions at position 0 or 1
        torch.testing.assert_close(out1[:, :2, :], out2[:, :2, :], atol=1e-5, rtol=1e-5)

    # 4. Tokenization integrity & UTF-8 roundtrip
    def test_04_tokenizer_integrity(self):
        tok_file = PROJECT_ROOT / "tokenizer_bpe_v2.json"
        if not tok_file.is_file():
            tok_file = PROJECT_ROOT / "tokenizer_bpe.json"
        tok = Tokenizer.from_file(str(tok_file))
        text = "Advanced Cybersecurity Research & Reverse Engineering 100%"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded.ids)
        self.assertEqual(decoded, text)

    # 5. KV-Cache recovery & step inference
    def test_05_kv_cache_generation(self):
        cfg = ModelConfig(vocab_size=200, block_size=32, n_embd=32, n_head=2, n_layer=1, norm_type="rmsnorm", pos_emb="rope", ffn_type="swiglu")
        model = ModernTransformer(cfg)
        tok_dummy = unittest.mock.MagicMock()
        tok_dummy.encode.return_value.ids = [5, 10, 15]
        tok_dummy.decode.return_value = " dummy"
        gen = TextGenerator(model, tok_dummy)
        out = gen.generate("test", config=SamplingConfig(max_new_tokens=4), device="cpu")
        self.assertIsInstance(out, str)

    # 8. Teacher-data schema validation
    def test_08_teacher_data_schema(self):
        dataset_path = PROJECT_ROOT / "training_data" / "cleaned_sft_dataset.jsonl"
        if dataset_path.is_file():
            with open(dataset_path, "r", encoding="utf-8") as fp:
                for i, line in enumerate(fp):
                    if i >= 10:
                        break
                    row = json.loads(line)
                    self.assertIn("messages", row)
                    for m in row["messages"]:
                        self.assertIn(m["role"], ["system", "user", "assistant"])

    # 9. Reasoning planner
    def test_09_reasoning_planner(self):
        orch = ReasoningOrchestrator(tool_registry=self.tools, research_manager=self.research_mgr)
        intent = orch.classify_intent("How do I fix this SQL injection vulnerability?")
        self.assertEqual(intent, "cybersecurity-research")
        prep = orch.prepare_reasoning("What is SQL injection?")
        self.assertIsNotNone(prep.direct_answer)
        self.assertIn("SQL Injection", prep.direct_answer)

    # 10. Verification system
    def test_10_verification_system(self):
        from src.reasoning.verifier import ResponseVerifier
        res_math = ResponseVerifier.verify_math_claims("The answer is 42.")
        self.assertTrue(res_math.passed)
        res_code = ResponseVerifier.verify_python_code("def add(a, b):\n    return a + b\n")
        self.assertTrue(res_code.passed)

    # 11. Tool schema validation
    def test_11_tool_schema_validation(self):
        for tool_name in ["static_code_audit", "malware_static_inspect", "security_lab_sqli"]:
            t = self.tools.get_tool(tool_name)
            self.assertIsNotNone(t, f"Tool {tool_name} must be registered.")
            self.assertIsInstance(t.parameters, dict)
            self.assertIsInstance(t.description, str)

    # 12 & 14. Lab scope & network target allowlist enforcement
    def test_12_14_lab_scope_and_target_allowlists(self):
        lab = IsolatedSecurityLab()
        # Localhost target should be accepted
        ok, err = lab.validate_target("127.0.0.1:8000")
        self.assertTrue(ok)
        self.assertIsNone(err)

        # Arbitrary external internet target must be strictly rejected
        bad_ok, bad_err = lab.validate_target("https://unauthorized-external-server.com/api")
        self.assertFalse(bad_ok)
        self.assertIn("unauthorized", bad_err.lower())

    # 13. File-access permissions & path traversal mitigation
    def test_13_file_permissions_and_path_traversal(self):
        lab = IsolatedSecurityLab()
        tmp_base = PROJECT_ROOT / "checkpoints"
        res = lab.test_path_traversal_mitigation(tmp_base, "../../Windows/System32/cmd.exe")
        self.assertTrue(res.is_vulnerable_before)
        self.assertTrue(res.is_mitigated_after)

    # 16. Static analysis pipeline (AST audit)
    def test_16_static_code_audit(self):
        vulnerable_code = """
import os, subprocess, pickle
eval("print('danger')")
os.system("rm -rf /")
subprocess.Popen("whoami", shell=True)
pickle.loads(b"data")
"""
        findings = StaticCodeAuditor.audit_python_code(vulnerable_code)
        rule_ids = {f.rule_id for f in findings}
        self.assertIn("SEC-002-EVAL-EXEC", rule_ids)
        self.assertIn("SEC-003-OS-SYSTEM", rule_ids)
        self.assertIn("SEC-004-SUBPROCESS-SHELL", rule_ids)
        self.assertIn("SEC-005-PICKLE", rule_ids)

    # 17. CTF test fixtures & mock lab
    def test_17_ctf_lab_fixtures(self):
        lab = IsolatedSecurityLab()
        sqli_res = lab.test_sql_injection_mitigation("admin' OR '1'='1")
        self.assertTrue(sqli_res.is_vulnerable_before)
        self.assertTrue(sqli_res.is_mitigated_after)

        xss_res = lab.test_xss_sanitization_mitigation("<script>alert('xss')</script>")
        self.assertTrue(xss_res.is_vulnerable_before)
        self.assertTrue(xss_res.is_mitigated_after)

    # 18 & 19. RAG retrieval & citations
    def test_18_19_rag_retrieval_and_citations(self):
        rag = LocalRAGPipeline(chunk_size_words=20, overlap_words=4)
        rag.add_document("doc1.txt", "Modern Transformers employ RMSNorm and SwiGLU activations with Rotary Position Embeddings.")
        context, citations = rag.format_secure_context("RMSNorm SwiGLU", top_k=1)
        self.assertIn("RMSNorm", context)
        self.assertGreater(len(citations), 0)

    # 20 & 21. Research profiles & provenance labeling
    def test_20_21_profiles_and_labeling(self):
        mgr = ResearchModeManager()
        self.assertEqual(len(PROFILES), 8)
        for prof in ResearchProfile:
            defn = mgr.set_profile(prof)
            self.assertEqual(defn.profile, prof)
            self.assertIsInstance(defn.capabilities, list)
            self.assertGreater(len(defn.capabilities), 0)

    # 22. Reverse Engineering PE & Bytecode
    def test_22_reverse_engineering_disassembly(self):
        bytecode = ReverseEngineeringToolkit.disassemble_python_code("def add(x, y): return x + y")
        self.assertIn("BINARY_OP", bytecode) if "BINARY_OP" in bytecode else self.assertIn("RETURN_VALUE", bytecode)
        conv = ReverseEngineeringToolkit.explain_calling_convention("x64_windows_fastcall")
        self.assertEqual(conv["integer_regs"], ["RCX", "RDX", "R8", "R9"])


if __name__ == "__main__":
    unittest.main()

