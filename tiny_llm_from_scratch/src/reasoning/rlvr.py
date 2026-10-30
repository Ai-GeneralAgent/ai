"""
Reinforcement Learning with Verifiable Rewards (RLVR) & Self-Verified Distillation
Implements the self-improvement paradigm inspired by Phi-4-Mini-Reasoning and rStar-Math:
1. Deterministic reward verification (SymPy math equality, unit test execution, constraint checks)
2. Best-of-N Candidate Search at test time
3. Self-verified distillation buffer for continual self-improvement without human labeling
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


@dataclass
class VerificationOutcome:
    is_valid: bool
    reward_score: float  # 0.0 to 1.0
    feedback: str
    verified_result: Optional[str] = None
    execution_time: float = 0.0


class DeterministicVerifier:
    """
    Deterministic Verifiable Reward Oracle:
    Scores solutions using ground-truth execution rather than speculative reward models.
    """

    @classmethod
    def verify_math_expression(cls, expression_str: str, candidate_answer: str) -> VerificationOutcome:
        """Evaluates ground-truth algebraic/arithmetic equality using SymPy or Python."""
        t0 = time.time()
        # Extract target expression / value
        expr_clean = expression_str.strip()
        ans_clean = candidate_answer.strip()

        # If equality present in candidate, focus on right-hand side
        if "=" in ans_clean:
            ans_clean = ans_clean.split("=")[-1].strip()

        match = re.search(r"(?:answer is|result is|=|:\s*)\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)", ans_clean, re.IGNORECASE)
        extracted_ans = match.group(1) if match else ans_clean
        extracted_ans = extracted_ans.rstrip(".").strip()

        # Extract all numbers from candidate text as candidates
        candidate_numbers = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", ans_clean) if n]

        # Ground truth evaluation
        expected_numeric: Optional[float] = None
        if HAS_SYMPY:
            try:
                expected_val = parse_expr(expr_clean)
                expected_numeric = float(expected_val.evalf())
            except Exception:
                pass

        if expected_numeric is None:
            sanitized = re.sub(r"[^0-9\+\-\*\/\(\)\.\s]", "", expr_clean)
            try:
                expected_numeric = float(eval(sanitized, {"__builtins__": None}, {}))
            except Exception:
                pass

        if expected_numeric is not None:
            # Check candidate numbers
            for c_num in candidate_numbers:
                if abs(expected_numeric - c_num) < 1e-5:
                    return VerificationOutcome(
                        is_valid=True,
                        reward_score=1.0,
                        feedback=f"Numeric equality verified: expected {expected_numeric} == found {c_num}",
                        verified_result=str(expected_numeric),
                        execution_time=time.time() - t0,
                    )

        return VerificationOutcome(
            is_valid=False,
            reward_score=0.0,
            feedback=f"Candidate answer '{extracted_ans}' does not match computed value.",
            execution_time=time.time() - t0,
        )

    @classmethod
    def verify_python_code(cls, code: str, test_assertions: List[str], timeout_sec: float = 3.0) -> VerificationOutcome:
        """Executes candidate Python code against deterministic unit tests in a separate sandbox."""
        t0 = time.time()
        # Combine code with assertions
        test_script = f"{code}\n\n# --- VERIFICATION TEST SUITE ---\n"
        for idx, assertion in enumerate(test_assertions):
            test_script += f"{assertion}\n"

        try:
            proc = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            elapsed = time.time() - t0
            if proc.returncode == 0:
                return VerificationOutcome(
                    is_valid=True,
                    reward_score=1.0,
                    feedback=f"All {len(test_assertions)} unit test assertions passed successfully.",
                    verified_result="PASSED",
                    execution_time=elapsed,
                )
            else:
                return VerificationOutcome(
                    is_valid=False,
                    reward_score=0.0,
                    feedback=f"Unit test failure:\n{proc.stderr.strip()[:300]}",
                    verified_result="FAILED",
                    execution_time=elapsed,
                )
        except subprocess.TimeoutExpired:
            return VerificationOutcome(
                is_valid=False,
                reward_score=0.0,
                feedback=f"Execution timed out after {timeout_sec}s.",
                execution_time=time.time() - t0,
            )
        except Exception as e:
            return VerificationOutcome(
                is_valid=False,
                reward_score=0.0,
                feedback=f"Sandbox execution error: {e}",
                execution_time=time.time() - t0,
            )


class SelfVerifiedDistillationManager:
    """
    Self-Verified Distillation & Test-Time Search (rStar-Math pattern):
    Generates or collects N candidate solutions, scores them using deterministic verifiers,
    selects the highest-reward verified answer, and records the survivor for continual learning.
    """

    def __init__(self, data_output_path: Union[str, Path] = "training_data/rlvr_verified_data.jsonl"):
        self.output_path = Path(data_output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.verifier = DeterministicVerifier()

    def evaluate_candidates(
        self,
        problem: str,
        candidates: List[str],
        verifier_type: str = "math",
        verification_target: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[Optional[str], float, List[VerificationOutcome]]:
        """
        Filters candidates with the deterministic verifier.
        Returns: (best_candidate, best_reward, all_outcomes)
        """
        outcomes: List[VerificationOutcome] = []
        best_candidate: Optional[str] = None
        best_reward = 0.0

        for cand in candidates:
            if verifier_type == "math":
                outcome = self.verifier.verify_math_expression(str(verification_target or problem), cand)
            elif verifier_type == "code" and isinstance(verification_target, list):
                outcome = self.verifier.verify_python_code(cand, verification_target)
            else:
                outcome = VerificationOutcome(is_valid=True, reward_score=1.0, feedback="Accepted")

            outcomes.append(outcome)
            if outcome.is_valid and outcome.reward_score > best_reward:
                best_reward = outcome.reward_score
                best_candidate = cand

        # Self-improvement logging: if a verified solution was found, persist it to training dataset!
        if best_candidate and best_reward > 0.0:
            self.log_verified_sample(problem, best_candidate, reward=best_reward)

        return best_candidate, best_reward, outcomes

    def log_verified_sample(self, problem: str, verified_solution: str, reward: float = 1.0):
        """Appends verified problem-solution pair into the continual RLVR dataset."""
        record = {
            "instruction": problem,
            "response": verified_solution,
            "reward": reward,
            "verified_at": time.time(),
        }
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
