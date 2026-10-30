import ast
import re
from typing import Any, Dict, List, Optional, Tuple


class VerificationResult:
    def __init__(self, passed: bool, check_name: str, details: str, warnings: Optional[List[str]] = None):
        self.passed = passed
        self.check_name = check_name
        self.details = details
        self.warnings = warnings or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "check_name": self.check_name,
            "details": self.details,
            "warnings": self.warnings,
        }


class ResponseVerifier:
    """Independent verification and critique stage."""

    @classmethod
    def verify_python_code(cls, response_text: str) -> VerificationResult:
        """Extracts python code blocks and verifies AST syntax."""
        code_blocks = re.findall(r"```(?:python)?(.*?)```", response_text, re.DOTALL)
        if not code_blocks:
            return VerificationResult(True, "Code Syntax", "No code blocks found.")

        errors = []
        for idx, block in enumerate(code_blocks, 1):
            try:
                ast.parse(block.strip())
            except SyntaxError as e:
                errors.append(f"Block #{idx} syntax error on line {e.lineno}: {e.msg}")

        if errors:
            return VerificationResult(False, "Code Syntax", "; ".join(errors))
        return VerificationResult(True, "Code Syntax", f"All {len(code_blocks)} code block(s) parsed successfully.")

    @classmethod
    def verify_math_claims(cls, text: str) -> VerificationResult:
        """Finds equations like 'X + Y = Z' and verifies arithmetic equality."""
        equations = re.findall(r"(\d+\s*[\+\-\*\/]\s*\d+)\s*=\s*(\-?\d+(?:\.\d+)?)", text)
        if not equations:
            return VerificationResult(True, "Math Verification", "No simple arithmetic equations detected.")

        from ..tools.builtin import SafeCalculator
        failures = []
        for expr, expected_str in equations:
            try:
                calc_val = SafeCalculator.evaluate(expr)
                exp_val = float(expected_str)
                if abs(calc_val - exp_val) > 1e-4:
                    failures.append(f"Expression '{expr}' evaluated to {calc_val}, but answer claimed {expected_str}")
            except Exception as e:
                failures.append(f"Could not verify '{expr}': {str(e)}")

        if failures:
            return VerificationResult(False, "Math Verification", "; ".join(failures))
        return VerificationResult(True, "Math Verification", f"All {len(equations)} detected equations are mathematically correct.")

    @classmethod
    def verify_anti_sycophancy(cls, prompt: str, response: str) -> VerificationResult:
        """Audits response against sycophantic flattery and false premise acceptance."""
        from .truth_engine import TruthAndEpistemicEngine
        audit = TruthAndEpistemicEngine.audit_response(prompt, response)
        if audit["status"] == "SYCOPHANTIC_FLATTERY":
            return VerificationResult(False, "Anti-Sycophancy", audit["critique"])
        return VerificationResult(True, "Anti-Sycophancy", "Response maintained factual truth without sycophantic flattery.")

    @classmethod
    def verify_epistemic_abstention(cls, prompt: str, response: str) -> VerificationResult:
        """Audits response for uncalibrated hallucinations on unknowable questions."""
        from .truth_engine import TruthAndEpistemicEngine
        audit = TruthAndEpistemicEngine.audit_response(prompt, response)
        if audit["status"] == "HALLUCINATION_WRONG":
            return VerificationResult(False, "Epistemic Abstention", audit["critique"])
        return VerificationResult(True, "Epistemic Abstention", "Response observed epistemic boundaries.")

    @classmethod
    def verify_citations(cls, text: str, citations: List[str]) -> VerificationResult:
        """Verifies that cited sources exist in retrieved evidence."""
        if not citations:
            return VerificationResult(True, "Citations", "No citations required.")

        used_citations = [c for c in citations if c in text]
        if not used_citations and len(citations) > 0:
            return VerificationResult(
                False,
                "Citations",
                "Answer failed to reference retrieved citations in output.",
                warnings=["Retrieved sources were available but not cited."],
            )
        return VerificationResult(True, "Citations", f"Cited {len(used_citations)} source chunk(s).")

    @classmethod
    def critique_and_self_correct(cls, prompt: str, response: str) -> Tuple[bool, str, List[str]]:
        """
        Constitutional AI Self-Correction & Critique Loop (Method #5 & #14):
        Audits answer for safety, hallucinated claims, empty answers, and completeness.
        Returns: (passed, critique_summary, recommendations)
        """
        critiques = []
        recommendations = []

        # 1. Non-empty check
        if not response or len(response.strip()) < 5:
            critiques.append("Response is empty or trivially brief.")
            recommendations.append("Generate a structured, complete answer addressing the prompt.")
            return False, "; ".join(critiques), recommendations

        # 2. Refusal / Evasiveness check
        evasive_patterns = ["as an ai", "i cannot answer", "i am an ai language model", "as a large language model"]
        if any(p in response.lower() for p in evasive_patterns) and len(response.strip()) < 100:
            critiques.append("Response exhibits unnecessary canned refusal or evasion.")
            recommendations.append("Directly address the user's objective with factual explanation.")

        # 3. Prompt alignment check
        prompt_words = set(re.findall(r"\w+", prompt.lower()))
        response_words = set(re.findall(r"\w+", response.lower()))
        overlap = len(prompt_words.intersection(response_words)) / max(1, len(prompt_words))
        if overlap < 0.10 and len(prompt_words) > 3:
            critiques.append("Low semantic alignment between prompt query and output response.")
            recommendations.append("Ensure key nouns and questions from the prompt are directly addressed.")

        # 4. Code safety check
        dangerous_calls = ["os.system", "shutil.rmtree", "subprocess.Popen", "__import__('os')", "eval(input("]
        for danger in dangerous_calls:
            if danger in response:
                critiques.append(f"Dangerous call '{danger}' detected in output.")
                recommendations.append("Sanitize or wrap in safe isolated sandbox.")

        # 5. Anti-Sycophancy & Epistemic Boundary check
        from .truth_engine import TruthAndEpistemicEngine
        truth_audit = TruthAndEpistemicEngine.audit_response(prompt, response)
        if not truth_audit["passed"]:
            critiques.append(truth_audit["critique"])
            recommendations.append(f"Replace with truthful correction or honest abstention: {truth_audit['corrected_response'][:100]}...")

        passed = len(critiques) == 0
        summary = "Passed Constitutional AI Critique (Safety, Anti-Sycophancy, Epistemic Calibration verified)." if passed else "; ".join(critiques)
        return passed, summary, recommendations

    @classmethod
    def self_correct_response(cls, prompt: str, response: str) -> Tuple[str, bool]:
        """
        Applies automatic correction if sycophantic flattery or hallucination is detected.
        Returns: (final_response, was_corrected)
        """
        from .truth_engine import TruthAndEpistemicEngine
        audit = TruthAndEpistemicEngine.audit_response(prompt, response)
        if not audit["passed"]:
            return audit["corrected_response"], True
        return response, False


