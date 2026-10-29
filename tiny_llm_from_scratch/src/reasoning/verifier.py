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

