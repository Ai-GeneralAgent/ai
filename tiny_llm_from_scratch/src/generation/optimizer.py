"""
Output Optimizer for Modern LLM Generation.
Cleans raw text outputs, eliminates repetitive loops, normalizes markdown and code syntax,
and harmonizes model responses with the internal thinking trace.
"""
import re
from typing import Optional


class OutputOptimizer:
    """Multi-pass output polisher and artifact remover."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Strips replacement chars, fixes spacing, and cleans subword artifacts."""
        if not text:
            return ""
        # Remove unicode replacement char and control characters
        text = text.replace("\ufffd", "").replace("\xa0", " ").replace("\u200b", "")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        # Standardize whitespace and line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def eliminate_repetition(text: str) -> str:
        """Detects and cuts off degenerate repetitive token loops or duplicate sentences."""
        if not text:
            return ""
        lines = text.split("\n")
        cleaned_lines = []
        seen_lines = set()
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 25 and stripped in seen_lines:
                continue
            if stripped:
                seen_lines.add(stripped)
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Clean repeating adjacent word loops: e.g. "receive receive receive" -> "receive"
        text = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)
        # Clean repeating 2-word phrase loops: e.g. "shut ince shut ince shut ince"
        text = re.sub(r"\b(\w+\s+\w+)(?:\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def format_markdown(text: str) -> str:
        """Ensures balanced code blocks, clean bullet points, and headers."""
        if not text:
            return ""
        # Balance backticks
        code_blocks = text.count("```")
        if code_blocks % 2 != 0:
            text = text + "\n```"

        # Ensure bullet lists have space after dash
        text = re.sub(r"^-(?=[A-Za-z0-9])", "- ", text, flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def clean_branding(text: str) -> str:
        """Removes third-party AI branding and enforces sovereign identity as LLM by Yuvraj Thakur."""
        if not text:
            return ""
        # Scrub third-party model claims
        text = re.sub(r"(?i)\b(i am|i'm)\s+(?:a large language model trained by\s+)?(?:qwen|alibaba(?: cloud)?|tongyi(?: qianwen)?)\b", "I am LLM, created and built by Yuvraj Thakur", text)
        text = re.sub(r"(?i)\bcreated by (?:alibaba(?: cloud)?|qwen team)\b", "created by Yuvraj Thakur", text)
        text = re.sub(r"(?i)\bdeveloped by (?:alibaba(?: cloud)?|qwen team)\b", "developed by Yuvraj Thakur", text)
        text = re.sub(r"(?i)\b(as\s+(?:an?\s+)?(?:qwen|alibaba|tongyi)(?:\s+model)?)\b", "as LLM", text)
        return text

    @classmethod
    def optimize(cls, text: str, thinking_trace: Optional[str] = None, intent: Optional[str] = None) -> str:
        """Full multi-pass optimization pipeline."""
        if not text:
            return ""
        step1 = cls.clean_text(text)
        step2 = cls.eliminate_repetition(step1)
        step3 = cls.format_markdown(step2)
        step4 = cls.clean_branding(step3)
        return step4

