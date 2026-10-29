from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from .planner import TaskPlanner, TaskPlan
from .verifier import ResponseVerifier, VerificationResult
from ..rag.pipeline import LocalRAGPipeline
from ..tools.registry import ToolRegistry
from ..tools.agent_loop import BoundedAgentLoop


@dataclass
class ReasoningSummary:
    intent: str
    complexity: str
    plan: Optional[TaskPlan] = None
    retrieved_citations: List[str] = field(default_factory=list)
    verifications: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""


class ReasoningOrchestrator:
    """
    Modular reasoning orchestration pipeline:
    Intent -> Complexity -> Retrieval -> Plan -> Execute -> Verify -> Answer.
    """

    def __init__(
        self,
        rag_pipeline: Optional[LocalRAGPipeline] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.rag = rag_pipeline
        self.tools = tool_registry
        self.agent_loop = BoundedAgentLoop(tool_registry) if tool_registry else None

    def classify_intent(self, prompt: str) -> str:
        """Deterministic intent classifier."""
        lower = prompt.lower().strip()

        # Math check
        if re.search(r"(\d+\s*[\+\-\*\/]\s*\d+|calculate|compute|sqrt|math|equation)", lower):
            return "mathematical"

        # Programming / code check
        if any(k in lower for k in ["def ", "class ", "python", "code", "bug", "syntax", "function", "import "]):
            return "programming"

        # Document / search check
        if any(k in lower for k in ["search", "find in document", "according to", "article", "paper", "reference"]):
            return "document-based"

        # Time / date
        if any(k in lower for k in ["what time", "what date", "current time", "today"]):
            return "tool-assisted"

        return "general"

    def estimate_complexity(self, prompt: str, intent: str) -> str:
        """Determines if request requires direct response, short plan, or multi-step execution."""
        words = len(prompt.split())
        if intent in ["mathematical", "tool-assisted"]:
            return "simple"
        if intent in ["programming", "document-based"] or words > 25:
            return "moderate"
        return "simple"

    def orchestrate(
        self,
        user_prompt: str,
        generator_fn=None,
    ) -> ReasoningSummary:
        """Executes full reasoning lifecycle."""
        intent = self.classify_intent(user_prompt)
        complexity = self.estimate_complexity(user_prompt, intent)
        plan = TaskPlanner.create_plan(user_prompt, intent, complexity)
        summary = ReasoningSummary(intent=intent, complexity=complexity, plan=plan)

        # 1. Retrieval stage if document-based or if RAG is available
        retrieved_context = ""
        citations = []
        if self.rag and intent == "document-based":
            retrieved_context, citations = self.rag.format_context_for_prompt(user_prompt, top_k=2)
            summary.retrieved_citations = citations

        # 2. Tool-assisted execution for deterministic tools
        if intent == "mathematical" and self.tools:
            # Extract expression
            math_match = re.search(r"(\d+[\d\s\+\-\*\/\.\(\)]+)", user_prompt)
            if math_match:
                expr = math_match.group(1).strip()
                res = self.tools.execute("calculator", {"expression": expr})
                if res.get("success"):
                    summary.final_answer = f"The calculated result for `{expr}` is: **{res['result']}**"
                    summary.verifications.append({"check": "Math Verification", "passed": True, "details": "Direct calculator execution."})
                    return summary

        if intent == "tool-assisted" and self.tools:
            if "time" in user_prompt.lower() or "date" in user_prompt.lower():
                res = self.tools.execute("datetime_now", {})
                summary.final_answer = f"The current date and time is: **{res.get('result')}**"
                return summary

        # 3. Model generation fallback
        final_text = ""
        if generator_fn:
            prompt_with_context = user_prompt
            if retrieved_context:
                prompt_with_context = f"{retrieved_context}\n\nQuestion: {user_prompt}"
            final_text = generator_fn(prompt_with_context)
        else:
            final_text = f"Processed intent [{intent}] for request: {user_prompt}"

        # 4. Independent Verification
        math_check = ResponseVerifier.verify_math_claims(final_text)
        summary.verifications.append(math_check.to_dict())

        code_check = ResponseVerifier.verify_python_code(final_text)
        summary.verifications.append(code_check.to_dict())

        if citations:
            cite_check = ResponseVerifier.verify_citations(final_text, citations)
            summary.verifications.append(cite_check.to_dict())

        summary.final_answer = final_text
        return summary

