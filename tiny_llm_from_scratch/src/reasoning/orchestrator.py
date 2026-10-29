from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .planner import TaskPlanner, TaskPlan
from .verifier import ResponseVerifier, VerificationResult
from ..rag.pipeline import LocalRAGPipeline
from ..tools.registry import ToolRegistry
from ..tools.builtin import solve_math_problem
from ..tools.agent_loop import BoundedAgentLoop


@dataclass
class ReasoningPreparation:
    intent: str
    complexity: str
    thinking: str = ""
    prompt_with_context: str = ""
    direct_answer: Optional[str] = None
    retrieved_citations: List[str] = field(default_factory=list)
    verifications: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReasoningSummary:
    intent: str
    complexity: str
    thinking: str = ""
    plan: Optional[TaskPlan] = None
    retrieved_citations: List[str] = field(default_factory=list)
    verifications: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""


class ReasoningOrchestrator:
    """
    Claude & ChatGPT-grade reasoning orchestration engine:
    Decomposition -> Multi-Step Thinking -> Tool/Knowledge Dispatch -> Verification -> Synthesis.
    """

    def __init__(
        self,
        rag_pipeline: Optional[LocalRAGPipeline] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_thinking: bool = True,
    ):
        self.rag = rag_pipeline
        self.tools = tool_registry
        self.enable_thinking = enable_thinking
        self.agent_loop = BoundedAgentLoop(tool_registry) if tool_registry else None

    def classify_intent(self, prompt: str) -> str:
        """Deterministic intent classifier."""
        lower = prompt.lower().strip()

        # Complex task check (multi-step instructions, comparisons, itineraries)
        if any(k in lower for k in [
            "step by step", "first,", "and then", "compare and contrast",
            "create a plan", "pros and cons", "outline", "roadmap"
        ]):
            return "complex-task"

        # Math / Quantitative / Conversion / Equation check
        if any(k in lower for k in [
            "calculate", "compute", "solve", "sqrt", "equation", "formula",
            "%", "percent", "convert", "miles to", "km to", "celsius", "fahrenheit"
        ]) or re.search(r"(\d+\s*[\+\-\*\/\^]\s*\d+)", lower) or re.search(r"x\s*[\+\-\*=]", lower):
            return "mathematical"

        # Programming / code check
        if any(k in lower for k in [
            "def ", "class ", "python", "code", "bug", "syntax", "function",
            "import ", "write a script", "algorithm", "javascript", "sql"
        ]):
            return "programming"

        # Document / Knowledge check
        if any(k in lower for k in [
            "search", "find in document", "according to", "article", "paper",
            "reference", "what is", "who is", "explain", "history of", "how does"
        ]):
            return "document-based"

        # Time / date / system tools
        if any(k in lower for k in ["what time", "what date", "current time", "today"]):
            return "tool-assisted"

        return "general"

    def estimate_complexity(self, prompt: str, intent: str) -> str:
        """Determines if request requires direct response, short plan, or multi-step execution."""
        words = len(prompt.split())
        if intent == "complex-task" or words > 30:
            return "complex"
        if intent in ["programming", "document-based"] or words > 15:
            return "moderate"
        return "simple"

    def prepare_reasoning(self, user_prompt: str) -> ReasoningPreparation:
        """Executes pre-generation reasoning, tool checks, and knowledge retrieval."""
        intent = self.classify_intent(user_prompt)
        complexity = self.estimate_complexity(user_prompt, intent)

        thinking_steps = []
        thinking_steps.append(f"1. Intent Analysis: Identified goal as [{intent}] with complexity [{complexity}].")

        prep = ReasoningPreparation(intent=intent, complexity=complexity)

        # 1. RAG Knowledge Retrieval
        retrieved_context = ""
        if self.rag:
            retrieved_context, citations = self.rag.format_context_for_prompt(user_prompt, top_k=2)
            if citations:
                prep.retrieved_citations = citations
                thinking_steps.append(f"2. Knowledge Retrieval: Grounded request with {len(citations)} chunks from corpus ({', '.join(citations[:2])}).")

        # 2. Mathematical Solving
        if intent == "mathematical":
            thinking_steps.append("2. Mathematical Formulation: Dispatching to verified Advanced Math & Algebra Solver.")
            math_res = solve_math_problem(user_prompt)
            if math_res.get("success"):
                thinking_steps.append(f"3. Verification: Deterministic solution calculated: {math_res['result']}.")
                prep.thinking = "\n".join(thinking_steps)
                prep.direct_answer = math_res["explanation"]
                prep.verifications.append({
                    "check": "Math Verification",
                    "passed": True,
                    "details": f"Verified calculation: {math_res['result']}",
                })
                return prep

        # 3. Tool Execution
        if intent == "tool-assisted" and self.tools:
            if "time" in user_prompt.lower() or "date" in user_prompt.lower():
                res = self.tools.execute("datetime_now", {})
                thinking_steps.append("2. Environment Tool: Querying real-time system clock.")
                prep.thinking = "\n".join(thinking_steps)
                prep.direct_answer = f"The current date and time is: **{res.get('result')}**"
                return prep

        # 4. Preparation for generation
        thinking_steps.append("3. Synthesis: Structuring comprehensive, logical response.")
        prompt_with_context = user_prompt
        if retrieved_context:
            prompt_with_context = f"Context Information:\n{retrieved_context}\n\nTask / Question:\n{user_prompt}"

        prep.prompt_with_context = prompt_with_context
        prep.thinking = "\n".join(thinking_steps)
        return prep

    def orchestrate(
        self,
        user_prompt: str,
        generator_fn: Optional[Callable[[str], str]] = None,
    ) -> ReasoningSummary:
        """Executes end-to-end reasoning and returns structured summary."""
        prep = self.prepare_reasoning(user_prompt)
        plan = TaskPlanner.create_plan(user_prompt, prep.intent, prep.complexity)

        summary = ReasoningSummary(
            intent=prep.intent,
            complexity=prep.complexity,
            thinking=prep.thinking,
            plan=plan,
            retrieved_citations=prep.retrieved_citations,
            verifications=prep.verifications,
        )

        if prep.direct_answer:
            summary.final_answer = prep.direct_answer
            return summary

        final_text = ""
        if generator_fn:
            final_text = generator_fn(prep.prompt_with_context)
        else:
            final_text = f"Processed intent [{prep.intent}] for: {user_prompt}"

        # Independent Verifications
        math_check = ResponseVerifier.verify_math_claims(final_text)
        summary.verifications.append(math_check.to_dict())

        code_check = ResponseVerifier.verify_python_code(final_text)
        summary.verifications.append(code_check.to_dict())

        if prep.retrieved_citations:
            cite_check = ResponseVerifier.verify_citations(final_text, prep.retrieved_citations)
            summary.verifications.append(cite_check.to_dict())

        summary.final_answer = final_text
        return summary
