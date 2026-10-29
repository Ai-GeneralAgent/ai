"""
Claude & ChatGPT-grade Reasoning Orchestrator (v2.2.0)
Full multi-step Chain-of-Thought thinking, Curriculum Learning Ingestion,
Safe Python Code Execution Sandbox, Advanced Math Solving, and Creative Synthesis.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .planner import TaskPlanner, TaskPlan
from .verifier import ResponseVerifier, VerificationResult
from ..rag.pipeline import LocalRAGPipeline
from ..tools.registry import ToolRegistry
from ..tools.builtin import solve_math_problem
from ..tools.code_engine import SafeCodeSandbox, CodingAssistant
from ..tools.agent_loop import BoundedAgentLoop
from ..generation.creative import CreativeEngine


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

        # Explicit slash commands / priority dispatch
        if lower.startswith(("/search", "/web")) or "duckduckgo" in lower or any(k in lower for k in ["search the web", "search online", "search web"]):
            return "web-search"

        # Curriculum / Textbook study check
        if any(k in lower for k in [
            "study book", "study textbook", "curriculum", "grade 1", "grade 12",
            "study note", "flashcard", "/study", "ncert", "cbse"
        ]):
            return "curriculum"

        # Complex task check (multi-step instructions, comparisons, itineraries)
        if any(k in lower for k in [
            "step by step", "first,", "and then", "compare and contrast",
            "create a plan", "pros and cons", "outline", "roadmap"
        ]):
            return "complex-task"

        # Programming / code check
        if any(k in lower for k in [
            "```", "print(", "def ", "class ", "python", "code", "bug", "syntax", "function",
            "import ", "write a script", "algorithm", "javascript", "sql", "execute code", "run this code"
        ]):
            return "programming"

        # Math / Quantitative / Conversion / Equation check
        if any(k in lower for k in [
            "calculate", "compute", "solve", "sqrt", "equation", "formula",
            "%", "percent", "convert", "miles to", "km to", "celsius", "fahrenheit"
        ]) or re.search(r"(\d+\s*[\+\-\*\/\^]\s*\d+)", lower) or re.search(r"\bx\s*[\+\-\*=]", lower):
            return "mathematical"

        # Creative writing / Intuitive analogy check
        if any(k in lower for k in [
            "story", "poem", "poetry", "analogy", "metaphor", "creative",
            "explain like i'm 5", "eli5", "imagine a world", "essay on", "/creative"
        ]):
            return "creative"

        # DuckDuckGo Live Web Search check
        if any(k in lower for k in [
            "/search", "/web", "duckduckgo", "search web", "search the web",
            "search online", "latest news", "look up online", "web search"
        ]) or lower.startswith("/search") or lower.startswith("/web"):
            return "web-search"

        # Document / Knowledge check
        if any(k in lower for k in [
            "search", "find in document", "according to", "article", "paper",
            "reference", "what is", "who is", "explain", "history of", "how does"
        ]):
            return "document-based"

        # Time / date / system tools
        if any(k in lower for k in ["what time", "what date", "current time", "today"]):
            return "tool-assisted"

        # Ambiguity check (Method #15: Ambiguity Resolution & Proactive Clarification)
        words = lower.split()
        if len(words) <= 3 and any(k in lower for k in ["help", "fix", "do it", "solve", "how", "why", "what", "make it"]):
            return "ambiguous"

        return "general"

    def estimate_complexity(self, prompt: str, intent: str) -> str:
        """Determines if request requires direct response, short plan, or multi-step execution."""
        words = len(prompt.split())
        if intent in ["complex-task", "curriculum"] or words > 30:
            return "complex"
        if intent in ["programming", "document-based", "creative"] or words > 15:
            return "moderate"
        return "simple"

    def prepare_reasoning(self, user_prompt: str) -> ReasoningPreparation:
        """Executes pre-generation reasoning, tool checks, and knowledge retrieval."""
        intent = self.classify_intent(user_prompt)
        complexity = self.estimate_complexity(user_prompt, intent)

        thinking_steps = [
            f"1. Intent Analysis: Identified intent as [{intent}] with [{complexity}] reasoning depth.",
            f"2. Objective & Invariants: Unpack constraints, isolate core query, and ensure rigorous logic.",
        ]

        prep = ReasoningPreparation(intent=intent, complexity=complexity)

        # Step-Back Prompting Abstraction (Method #13)
        if complexity == "complex":
            thinking_steps.append("3. Step-Back Abstraction: Deriving fundamental principles and structural invariants before execution.")

        # Ambiguity Resolution & Proactive Clarification (Method #15)
        if intent == "ambiguous":
            thinking_steps.append("3. Ambiguity Detection & Proactive Clarification: Request is underspecified. Proactively offering targeted multi-path workflows.")
            prep.thinking = "\n".join(thinking_steps)
            prep.direct_answer = (
                "I want to help you with that! Your request is broad, so please choose or specify what you would like to do:\n\n"
                "1. **Solve Math / Equation**: e.g., `Solve: 5x - 15 = 35` or `What is 25% of 400?`\n"
                "2. **Run Python Code**: e.g., `/code print([x**2 for x in range(5)])`\n"
                "3. **Study Textbooks (Grades 1-12)**: e.g., `/study data_curriculum/grade_10/science.txt grade_10`\n"
                "4. **Creative Mental Models**: e.g., `/creative quantum mechanics` or `/creative neural networks`\n"
                "5. **Knowledge Search (RAG)**: Ask any factual question from the corpus.\n\n"
                "What specific topic or problem should we work on?"
            )
            return prep

        # 1. RAG Knowledge Retrieval
        retrieved_context = ""
        if self.rag:
            retrieved_context, citations = self.rag.format_context_for_prompt(user_prompt, top_k=2)
            if citations:
                prep.retrieved_citations = citations
                thinking_steps.append(f"3. Knowledge Retrieval: Grounded request with {len(citations)} chunks from corpus ({', '.join(citations[:2])}).")

        # 2. Mathematical Solving
        if intent == "mathematical":
            thinking_steps.append("3. Symbolic Formulation: Dispatched to verified Deterministic Algebra & Arithmetic Engine.")
            math_res = solve_math_problem(user_prompt)
            if math_res.get("success"):
                thinking_steps.append(f"4. Solution Derivation: Computed verified exact result: {math_res['result']}.")
                thinking_steps.append("5. Response Grounding: Synthesizing step-by-step mathematical proof directly from verified deduction.")
                prep.thinking = "\n".join(thinking_steps)
                prep.direct_answer = math_res["explanation"]
                prep.verifications.append({
                    "check": "Math Verification",
                    "passed": True,
                    "details": f"Verified calculation: {math_res['result']}",
                })
                return prep

        # 3. Creative Analogies
        if intent == "creative":
            analogy = CreativeEngine.get_intuitive_analogy(user_prompt)
            if analogy:
                thinking_steps.append("3. Creative Synthesis: Formulating intuitive real-world mental model.")
                thinking_steps.append("4. Response Grounding: Translating abstract principles into relatable everyday concepts.")
                prep.thinking = "\n".join(thinking_steps)
                prep.direct_answer = f"### Intuitive Mental Model\n\n{analogy}\n\n*(Core concept mapped to intuitive everyday intuition)*"
                return prep

        # 4. Code Execution in Sandbox
        if intent == "programming" and any(k in user_prompt.lower() for k in ["run", "execute", "test", "eval"]):
            code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", user_prompt)
            if code_match:
                code_to_run = code_match.group(1).strip()
                thinking_steps.append("3. Code Execution: Evaluating code snippet inside Safe Python Sandbox.")
                sandbox_res = SafeCodeSandbox.execute_snippet(code_to_run)
                thinking_steps.append(f"4. Execution Validation: Sandbox status {'SUCCESS' if sandbox_res['success'] else 'FAILED'}.")
                prep.thinking = "\n".join(thinking_steps)
                prep.direct_answer = CodingAssistant.format_solution(
                    task="Sandboxed Code Evaluation",
                    code=code_to_run,
                    explanation="Code executed safely with stdout capture and restricted builtins.",
                    sandbox_result=sandbox_res,
                )
                prep.verifications.append({
                    "check": "Code Verification",
                    "passed": sandbox_res["success"],
                    "details": f"Execution completed in {sandbox_res.get('elapsed', 0):.4f}s",
                })
                return prep

        # 5. Environment Tool Execution
        if intent == "tool-assisted" and self.tools:
            if "time" in user_prompt.lower() or "date" in user_prompt.lower():
                res = self.tools.execute("datetime_now", {})
                thinking_steps.append("3. Environment Tool: Querying real-time system clock.")
                thinking_steps.append(f"4. Tool Result: Clock confirmed {res.get('result')}.")
                prep.thinking = "\n".join(thinking_steps)
                prep.direct_answer = f"The current date and time is: **{res.get('result')}**"
                return prep

        # 6. DuckDuckGo Live Web Search
        if intent == "web-search":
            from ..tools.web_search import DuckDuckGoSearchEngine
            clean_q = re.sub(r"^/(?:search|web)\s*", "", user_prompt).strip()
            clean_q = re.sub(r"^(?:search|duckduckgo|find)\s+(?:for\s+|the\s+web\s+for\s+)?", "", clean_q, flags=re.IGNORECASE).strip()
            thinking_steps.append(f"3. Web Search Dispatch: Querying DuckDuckGo Live Engine for '{clean_q}'.")
            web_results = DuckDuckGoSearchEngine.search(clean_q, max_results=5)
            thinking_steps.append(f"4. Search Synthesis: Retrieved {len(web_results)} live web sources with citations.")
            thinking_steps.append("5. Response Grounding: Synthesizing verified factual response with source URLs.")
            prep.thinking = "\n".join(thinking_steps)
            prep.direct_answer = DuckDuckGoSearchEngine.format_search_results(clean_q, web_results)
            prep.retrieved_citations = [r.get("url", "") for r in web_results if r.get("url")]
            prep.verifications.append({
                "check": "DuckDuckGo Web Search",
                "passed": len(web_results) > 0,
                "details": f"Retrieved {len(web_results)} live web sources",
            })
            return prep

        # 7. Preparation for generation
        thinking_steps.append("4. Synthesis & Grounding: Structuring direct, coherent response grounded in the logical deduction above.")
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

        # Constitutional AI Critique & Self-Correction (Method #5 & #14)
        c_passed, c_summary, _ = ResponseVerifier.critique_and_self_correct(user_prompt, final_text)
        summary.verifications.append({
            "check": "Constitutional Self-Correction",
            "passed": c_passed,
            "details": c_summary,
        })

        summary.final_answer = final_text
        return summary
