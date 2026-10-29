from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SubTask:
    id: int
    description: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    status: str = "pending"  # "pending", "completed", "failed"
    result: Optional[Any] = None


@dataclass
class TaskPlan:
    goal: str
    intent: str
    complexity: str  # "simple", "moderate", "complex"
    subtasks: List[SubTask] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    verification_checks: List[str] = field(default_factory=list)
    completed: bool = False


class TaskPlanner:
    """Generates structured task plans and decomposes complex user requests."""

    @classmethod
    def create_plan(cls, goal: str, intent: str, complexity: str) -> TaskPlan:
        plan = TaskPlan(goal=goal, intent=intent, complexity=complexity)

        if intent == "mathematical":
            plan.subtasks.append(SubTask(
                id=1,
                description="Parse mathematical structure (arithmetic, algebra, percentage, or unit conversion).",
                tool="math_solver",
                args={"query": goal},
                expected_output="Verified numerical solution and step-by-step derivation",
            ))
            plan.verification_checks.append("Verify calculation correctness and dimensional consistency.")

        elif intent == "complex-task":
            plan.subtasks.append(SubTask(
                id=1,
                description="Decompose objective into sequential requirements and constraints.",
                expected_output="Logical execution blueprint",
            ))
            plan.subtasks.append(SubTask(
                id=2,
                description="Execute intermediate reasoning and retrieve supporting knowledge or tools.",
                expected_output="Subtask artifacts and computations",
            ))
            plan.subtasks.append(SubTask(
                id=3,
                description="Synthesize structured comprehensive response.",
                expected_output="Final organized answer",
            ))
            plan.verification_checks.append("Verify all requirements from prompt are fully satisfied.")

        elif intent == "programming":
            plan.subtasks.append(SubTask(
                id=1,
                description="Design algorithm and define clean Python code implementation.",
                expected_output="Python code snippet with docstrings and type annotations",
            ))
            plan.verification_checks.append("Verify Python code syntax via AST parser.")

        elif intent == "document-based":
            plan.subtasks.append(SubTask(
                id=1,
                description="Retrieve relevant document chunks via local BM25 RAG.",
                expected_output="Citations and evidence text",
            ))
            plan.verification_checks.append("Confirm all factual assertions match retrieved citations.")

        elif intent == "tool-assisted":
            plan.subtasks.append(SubTask(
                id=1,
                description="Query registered environment tools (datetime, filesystem, or codebase).",
                expected_output="Deterministic tool response",
            ))

        else:
            plan.subtasks.append(SubTask(
                id=1,
                description="Formulate direct, helpful, and logical response.",
                expected_output="Well-reasoned response",
            ))

        return plan
