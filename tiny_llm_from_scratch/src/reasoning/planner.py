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
    """Generates structured task plans for complex requests."""

    @classmethod
    def create_plan(cls, goal: str, intent: str, complexity: str) -> TaskPlan:
        plan = TaskPlan(goal=goal, intent=intent, complexity=complexity)

        if intent == "mathematical":
            plan.subtasks.append(SubTask(
                id=1,
                description="Extract arithmetic equation and evaluate with verified calculator.",
                tool="calculator",
                args={"expression": goal},
                expected_output="Deterministic numerical value",
            ))
            plan.verification_checks.append("Verify calculation consistency.")

        elif intent == "programming":
            plan.subtasks.append(SubTask(
                id=1,
                description="Formulate code solution.",
                expected_output="Python code snippet",
            ))
            plan.verification_checks.append("Verify Python code syntax via AST parser.")

        elif intent == "document-based":
            plan.subtasks.append(SubTask(
                id=1,
                description="Retrieve relevant document chunks via local RAG.",
                expected_output="Citations and evidence text",
            ))
            plan.verification_checks.append("Confirm all factual claims are backed by citations.")

        else:
            plan.subtasks.append(SubTask(
                id=1,
                description="Formulate direct response.",
                expected_output="Helpful answer",
            ))

        return plan

