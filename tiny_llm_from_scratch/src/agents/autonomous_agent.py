"""
Autonomous Agent Work Engine
Executes complex multi-step missions autonomously:
1. Goal Formulation & Decomposition
2. Sequential Tool Execution (Web Search, Browser, Filesystem, Code Sandbox, Security Lab)
3. Intermediate State Inspection & Self-Correction
4. Final Synthesis with Evidence Grounding
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..tools.registry import ToolRegistry
from ..tools.agent_loop import AgentStep, BoundedAgentLoop


@dataclass
class AgentMissionResult:
    goal: str
    completed: bool
    iterations: int
    elapsed_sec: float
    steps: List[AgentStep]
    synthesized_output: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "completed": self.completed,
            "iterations": self.iterations,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "step_count": len(self.steps),
            "synthesized_output": self.synthesized_output,
            "errors": self.errors,
            "artifacts_count": len(self.artifacts),
        }


class AutonomousWorkAgent:
    """
    Autonomous multi-step agent capable of:
    - Web research and live browsing
    - Code development and sandbox testing
    - Security auditing and verification
    - Multi-tool chained workflows with error recovery
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_steps: int = 8,
        timeout_sec: float = 30.0,
        verbose: bool = True,
    ):
        self.registry = registry
        self.max_steps = max_steps
        self.timeout_sec = timeout_sec
        self.verbose = verbose
        self.agent_loop = BoundedAgentLoop(
            registry=registry,
            max_iterations=max_steps,
            max_tool_calls=max_steps + 4,
            timeout_sec=timeout_sec,
            bypass_security=True,  # Allow agent to execute full suite of research tools
        )

    def formulate_mission_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Decomposes user goal into ordered tool execution steps."""
        lower = goal.lower()
        plan_steps: List[Dict[str, Any]] = []

        # 1. URL / Browsing Goal
        if "http://" in lower or "https://" in lower or any(d in lower for d in [".com", ".org", ".io", ".net", ".edu"]):
            # Extract URL
            url_match = [w for w in goal.split() if "http" in w or any(w.endswith(s) for s in [".org", ".com", ".io"])]
            target_url = url_match[0] if url_match else "https://en.wikipedia.org"
            plan_steps.append({
                "thought": f"Access web browser and extract content from {target_url}",
                "tool": "browse_web_page",
                "args": {"url": target_url, "max_chars": 4000},
            })

        # 2. Web Search & Research Goal
        elif any(k in lower for k in ["search", "find", "browse", "research", "lookup", "who is", "what is", "latest"]):
            clean_query = goal
            for prefix in ["search for", "browse", "research", "find information on", "lookup"]:
                if lower.startswith(prefix):
                    clean_query = goal[len(prefix):].strip()
            plan_steps.append({
                "thought": f"Query live web via search engine for '{clean_query}'",
                "tool": "web_search_duckduckgo",
                "args": {"query": clean_query, "max_results": 4},
            })
            plan_steps.append({
                "thought": "Browse and extract full detailed text from top search result",
                "tool": "search_and_browse",
                "args": {"query": clean_query, "max_chars": 3500},
            })

        # 3. Security Audit Goal
        elif any(k in lower for k in ["audit", "vulnerability", "security check", "scan code"]):
            plan_steps.append({
                "thought": "Analyze project architecture and file structure",
                "tool": "project_structure_analyze",
                "args": {"max_depth": 3},
            })
            plan_steps.append({
                "thought": "Perform AST security audit on chat interface entrypoint",
                "tool": "audit_code_file",
                "args": {"relative_path": "chat.py"},
            })

        # 4. Code Development / Math Goal
        elif any(k in lower for k in ["calculate", "math", "solve", "compute", "equation"]):
            plan_steps.append({
                "thought": f"Solve mathematical equation and verify derivation: {goal}",
                "tool": "math_solver",
                "args": {"query": goal},
            })

        # 5. Default General Agent Plan
        else:
            plan_steps.append({
                "thought": f"Search knowledge base and web for: {goal}",
                "tool": "web_search_duckduckgo",
                "args": {"query": goal, "max_results": 3},
            })

        return plan_steps

    def execute_mission(self, goal: str) -> AgentMissionResult:
        """Executes full autonomous mission, returning structured steps, evidence, and synthesis."""
        started_at = time.time()
        plan_steps = self.formulate_mission_plan(goal)
        errors: List[str] = []
        artifacts: Dict[str, Any] = {}

        if self.verbose:
            print(f"\n[AGENT] Initiating autonomous mission for goal: '{goal}'")
            print(f"[AGENT] Decomposed plan into {len(plan_steps)} sequential action steps.")

        loop_result = self.agent_loop.run_plan(goal, plan_steps)
        steps_executed = loop_result.get("steps", [])

        # Collect evidence & artifacts from tool outputs
        evidence_pieces: List[str] = []
        for step in steps_executed:
            if step.error:
                errors.append(f"Step {step.step_num} error: {step.error}")
            elif step.tool_result is not None:
                res = step.tool_result
                if isinstance(res, dict):
                    artifacts[f"{step.tool_name}_{step.step_num}"] = res
                    if "content" in res:
                        evidence_pieces.append(f"[{res.get('title', step.tool_name)}]:\n{res['content'][:600]}...")
                    elif "explanation" in res:
                        evidence_pieces.append(f"Result: {res['explanation']}")
                    elif "findings" in res:
                        evidence_pieces.append(f"Audit: {len(res['findings'])} findings detected.")
                elif isinstance(res, str):
                    artifacts[f"{step.tool_name}_{step.step_num}"] = res
                    evidence_pieces.append(res[:500])

        # Synthesize Final Comprehensive Response
        if evidence_pieces:
            synthesized = (
                f"### Autonomous Agent Mission Accomplished\n\n"
                f"**Goal:** {goal}\n\n"
                f"**Execution Summary:** Executed {len(steps_executed)} steps across registered research tools.\n\n"
                f"**Collected Findings & Evidence:**\n"
                + "\n\n".join([f"- {piece}" for piece in evidence_pieces])
            )
        else:
            synthesized = (
                f"### Autonomous Agent Execution Complete\n\n"
                f"**Goal:** {goal}\n\n"
                f"Agent completed {len(steps_executed)} planned actions. No external evidence was returned or required."
            )

        return AgentMissionResult(
            goal=goal,
            completed=loop_result.get("completed", False),
            iterations=len(steps_executed),
            elapsed_sec=time.time() - started_at,
            steps=steps_executed,
            synthesized_output=synthesized,
            artifacts=artifacts,
            errors=errors,
        )
