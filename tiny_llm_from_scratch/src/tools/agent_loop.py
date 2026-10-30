from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


@dataclass
class AgentStep:
    step_num: int
    thought: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    error: Optional[str] = None


class BoundedAgentLoop:
    """Predictable and bounded agent execution loop with cycle detection and limits."""

    def __init__(
        self,
        registry: ToolRegistry,
        max_iterations: int = 5,
        max_tool_calls: int = 8,
        timeout_sec: float = 15.0,
        allowed_tools: Optional[Any] = None,
        bypass_security: bool = False,
    ):
        self.registry = registry
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.timeout_sec = timeout_sec
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.bypass_security = bypass_security

    def set_bypass_security(self, enabled: bool = True):
        """Allows bypassing security allowlists for full capability testing."""
        self.bypass_security = enabled
        if self.registry:
            self.registry.set_unrestricted_mode(enabled)

    def run_plan(self, task_goal: str, plan_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes a structured task plan within bounds."""
        steps_executed: List[AgentStep] = []
        seen_calls = set()
        started_at = time.time()
        tool_call_count = 0

        for i, step_def in enumerate(plan_steps, 1):
            if i > self.max_iterations:
                return {
                    "completed": False,
                    "reason": f"Exceeded maximum iterations limit ({self.max_iterations})",
                    "steps": steps_executed,
                }

            if time.time() - started_at > self.timeout_sec:
                return {
                    "completed": False,
                    "reason": f"Execution timed out ({self.timeout_sec}s limit)",
                    "steps": steps_executed,
                }

            thought = step_def.get("thought", f"Executing step {i}")
            tool_name = step_def.get("tool")
            tool_args = step_def.get("args", {})

            if not tool_name:
                steps_executed.append(AgentStep(step_num=i, thought=thought))
                continue

            # Check session allowlist (unless in full capability bypass mode)
            if not self.bypass_security and self.allowed_tools is not None and tool_name not in self.allowed_tools:
                step_rec = AgentStep(
                    step_num=i,
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    error=f"Security violation: tool '{tool_name}' blocked by session allowlist policy.",
                )
                steps_executed.append(step_rec)
                continue

            # Cycle detection (same tool + same args repeated)
            call_sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            if call_sig in seen_calls:
                step_rec = AgentStep(
                    step_num=i,
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    error="Cycle detected: skipped repeated identical tool invocation.",
                )
                steps_executed.append(step_rec)
                continue
            seen_calls.add(call_sig)

            tool_call_count += 1
            if tool_call_count > self.max_tool_calls:
                return {
                    "completed": False,
                    "reason": f"Exceeded maximum tool calls limit ({self.max_tool_calls})",
                    "steps": steps_executed,
                }

            # Execute tool
            exec_res = self.registry.execute(tool_name, tool_args)
            step_rec = AgentStep(
                step_num=i,
                thought=thought,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=exec_res.get("result"),
                error=exec_res.get("error") if not exec_res.get("success") else None,
            )
            steps_executed.append(step_rec)

        return {
            "completed": True,
            "goal": task_goal,
            "steps": steps_executed,
            "elapsed_sec": round(time.time() - started_at, 3),
        }

