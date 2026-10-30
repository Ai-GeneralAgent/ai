from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
import time


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]  # Parameter name -> type description
    execute_fn: Callable[..., Any]
    requires_approval: bool = False
    timeout_sec: float = 5.0


class ToolRegistry:
    """Type-safe registry of tools with argument validation, strict allowlist, and permission controls."""

    def __init__(self, allowlist: Optional[Iterable[str]] = None):
        self.tools: Dict[str, ToolDefinition] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.allowlist: Optional[Set[str]] = set(allowlist) if allowlist is not None else None

    def register(self, tool: ToolDefinition):
        self.tools[tool.name] = tool

    def set_allowlist(self, allowed_tools: Optional[Iterable[str]]):
        """Sets or updates the active tool allowlist (principle of least privilege)."""
        self.allowlist = set(allowed_tools) if allowed_tools is not None else None

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "requires_approval": t.requires_approval,
            }
            for t in self.tools.values()
            if self.allowlist is None or t.name in self.allowlist
        ]

    def execute(self, name: str, arguments: Dict[str, Any], user_approved: bool = False) -> Dict[str, Any]:
        """Validates allowlist, parameters, checks permissions, and executes tool safely."""
        # 1. Enforce strict allowlist policy
        if self.allowlist is not None and name not in self.allowlist:
            log_entry = {
                "tool": name,
                "arguments": arguments,
                "success": False,
                "error": f"Tool '{name}' is not permitted by active allowlist policy.",
                "elapsed_sec": 0.0,
            }
            self.execution_log.append(log_entry)
            return {
                "success": False,
                "error": f"Tool '{name}' is blocked by active tool allowlist policy.",
                "allowlist_violation": True,
            }

        # 2. Check if tool is registered
        tool = self.get_tool(name)
        if not tool:
            return {"success": False, "error": f"Tool '{name}' is not registered."}

        # 3. Check explicit approval requirements
        if tool.requires_approval and not user_approved:
            return {
                "success": False,
                "error": f"Action requires explicit user approval before execution.",
                "pending_approval": True,
            }

        # 4. Parameter validation
        if not isinstance(arguments, dict):
            return {"success": False, "error": "Arguments must be provided as a dictionary."}

        for param, param_type in tool.parameters.items():
            if param not in arguments:
                return {"success": False, "error": f"Missing required parameter '{param}'"}

        start_time = time.time()
        try:
            result = tool.execute_fn(**arguments)
            elapsed = time.time() - start_time
            log_entry = {
                "tool": name,
                "arguments": arguments,
                "success": True,
                "elapsed_sec": round(elapsed, 4),
            }
            self.execution_log.append(log_entry)
            return {"success": True, "result": result, "elapsed_sec": elapsed}
        except Exception as e:
            elapsed = time.time() - start_time
            log_entry = {
                "tool": name,
                "arguments": arguments,
                "success": False,
                "error": str(e),
                "elapsed_sec": round(elapsed, 4),
            }
            self.execution_log.append(log_entry)
            return {"success": False, "error": f"Tool execution failed: {str(e)}"}


