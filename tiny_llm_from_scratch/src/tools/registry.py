from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
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
    """Type-safe registry of tools with argument validation and permission controls."""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.execution_log: List[Dict[str, Any]] = []

    def register(self, tool: ToolDefinition):
        self.tools[tool.name] = tool

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
        ]

    def execute(self, name: str, arguments: Dict[str, Any], user_approved: bool = False) -> Dict[str, Any]:
        """Validates parameters, checks permissions, and executes tool safely."""
        tool = self.get_tool(name)
        if not tool:
            return {"success": False, "error": f"Tool '{name}' is not registered."}

        if tool.requires_approval and not user_approved:
            return {
                "success": False,
                "error": f"Action requires explicit user approval before execution.",
                "pending_approval": True,
            }

        # Parameter validation
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

