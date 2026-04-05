"""Execution helpers for runtime tools."""

from __future__ import annotations

import json

from merl_env.core.tool_call import ToolCall
from merl_env.tools.base import ToolExecutionError, ToolExecutionResult, ToolValidationError
from merl_env.tools.registry import ToolRegistry


class ToolExecutor:
    """Execute normalized tool calls against a registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, tool_call: ToolCall) -> ToolExecutionResult:
        tool = self._registry.get(tool_call.tool_name)
        if tool is None:
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                ok=False,
                error=f"unknown tool {tool_call.tool_name!r}",
                content=json.dumps(
                    {"error": f"unknown tool {tool_call.tool_name!r}"},
                    sort_keys=True,
                ),
            )

        try:
            return tool.execute(tool_call)
        except ToolValidationError as exc:
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                ok=False,
                error=str(exc),
                content=json.dumps({"error": str(exc)}, sort_keys=True),
            )
        except ToolExecutionError as exc:
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                ok=False,
                error=str(exc),
                content=json.dumps({"error": str(exc)}, sort_keys=True),
            )
        except Exception as exc:
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                ok=False,
                error=f"unexpected tool failure: {exc}",
                content=json.dumps(
                    {"error": f"unexpected tool failure: {exc}"},
                    sort_keys=True,
                ),
            )


__all__ = ["ToolExecutor"]
