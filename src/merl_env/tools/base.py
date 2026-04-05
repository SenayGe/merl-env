"""Base classes and result models for runtime tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Any, Mapping

from merl_env.core.message import Message
from merl_env.core.tool_call import ToolCall
from merl_env.utils import first_error_message, validate_schema


class ToolValidationError(ValueError):
    """Raised when tool-call arguments do not match the declared schema."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails during execution."""


@dataclass(slots=True, kw_only=True)
class ToolExecutionResult:
    """Normalized output of one executed tool call."""

    call_id: str
    tool_name: str
    ok: bool
    content: str
    payload: dict[str, Any] | None = None
    error: str | None = None

    def to_message(self) -> Message:
        return Message(
            role="tool",
            name=self.tool_name,
            tool_call_id=self.call_id,
            content=self.content,
            metadata={"ok": self.ok, "error": self.error},
        )

    def to_trace_event(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "ok": self.ok,
            "payload": self.payload,
            "error": self.error,
        }


class Tool(ABC):
    """Abstract runtime tool with JSON-schema-like argument validation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable tool description."""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON-schema-style description of accepted arguments."""

    @abstractmethod
    def run(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Execute the tool and return a normalized JSON-serializable payload."""

    def validate_arguments(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        errors = validate_schema(dict(arguments), self.input_schema)
        if errors:
            raise ToolValidationError(first_error_message(errors) or "tool arguments are invalid")
        return dict(arguments)

    def execute(self, tool_call: ToolCall) -> ToolExecutionResult:
        validated_arguments = self.validate_arguments(tool_call.arguments)
        payload = self.run(validated_arguments)
        return ToolExecutionResult(
            call_id=tool_call.call_id,
            tool_name=self.name,
            ok=True,
            payload=payload,
            content=json.dumps(payload, sort_keys=True),
        )

    def as_model_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


__all__ = [
    "Tool",
    "ToolExecutionError",
    "ToolExecutionResult",
    "ToolValidationError",
]
