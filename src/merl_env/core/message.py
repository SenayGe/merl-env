"""Message models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from merl_env.core.tool_call import ToolCall


@dataclass(slots=True, kw_only=True)
class Message:
    """Message exchanged with a model or tool."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

