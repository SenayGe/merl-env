"""Tool call models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, kw_only=True)
class ToolCall:
    """Normalized representation of a tool request."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    raw_text: str | None = None

