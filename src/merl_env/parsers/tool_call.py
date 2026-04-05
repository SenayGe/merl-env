"""Tool-call normalization from structured or tagged outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from merl_env.core.tool_call import ToolCall
from merl_env.parsers.final_answer import ParseError

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True, kw_only=True)
class ToolCallParseResult:
    """Normalized tool-call parse result."""

    success: bool
    tool_calls: tuple[ToolCall, ...]
    error: ParseError | None = None


class ToolCallParser:
    """Prefer structured tool calls and fall back to tagged JSON blocks."""

    def parse(
        self,
        text: str,
        *,
        structured_tool_calls: Sequence[ToolCall | Mapping[str, Any]] | None = None,
    ) -> ToolCallParseResult:
        if structured_tool_calls:
            normalized = self._normalize_structured_calls(structured_tool_calls)
            if isinstance(normalized, ParseError):
                return ToolCallParseResult(success=False, tool_calls=(), error=normalized)
            return ToolCallParseResult(success=True, tool_calls=tuple(normalized))

        matches = list(TOOL_CALL_PATTERN.finditer(text or ""))
        if not matches:
            if "<tool_call>" in (text or ""):
                return ToolCallParseResult(
                    success=False,
                    tool_calls=(),
                    error=ParseError(
                        code="partial_tool_call",
                        message="tool call tag was opened but not closed",
                    ),
                )
            return ToolCallParseResult(success=True, tool_calls=())

        tool_calls: list[ToolCall] = []
        for index, match in enumerate(matches, start=1):
            raw_fragment = match.group("body")
            try:
                parsed = json.loads(raw_fragment)
            except json.JSONDecodeError as exc:
                return ToolCallParseResult(
                    success=False,
                    tool_calls=(),
                    error=ParseError(
                        code="invalid_tool_call_json",
                        message=f"invalid JSON inside tool call block: {exc.msg}",
                        raw_fragment=raw_fragment,
                    ),
                )
            normalized = self._normalize_single_call(parsed, default_call_id=f"tool-call-{index}")
            if isinstance(normalized, ParseError):
                return ToolCallParseResult(success=False, tool_calls=(), error=normalized)
            tool_calls.append(normalized)
        return ToolCallParseResult(success=True, tool_calls=tuple(tool_calls))

    def _normalize_structured_calls(
        self,
        tool_calls: Sequence[ToolCall | Mapping[str, Any]],
    ) -> list[ToolCall] | ParseError:
        normalized: list[ToolCall] = []
        for index, tool_call in enumerate(tool_calls, start=1):
            if isinstance(tool_call, ToolCall):
                if not isinstance(tool_call.arguments, dict):
                    return ParseError(
                        code="invalid_structured_tool_call",
                        message="structured tool call arguments must be a JSON object",
                    )
                normalized.append(tool_call)
                continue
            normalized_call = self._normalize_single_call(
                tool_call,
                default_call_id=f"tool-call-{index}",
            )
            if isinstance(normalized_call, ParseError):
                return normalized_call
            normalized.append(normalized_call)
        return normalized

    def _normalize_single_call(
        self,
        raw_call: Mapping[str, Any] | Any,
        *,
        default_call_id: str,
    ) -> ToolCall | ParseError:
        if not isinstance(raw_call, Mapping):
            return ParseError(
                code="invalid_tool_call_shape",
                message="tool call payload must be a JSON object",
            )
        tool_name = str(raw_call.get("tool_name") or raw_call.get("name") or "").strip()
        if not tool_name:
            return ParseError(
                code="missing_tool_name",
                message="tool call payload is missing a tool_name",
            )
        arguments = raw_call.get("arguments", raw_call.get("args", {}))
        if not isinstance(arguments, dict):
            return ParseError(
                code="invalid_tool_arguments",
                message="tool call arguments must decode to a JSON object",
            )
        return ToolCall(
            call_id=str(raw_call.get("call_id") or raw_call.get("id") or default_call_id),
            tool_name=tool_name,
            arguments=dict(arguments),
            raw_text=json.dumps(dict(raw_call), sort_keys=True),
        )


__all__ = ["TOOL_CALL_PATTERN", "ToolCallParseResult", "ToolCallParser"]
