"""Evaluation result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from merl_env.core.message import Message


@dataclass(slots=True, kw_only=True)
class EvalTrace:
    """Trace metadata collected while solving one sample."""

    messages: tuple[Message, ...]
    tool_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    model_outputs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True, kw_only=True)
class EvalResult:
    """Normalized output of an environment run."""

    sample_id: str
    task_name: str
    passed: bool
    score: float | None = None
    reward: float | None = None
    parsed_answer: dict[str, Any] | None = None
    raw_output: str | None = None
    tool_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    verification: dict[str, Any] | None = None
    trace: EvalTrace | None = None
    error: str | None = None
    stop_reason: str | None = None
