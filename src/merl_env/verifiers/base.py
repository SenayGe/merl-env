"""Verifier contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from merl_env.core.sample import TaskSample


@dataclass(slots=True, kw_only=True)
class VerificationResult:
    """Normalized verifier output."""

    passed: bool
    score: float | None = None
    reward: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "reward": self.reward,
            **self.details,
        }


@runtime_checkable
class Verifier(Protocol):
    """Protocol for task verification."""

    def verify(self, sample: TaskSample, parsed_answer: dict[str, Any]) -> VerificationResult:
        """Score a parsed answer against a sample reference."""


__all__ = ["VerificationResult", "Verifier"]
