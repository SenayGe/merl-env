"""Discharge-task verification."""

from __future__ import annotations

from merl_env.core.sample import TaskSample
from merl_env.verifiers.base import VerificationResult


class DischargeVerifier:
    """Deterministic verifier for discharge safety predictions."""

    def verify(self, sample: TaskSample, parsed_answer: dict[str, object]) -> VerificationResult:
        safe_match = parsed_answer.get("safe_for_discharge_24h") == sample.reference.get(
            "safe_for_discharge_24h"
        )
        barrier_match = parsed_answer.get("has_hard_barrier") == sample.reference.get(
            "has_hard_barrier"
        )
        score = (float(safe_match) + float(barrier_match)) / 2.0
        return VerificationResult(
            passed=bool(safe_match and barrier_match),
            score=score,
            reward=score,
            details={
                "safe_for_discharge_match": safe_match,
                "hard_barrier_match": barrier_match,
                "expected": dict(sample.reference),
            },
        )


__all__ = ["DischargeVerifier"]
