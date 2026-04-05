"""Diagnosis-task verification."""

from __future__ import annotations

from merl_env.core.sample import TaskSample
from merl_env.verifiers.base import VerificationResult


def _normalize(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


class DiagnosisVerifier:
    """Exact-match diagnosis verifier with normalized whitespace and case."""

    def verify(self, sample: TaskSample, parsed_answer: dict[str, object]) -> VerificationResult:
        predicted = _normalize(str(parsed_answer.get("primary_diagnosis") or ""))
        expected = _normalize(str(sample.reference.get("primary_diagnosis") or ""))
        passed = bool(predicted and predicted == expected)
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reward=1.0 if passed else 0.0,
            details={
                "expected_primary_diagnosis": sample.reference.get("primary_diagnosis"),
                "predicted_primary_diagnosis": parsed_answer.get("primary_diagnosis"),
                "match_type": "normalized_exact",
            },
        )


__all__ = ["DiagnosisVerifier"]
