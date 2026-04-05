"""ICD-task verification."""

from __future__ import annotations

from merl_env.core.sample import TaskSample
from merl_env.verifiers.base import VerificationResult


def _normalize_codes(codes: object) -> list[str]:
    if not isinstance(codes, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for code in codes:
        text = str(code).strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


class IcdVerifier:
    """Set-based verifier for ICD code prediction."""

    def verify(self, sample: TaskSample, parsed_answer: dict[str, object]) -> VerificationResult:
        predicted = set(_normalize_codes(parsed_answer.get("icd_codes")))
        expected = set(_normalize_codes(sample.reference.get("icd_codes")))
        overlap = len(predicted & expected)
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(expected) if expected else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
        passed = predicted == expected and bool(expected)
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else f1,
            reward=1.0 if passed else f1,
            details={
                "expected_codes": sorted(expected),
                "predicted_codes": sorted(predicted),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        )


__all__ = ["IcdVerifier"]
