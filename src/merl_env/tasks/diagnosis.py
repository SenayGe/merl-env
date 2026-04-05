"""Diagnosis task spec."""

from __future__ import annotations

from typing import Any

from merl_env.core.sample import TaskSample
from merl_env.tasks.base import PromptTaskSpec
from merl_env.verifiers import DiagnosisVerifier


class DiagnosisTaskSpec(PromptTaskSpec):
    """Prompted diagnosis task."""

    @property
    def name(self) -> str:
        return "diagnosis"

    @property
    def prompt_template_path(self) -> str:
        return "diagnosis.txt"

    @property
    def answer_schema_name(self) -> str:
        return "diagnosis"

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return ()

    def build_prompt_context(self, sample: TaskSample) -> dict[str, Any]:
        payload = sample.input_payload
        return {
            "clinical_summary": str(payload.get("clinical_summary") or "").strip(),
        }

    def build_verifier(self) -> DiagnosisVerifier:
        return DiagnosisVerifier()


__all__ = ["DiagnosisTaskSpec"]
