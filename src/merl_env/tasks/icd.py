"""ICD task spec."""

from __future__ import annotations

from merl_env.core.sample import TaskSample
from merl_env.tasks.base import PromptTaskSpec
from merl_env.verifiers import IcdVerifier


class IcdTaskSpec(PromptTaskSpec):
    """Prompted ICD coding task."""

    @property
    def name(self) -> str:
        return "icd"

    @property
    def prompt_template_path(self) -> str:
        return "icd.txt"

    @property
    def answer_schema_name(self) -> str:
        return "icd"

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return ("icd_lookup",)

    def build_prompt_context(self, sample: TaskSample) -> dict[str, object]:
        return {"note_text": str(sample.input_payload.get("note_text") or "").strip()}

    def build_verifier(self) -> IcdVerifier:
        return IcdVerifier()


__all__ = ["IcdTaskSpec"]
