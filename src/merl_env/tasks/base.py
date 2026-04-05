"""Shared task-spec helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from merl_env.core.message import Message
from merl_env.core.sample import TaskSample
from merl_env.core.task import TaskSpec
from merl_env.parsers.final_answer import TaggedJsonFinalAnswerParser
from merl_env.prompts import load_prompt_template


class PromptTaskSpec(TaskSpec, ABC):
    """Task spec backed by a package prompt template."""

    def __init__(self) -> None:
        self._prompt_template = load_prompt_template(self.prompt_template_path or "")
        self._parser = TaggedJsonFinalAnswerParser(schema_name=self.answer_schema_name)
        self._verifier = self.build_verifier()

    @property
    def parser(self) -> TaggedJsonFinalAnswerParser:
        return self._parser

    @property
    def verifier(self) -> Any | None:
        return self._verifier

    def build_messages(self, sample: TaskSample) -> list[Message]:
        if sample.task_name != self.name:
            raise ValueError(
                f"Task spec {self.name!r} cannot render sample for task {sample.task_name!r}"
            )
        return [
            Message(role="system", content=self._build_system_prompt(sample)),
            Message(role="user", content=self.render_prompt(sample)),
        ]

    def render_prompt(self, sample: TaskSample) -> str:
        return self._prompt_template.format(**self.build_prompt_context(sample))

    def _build_system_prompt(self, sample: TaskSample) -> str:
        allowed_tools = sample.allowed_tools or self.allowed_tools
        tool_guidance = (
            f"Allowed tools for this sample: {', '.join(allowed_tools)}. "
            "Use structured tool calls when available, or <tool_call>{json}</tool_call> in text fallback."
            if allowed_tools
            else "No tool use is allowed for this sample."
        )
        return (
            f"You are solving the '{self.name}' clinical task. "
            "Think through the case, but only expose the final answer in the required tagged JSON format. "
            f"{tool_guidance}"
        )

    @abstractmethod
    def build_prompt_context(self, sample: TaskSample) -> dict[str, Any]:
        """Return template variables for one sample."""

    @abstractmethod
    def build_verifier(self) -> Any | None:
        """Construct the verifier attached to this task."""


__all__ = ["PromptTaskSpec"]
