"""Abstract task interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from merl_env.core.message import Message
from merl_env.core.sample import TaskSample


class TaskSpec(ABC):
    """Abstract interface for task-specific prompting and validation hooks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique task name."""

    @property
    def prompt_template_path(self) -> str | None:
        """Package-relative prompt template resource path."""
        return None

    @property
    @abstractmethod
    def answer_schema_name(self) -> str:
        """Name of the answer schema expected from this task."""

    @property
    @abstractmethod
    def allowed_tools(self) -> tuple[str, ...]:
        """Tools that may be used while solving this task."""

    @property
    def parser(self) -> Any | None:
        """Optional parser hook for later concrete task implementations."""
        return None

    @property
    def verifier(self) -> Any | None:
        """Optional verifier hook for later concrete task implementations."""
        return None

    @abstractmethod
    def build_messages(self, sample: TaskSample) -> list[Message]:
        """Render a sample into model-ready messages."""
