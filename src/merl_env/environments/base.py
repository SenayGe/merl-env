"""Abstract environment and model contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from merl_env.core.message import Message
from merl_env.core.result import EvalResult
from merl_env.core.registry import Registry
from merl_env.core.sample import TaskSample
from merl_env.core.task import TaskSpec
from merl_env.core.tool_call import ToolCall
from merl_env.parsers import TaggedJsonFinalAnswerParser, ToolCallParser
from merl_env.tools import ToolExecutor, ToolRegistry
from merl_env.verifiers import VerificationResult


@dataclass(slots=True, kw_only=True)
class ModelResponse:
    """Normalized model output returned by a model adapter."""

    text: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    raw: Any | None = None
    stop_reason: str | None = None


@runtime_checkable
class ModelAdapter(Protocol):
    """Provider-agnostic interface for model invocation."""

    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> ModelResponse:
        """Generate a model response from messages and optional tool specs."""


class BaseEnvironment(ABC):
    """Abstract environment interface."""

    @abstractmethod
    def run(self, sample: TaskSample, model: ModelAdapter) -> EvalResult:
        """Execute a sample against a model and return a normalized result."""


class RuntimeEnvironment(BaseEnvironment, ABC):
    """Shared helpers for task-driven runtime environments."""

    def __init__(
        self,
        task_specs: Sequence[TaskSpec] | Mapping[str, TaskSpec],
        *,
        max_new_tokens: int | None = None,
        tool_parser: ToolCallParser | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._task_registry = Registry[TaskSpec]()
        if isinstance(task_specs, Mapping):
            specs = task_specs.values()
        else:
            specs = task_specs
        for task_spec in specs:
            self._task_registry.register(task_spec.name, task_spec)
        self._max_new_tokens = max_new_tokens
        self._tool_parser = tool_parser or ToolCallParser()
        self._tool_registry = tool_registry
        self._tool_executor = ToolExecutor(tool_registry) if tool_registry is not None else None

    def get_task_spec(self, sample: TaskSample) -> TaskSpec:
        return self._task_registry.require(sample.task_name)

    def get_allowed_tool_names(self, task_spec: TaskSpec, sample: TaskSample) -> tuple[str, ...]:
        task_allowed = tuple(task_spec.allowed_tools)
        if sample.allowed_tools:
            if task_allowed:
                return tuple(name for name in sample.allowed_tools if name in set(task_allowed))
            return tuple(sample.allowed_tools)
        return task_allowed

    def get_model_tools(self, task_spec: TaskSpec, sample: TaskSample) -> list[dict[str, Any]]:
        if self._tool_registry is None:
            return []
        allowed_names = self.get_allowed_tool_names(task_spec, sample)
        return self._tool_registry.model_tools(allowed_names)

    def parse_final_answer(self, task_spec: TaskSpec, text: str):
        parser = task_spec.parser or TaggedJsonFinalAnswerParser(schema_name=task_spec.answer_schema_name)
        return parser.parse(text)

    def parse_tool_calls(self, response: ModelResponse):
        return self._tool_parser.parse(
            response.text,
            structured_tool_calls=response.tool_calls,
        )

    def verify(self, task_spec: TaskSpec, sample: TaskSample, parsed_answer: dict[str, Any]) -> VerificationResult:
        verifier = task_spec.verifier
        if verifier is None:
            return VerificationResult(
                passed=True,
                score=1.0,
                reward=1.0,
                details={"status": "unverified"},
            )
        return verifier.verify(sample, parsed_answer)

    def build_assistant_message(self, response: ModelResponse) -> Message:
        return Message(
            role="assistant",
            content=response.text,
            tool_calls=response.tool_calls,
            metadata={"stop_reason": response.stop_reason},
        )
