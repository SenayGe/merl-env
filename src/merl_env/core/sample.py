"""Task sample models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True, kw_only=True)
class TaskSample:
    """Serializable representation of one task sample."""

    sample_id: str
    task_name: str
    split: Literal["train", "val", "test"]
    input_payload: dict[str, Any]
    reference: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)

