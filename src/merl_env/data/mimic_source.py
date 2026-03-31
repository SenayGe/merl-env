"""Source abstractions for fetching MIMIC-derived rows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(slots=True, kw_only=True)
class MimicQuery:
    """Logical query definition used by offline task builders."""

    name: str
    sql: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class MimicSource(ABC):
    """Abstract row source for MIMIC-derived task building."""

    @abstractmethod
    def fetch_rows(self, query: MimicQuery) -> list[dict[str, Any]]:
        """Execute a logical query and return rows as dictionaries."""


RowsFactory = Callable[[MimicQuery], Sequence[Mapping[str, Any]]]


class InMemoryMimicSource(MimicSource):
    """Test-friendly source that returns pre-seeded rows by query name."""

    def __init__(
        self,
        fixtures: Mapping[str, Sequence[Mapping[str, Any]] | RowsFactory] | None = None,
    ) -> None:
        self._fixtures = dict(fixtures or {})

    def fetch_rows(self, query: MimicQuery) -> list[dict[str, Any]]:
        fixture = self._fixtures.get(query.name)
        if fixture is None:
            raise KeyError(f"No fixture registered for query {query.name!r}")
        if callable(fixture):
            rows = fixture(query)
        else:
            rows = fixture
        return [dict(row) for row in rows]

