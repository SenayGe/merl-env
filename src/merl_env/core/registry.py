"""Generic string-key registry."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Small generic registry with explicit duplicate protection."""

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T, *, overwrite: bool = False) -> None:
        if not overwrite and name in self._items:
            raise KeyError(f"{name!r} is already registered")
        self._items[name] = item

    def get(self, name: str) -> T | None:
        return self._items.get(name)

    def require(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"{name!r} is not registered")
        return self._items[name]

    def list_names(self) -> list[str]:
        return sorted(self._items)
