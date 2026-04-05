"""Registry wrapper for runtime tools."""

from __future__ import annotations

from typing import Iterable, Sequence

from merl_env.core.registry import Registry
from merl_env.tools.base import Tool


class ToolRegistry:
    """Typed registry for runtime tools."""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._registry: Registry[Tool] = Registry()
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool, *, overwrite: bool = False) -> None:
        self._registry.register(tool.name, tool, overwrite=overwrite)

    def get(self, name: str) -> Tool | None:
        return self._registry.get(name)

    def require(self, name: str) -> Tool:
        return self._registry.require(name)

    def resolve_many(self, names: Sequence[str]) -> list[Tool]:
        resolved: list[Tool] = []
        for name in names:
            tool = self.get(name)
            if tool is not None:
                resolved.append(tool)
        return resolved

    def model_tools(self, names: Sequence[str]) -> list[dict[str, object]]:
        return [tool.as_model_tool() for tool in self.resolve_many(names)]

    def list_names(self) -> list[str]:
        return self._registry.list_names()


__all__ = ["ToolRegistry"]
