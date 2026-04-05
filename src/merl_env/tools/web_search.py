"""Normalized web-search tool with Exa and Tavily backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Mapping, Sequence
from urllib import request

from merl_env.tools.base import Tool, ToolExecutionError


@dataclass(slots=True, kw_only=True)
class WebSearchResult:
    """One normalized search result."""

    title: str
    url: str
    snippet: str | None = None
    score: float | None = None


class WebSearchBackend(ABC):
    """Backend contract for web search providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    def search(self, query: str, *, num_results: int = 5) -> list[WebSearchResult]:
        """Run a search query."""


class InMemoryWebSearchBackend(WebSearchBackend):
    """Deterministic backend used in tests."""

    def __init__(self, fixtures: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> None:
        self._fixtures = dict(fixtures or {})

    @property
    def provider_name(self) -> str:
        return "in_memory"

    def search(self, query: str, *, num_results: int = 5) -> list[WebSearchResult]:
        rows = self._fixtures.get(query, ())[:num_results]
        return [
            WebSearchResult(
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                snippet=str(row.get("snippet") or "") or None,
                score=float(row["score"]) if row.get("score") is not None else None,
            )
            for row in rows
        ]


class ExaWebSearchBackend(WebSearchBackend):
    """Minimal Exa backend using the public HTTP API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("EXA_API_KEY")

    @property
    def provider_name(self) -> str:
        return "exa"

    def search(self, query: str, *, num_results: int = 5) -> list[WebSearchResult]:
        if not self._api_key:
            raise ToolExecutionError("EXA_API_KEY is not configured")
        payload = _post_json(
            "https://api.exa.ai/search",
            {
                "query": query,
                "numResults": num_results,
            },
            headers={"x-api-key": self._api_key},
        )
        results = payload.get("results", [])
        return [
            WebSearchResult(
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                snippet=str(row.get("text") or "") or None,
                score=float(row["score"]) if row.get("score") is not None else None,
            )
            for row in results
        ]


class TavilyWebSearchBackend(WebSearchBackend):
    """Minimal Tavily backend using the public HTTP API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")

    @property
    def provider_name(self) -> str:
        return "tavily"

    def search(self, query: str, *, num_results: int = 5) -> list[WebSearchResult]:
        if not self._api_key:
            raise ToolExecutionError("TAVILY_API_KEY is not configured")
        payload = _post_json(
            "https://api.tavily.com/search",
            {
                "api_key": self._api_key,
                "query": query,
                "max_results": num_results,
                "search_depth": "basic",
            },
        )
        results = payload.get("results", [])
        return [
            WebSearchResult(
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                snippet=str(row.get("content") or "") or None,
                score=float(row["score"]) if row.get("score") is not None else None,
            )
            for row in results
        ]


class WebSearchTool(Tool):
    """Normalized web-search tool."""

    def __init__(self, backend: WebSearchBackend) -> None:
        self._backend = backend

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web and return normalized results from the configured provider."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer"},
            },
            "additionalProperties": False,
        }

    def run(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"]).strip()
        num_results = int(arguments.get("num_results", 5) or 5)
        results = self._backend.search(query, num_results=num_results)
        return {
            "provider": self._backend.provider_name,
            "query": query,
            "results": [asdict(result) for result in results],
        }


def _post_json(url: str, payload: Mapping[str, Any], *, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(dict(payload)).encode("utf-8")
    request_headers = {"content-type": "application/json"}
    request_headers.update(dict(headers or {}))
    req = request.Request(url, data=body, headers=request_headers, method="POST")
    try:
        with request.urlopen(req) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - exercised only with live provider credentials
        raise ToolExecutionError(f"web search request failed: {exc}") from exc


__all__ = [
    "ExaWebSearchBackend",
    "InMemoryWebSearchBackend",
    "TavilyWebSearchBackend",
    "WebSearchBackend",
    "WebSearchResult",
    "WebSearchTool",
]
