"""Local ICD dictionary lookup tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from merl_env.tools.base import Tool, ToolExecutionError


class IcdLookupTool(Tool):
    """Lookup ICD codes from a local dictionary artifact."""

    def __init__(
        self,
        *,
        entries: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
        artifact_path: str | Path | None = None,
    ) -> None:
        self._entries = _normalize_entries(entries, artifact_path=artifact_path)

    @property
    def name(self) -> str:
        return "icd_lookup"

    @property
    def description(self) -> str:
        return "Search a local ICD dictionary by exact code or free-text query."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "additionalProperties": False,
        }

    def run(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        code = str(arguments.get("code") or "").strip().upper()
        query = str(arguments.get("query") or "").strip().lower()
        limit = int(arguments.get("limit", 5) or 5)
        if not code and not query:
            raise ToolExecutionError("either 'code' or 'query' must be provided")

        if code:
            match = self._entries.get(code)
            return {
                "query": {"code": code},
                "results": ([match] if match is not None else []),
            }

        results: list[dict[str, Any]] = []
        for candidate in self._entries.values():
            haystack = " ".join(
                [
                    str(candidate.get("code") or ""),
                    str(candidate.get("short_title") or ""),
                    str(candidate.get("long_title") or ""),
                ]
            ).lower()
            if query in haystack:
                results.append(dict(candidate))
            if len(results) >= limit:
                break
        return {"query": {"query": query, "limit": limit}, "results": results}


def _normalize_entries(
    entries: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    *,
    artifact_path: str | Path | None,
) -> dict[str, dict[str, Any]]:
    loaded_entries: object = entries
    if artifact_path is not None:
        loaded_entries = _load_entries(Path(artifact_path))
    if loaded_entries is None:
        return {}
    if isinstance(loaded_entries, Mapping):
        normalized: dict[str, dict[str, Any]] = {}
        for code, payload in loaded_entries.items():
            normalized[str(code).strip().upper()] = {"code": str(code).strip().upper(), **dict(payload)}
        return normalized
    normalized = {}
    for item in loaded_entries:
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("code") or item.get("icd_code") or "").strip().upper()
        if not code:
            continue
        normalized[code] = {
            "code": code,
            "icd_version": item.get("icd_version"),
            "short_title": item.get("short_title"),
            "long_title": item.get("long_title") or item.get("description"),
        }
    return normalized


def _load_entries(path: Path) -> object:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["IcdLookupTool"]
