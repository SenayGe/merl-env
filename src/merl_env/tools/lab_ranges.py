"""Static reference tool for common lab ranges."""

from __future__ import annotations

from typing import Any, Mapping

from merl_env.tools.base import Tool, ToolExecutionError

DEFAULT_LAB_RANGES: dict[str, dict[str, Any]] = {
    "Potassium": {"units": "mmol/L", "normal_range": "3.5-5.1", "critical_low": 2.8, "critical_high": 6.0},
    "Sodium": {"units": "mmol/L", "normal_range": "135-145", "critical_low": 120.0, "critical_high": 160.0},
    "Glucose": {"units": "mg/dL", "normal_range": "70-140", "critical_low": 50.0, "critical_high": 400.0},
    "Lactate": {"units": "mmol/L", "normal_range": "0.5-2.0", "critical_high": 4.0},
    "Hemoglobin": {"units": "g/dL", "normal_range": "12.0-16.0", "critical_low": 7.0},
}


class LabRangesTool(Tool):
    """Lookup static reference ranges for supported laboratory tests."""

    def __init__(self, ranges: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self._ranges = {str(name): dict(values) for name, values in dict(ranges or DEFAULT_LAB_RANGES).items()}

    @property
    def name(self) -> str:
        return "lab_ranges"

    @property
    def description(self) -> str:
        return "Lookup normal and critical reference ranges for common lab tests."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["lab_name"],
            "properties": {
                "lab_name": {"type": "string"},
            },
            "additionalProperties": False,
        }

    def run(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        lab_name = str(arguments["lab_name"]).strip()
        if lab_name not in self._ranges:
            raise ToolExecutionError(f"no reference range found for lab {lab_name!r}")
        return {"lab_name": lab_name, **self._ranges[lab_name]}


__all__ = ["DEFAULT_LAB_RANGES", "LabRangesTool"]
