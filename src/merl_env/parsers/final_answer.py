"""Tagged final-answer parsing and schema validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from merl_env.utils import first_error_message, validate_schema

FINAL_ANSWER_PATTERN = re.compile(
    r"<final_answer>\s*(?P<body>.*?)\s*</final_answer>",
    re.IGNORECASE | re.DOTALL,
)

ANSWER_SCHEMAS: dict[str, dict[str, Any]] = {
    "diagnosis": {
        "type": "object",
        "required": ["primary_diagnosis"],
        "properties": {
            "primary_diagnosis": {"type": "string"},
        },
        "additionalProperties": False,
    },
    "discharge": {
        "type": "object",
        "required": ["safe_for_discharge_24h", "has_hard_barrier", "rationale"],
        "properties": {
            "safe_for_discharge_24h": {"type": "boolean"},
            "has_hard_barrier": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "additionalProperties": False,
    },
    "icd": {
        "type": "object",
        "required": ["icd_codes"],
        "properties": {
            "icd_codes": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "additionalProperties": False,
    },
}


@dataclass(slots=True, kw_only=True)
class ParseError:
    """Structured parse failure."""

    code: str
    message: str
    raw_fragment: str | None = None


@dataclass(slots=True, kw_only=True)
class FinalAnswerParseResult:
    """Normalized final-answer parse result."""

    success: bool
    answer: dict[str, Any] | None = None
    error: ParseError | None = None
    raw_fragment: str | None = None


class TaggedJsonFinalAnswerParser:
    """Parse `<final_answer>{json}</final_answer>` outputs."""

    def __init__(self, *, schema_name: str) -> None:
        if schema_name not in ANSWER_SCHEMAS:
            raise KeyError(f"Unknown answer schema {schema_name!r}")
        self._schema_name = schema_name
        self._schema = ANSWER_SCHEMAS[schema_name]

    @property
    def schema_name(self) -> str:
        return self._schema_name

    def parse(self, text: str) -> FinalAnswerParseResult:
        matches = list(FINAL_ANSWER_PATTERN.finditer(text or ""))
        if not matches:
            if "<final_answer>" in (text or ""):
                return FinalAnswerParseResult(
                    success=False,
                    error=ParseError(
                        code="partial_final_answer",
                        message="final answer tag was opened but not closed",
                    ),
                )
            return FinalAnswerParseResult(
                success=False,
                error=ParseError(
                    code="missing_final_answer",
                    message="no <final_answer> block found in model output",
                ),
            )

        raw_fragment = matches[-1].group("body")
        try:
            parsed = json.loads(raw_fragment)
        except json.JSONDecodeError as exc:
            return FinalAnswerParseResult(
                success=False,
                raw_fragment=raw_fragment,
                error=ParseError(
                    code="invalid_final_answer_json",
                    message=f"invalid JSON inside final answer block: {exc.msg}",
                    raw_fragment=raw_fragment,
                ),
            )
        if not isinstance(parsed, dict):
            return FinalAnswerParseResult(
                success=False,
                raw_fragment=raw_fragment,
                error=ParseError(
                    code="final_answer_not_object",
                    message="final answer JSON must decode to an object",
                    raw_fragment=raw_fragment,
                ),
            )

        errors = validate_schema(parsed, self._schema)
        if errors:
            return FinalAnswerParseResult(
                success=False,
                raw_fragment=raw_fragment,
                error=ParseError(
                    code="final_answer_schema_error",
                    message=first_error_message(errors) or "final answer schema validation failed",
                    raw_fragment=raw_fragment,
                ),
            )
        return FinalAnswerParseResult(
            success=True,
            answer=parsed,
            raw_fragment=raw_fragment,
        )


__all__ = [
    "ANSWER_SCHEMAS",
    "FINAL_ANSWER_PATTERN",
    "FinalAnswerParseResult",
    "ParseError",
    "TaggedJsonFinalAnswerParser",
]
