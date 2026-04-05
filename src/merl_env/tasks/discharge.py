"""Discharge task spec."""

from __future__ import annotations

import json
from typing import Any

from merl_env.core.sample import TaskSample
from merl_env.tasks.base import PromptTaskSpec
from merl_env.verifiers import DischargeVerifier


class DischargeTaskSpec(PromptTaskSpec):
    """Prompted discharge safety task."""

    @property
    def name(self) -> str:
        return "discharge"

    @property
    def prompt_template_path(self) -> str:
        return "discharge.txt"

    @property
    def answer_schema_name(self) -> str:
        return "discharge"

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return ("lab_ranges",)

    def build_prompt_context(self, sample: TaskSample) -> dict[str, Any]:
        payload = sample.input_payload
        return {
            "snapshot_time": payload.get("snapshot_time"),
            "admittime": payload.get("admittime"),
            "dischtime": payload.get("dischtime"),
            "age_at_admission": payload.get("age_at_admission"),
            "gender": payload.get("gender"),
            "admission_type": payload.get("admission_type"),
            "admission_location": payload.get("admission_location"),
            "discharge_location": payload.get("discharge_location"),
            "icu_t": payload.get("icu_t"),
            "vitals_block": _format_named_series(payload.get("vitals", {})),
            "labs_block": _format_named_series(payload.get("labs", {})),
            "diagnoses_block": _format_diagnoses(payload.get("diagnoses", [])),
            "charlson_block": _format_optional_json(payload.get("charlson")),
            "sofa_block": _format_optional_json(payload.get("sofa")),
        }

    def build_verifier(self) -> DischargeVerifier:
        return DischargeVerifier()


def _format_named_series(series_map: object) -> str:
    if not isinstance(series_map, dict) or not series_map:
        return "- none"
    lines: list[str] = []
    for name in sorted(series_map):
        values = series_map.get(name)
        if not isinstance(values, list) or not values:
            lines.append(f"- {name}: none")
            continue
        rendered = []
        for row in values:
            if not isinstance(row, dict):
                continue
            rendered.append(f"{row.get('value')} at {row.get('time')}")
        lines.append(f"- {name}: {', '.join(rendered) if rendered else 'none'}")
    return "\n".join(lines)


def _format_diagnoses(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "- none"
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = row.get("icd_code")
        title = row.get("long_title")
        seq_num = row.get("seq_num")
        lines.append(f"- seq {seq_num}: {code} {title}".strip())
    return "\n".join(lines) if lines else "- none"


def _format_optional_json(value: object) -> str:
    if value in (None, "", [], {}):
        return "- none"
    return json.dumps(value, indent=2, sort_keys=True)


__all__ = ["DischargeTaskSpec"]
