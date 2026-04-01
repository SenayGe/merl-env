"""Internal helpers shared by offline task builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
import math
from typing import Any

from merl_env.core.sample import TaskSample
from merl_env.data.splits import SPLIT_NAMES


def empty_split_buckets() -> dict[str, list[TaskSample]]:
    """Return a canonical split -> samples mapping."""

    return {split: [] for split in SPLIT_NAMES}


def sort_split_buckets(samples_by_split: dict[str, list[TaskSample]]) -> dict[str, list[TaskSample]]:
    """Sort sample lists in-place for deterministic artifact output."""

    for samples in samples_by_split.values():
        samples.sort(key=lambda sample: sample.sample_id)
    return samples_by_split


def build_sample_id(task_name: str, *parts: object) -> str:
    """Compose a deterministic sample identifier from non-empty parts."""

    normalized_parts = [str(part).strip() for part in parts if str(part).strip()]
    if not normalized_parts:
        return task_name
    return "-".join((task_name, *normalized_parts))


def coerce_bool(value: Any) -> bool:
    """Normalize common truthy and falsy values."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", "", "none", "null"}:
        return False
    return bool(text)


def coerce_int(value: Any) -> int | None:
    """Convert a value to int when possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    if isinstance(value, Decimal):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def coerce_float(value: Any) -> float | None:
    """Convert a value to float when possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_datetime(value: Any) -> datetime | None:
    """Parse a datetime-ish value emitted by fixtures, pandas, or BigQuery."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            converted = value.to_pydatetime()
        except TypeError:
            converted = None
        if isinstance(converted, datetime):
            return converted
    if isinstance(value, date):
        return datetime.combine(value, time.min)

    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        if len(normalized) == 10:
            try:
                return datetime.combine(date.fromisoformat(normalized), time.min)
            except ValueError:
                return None
        return None


def coerce_str_list(value: Any) -> list[str]:
    """Normalize sequence-like values into a list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def json_safe(value: Any) -> Any:
    """Recursively normalize values into JSON-serializable primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except ValueError:
            pass
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    return str(value)
