"""Small JSON-schema-style validator used by parsers and tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(slots=True, kw_only=True)
class SchemaValidationError:
    """One schema validation failure."""

    path: str
    message: str


def validate_schema(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str = "$",
) -> list[SchemaValidationError]:
    """Validate a value against a small subset of JSON Schema."""

    errors: list[SchemaValidationError] = []
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, str(expected_type)):
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected type {expected_type!r}, got {type(value).__name__!r}",
            )
        )
        return errors

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"value must be one of {list(enum_values)!r}",
            )
        )
        return errors

    if isinstance(value, dict):
        errors.extend(_validate_object(value, schema, path=path))
    elif isinstance(value, list):
        errors.extend(_validate_array(value, schema, path=path))
    return errors


def first_error_message(errors: Sequence[SchemaValidationError]) -> str | None:
    """Return the first validation error in human-readable form."""

    if not errors:
        return None
    first = errors[0]
    return f"{first.path}: {first.message}"


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _validate_object(
    value: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    path: str,
) -> list[SchemaValidationError]:
    errors: list[SchemaValidationError] = []
    required = schema.get("required", ())
    properties = schema.get("properties", {})
    additional_properties = schema.get("additionalProperties", True)

    for field_name in required:
        if field_name not in value:
            errors.append(
                SchemaValidationError(
                    path=f"{path}.{field_name}",
                    message="missing required field",
                )
            )

    for field_name, field_value in value.items():
        field_path = f"{path}.{field_name}"
        if field_name in properties:
            errors.extend(validate_schema(field_value, properties[field_name], path=field_path))
            continue
        if additional_properties is False:
            errors.append(
                SchemaValidationError(
                    path=field_path,
                    message="unexpected field",
                )
            )
            continue
        if isinstance(additional_properties, dict):
            errors.extend(validate_schema(field_value, additional_properties, path=field_path))
    return errors


def _validate_array(
    value: Sequence[Any],
    schema: Mapping[str, Any],
    *,
    path: str,
) -> list[SchemaValidationError]:
    errors: list[SchemaValidationError] = []
    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    item_schema = schema.get("items")

    if isinstance(min_items, int) and len(value) < min_items:
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected at least {min_items} item(s)",
            )
        )
    if isinstance(max_items, int) and len(value) > max_items:
        errors.append(
            SchemaValidationError(
                path=path,
                message=f"expected at most {max_items} item(s)",
            )
        )
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            errors.extend(validate_schema(item, item_schema, path=f"{path}[{index}]"))
    return errors


__all__ = ["SchemaValidationError", "first_error_message", "validate_schema"]
