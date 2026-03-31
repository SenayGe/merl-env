"""Artifact schema and read/write helpers for local task tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from merl_env.core.sample import TaskSample
from merl_env.data.splits import SPLIT_NAMES

PARQUET_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "task_name",
    "split",
    "input_payload_json",
    "reference_json",
    "metadata_json",
    "allowed_tools_json",
)


@dataclass(slots=True, kw_only=True)
class SplitArtifactPaths:
    """Artifact paths for one split."""

    split: str
    jsonl_path: Path
    parquet_path: Path


@dataclass(slots=True, kw_only=True)
class TaskArtifactPaths:
    """Resolved artifact paths for one task."""

    task_name: str
    task_dir: Path
    manifest_path: Path
    split_paths: dict[str, SplitArtifactPaths]


@dataclass(slots=True, kw_only=True)
class TaskArtifactManifest:
    """Manifest describing one local task artifact bundle."""

    task_name: str
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    splits: tuple[str, ...] = SPLIT_NAMES
    split_counts: dict[str, int] = field(default_factory=dict)
    files: dict[str, dict[str, str]] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskArtifactManifest":
        return cls(
            task_name=str(raw["task_name"]),
            schema_version=int(raw.get("schema_version", 1)),
            created_at=str(raw.get("created_at", "")),
            splits=tuple(raw.get("splits", SPLIT_NAMES)),
            split_counts={str(k): int(v) for k, v in dict(raw.get("split_counts", {})).items()},
            files={
                str(split): {str(fmt): str(path) for fmt, path in dict(paths).items()}
                for split, paths in dict(raw.get("files", {})).items()
            },
            source=dict(raw.get("source", {})),
            metadata=dict(raw.get("metadata", {})),
        )


def build_task_artifact_paths(
    out_dir: str | Path,
    task_name: str,
    *,
    splits: Sequence[str] = SPLIT_NAMES,
) -> TaskArtifactPaths:
    """Build the canonical path layout for one task's artifact bundle."""

    task_dir = Path(out_dir) / task_name
    split_paths = {
        split: SplitArtifactPaths(
            split=split,
            jsonl_path=task_dir / f"{split}.jsonl",
            parquet_path=task_dir / f"{split}.parquet",
        )
        for split in splits
    }
    return TaskArtifactPaths(
        task_name=task_name,
        task_dir=task_dir,
        manifest_path=task_dir / "manifest.json",
        split_paths=split_paths,
    )


def load_task_manifest(manifest_path: str | Path) -> TaskArtifactManifest:
    """Load a task manifest from disk."""

    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return TaskArtifactManifest.from_dict(raw)


def write_task_artifacts(
    out_dir: str | Path,
    task_name: str,
    samples_by_split: Mapping[str, Sequence[TaskSample]],
    *,
    source: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TaskArtifactManifest:
    """Write jsonl, parquet, and manifest files for all canonical splits."""

    paths = build_task_artifact_paths(out_dir, task_name)
    paths.task_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}
    files: dict[str, dict[str, str]] = {}

    for split in SPLIT_NAMES:
        samples = list(samples_by_split.get(split, ()))
        _validate_samples(task_name, split, samples)

        split_paths = paths.split_paths[split]
        _write_split_jsonl(split_paths.jsonl_path, samples)
        _write_split_parquet(split_paths.parquet_path, samples)

        split_counts[split] = len(samples)
        files[split] = {
            "jsonl": split_paths.jsonl_path.name,
            "parquet": split_paths.parquet_path.name,
        }

    manifest = TaskArtifactManifest(
        task_name=task_name,
        split_counts=split_counts,
        files=files,
        source=dict(source or {}),
        metadata=dict(metadata or {}),
    )
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
    return manifest


def load_task_artifacts(
    out_dir: str | Path,
    task_name: str,
    *,
    format_preference: str = "parquet",
) -> dict[str, list[TaskSample]]:
    """Load all canonical splits for one task from local artifacts."""

    paths = build_task_artifact_paths(out_dir, task_name)
    manifest = load_task_manifest(paths.manifest_path)
    if manifest.task_name != task_name:
        raise ValueError(
            f"Manifest task name {manifest.task_name!r} does not match requested task {task_name!r}"
        )

    return {
        split: load_task_split(paths.task_dir, split, format_preference=format_preference)
        for split in manifest.splits
    }


def load_task_split(
    task_dir: str | Path,
    split: str,
    *,
    format_preference: str = "parquet",
) -> list[TaskSample]:
    """Load one split from parquet or jsonl."""

    task_path = Path(task_dir)
    if format_preference == "parquet":
        return _load_split_parquet(task_path / f"{split}.parquet")
    if format_preference == "jsonl":
        return _load_split_jsonl(task_path / f"{split}.jsonl")
    raise ValueError(f"Unsupported format preference: {format_preference!r}")


def _validate_samples(task_name: str, split: str, samples: Sequence[TaskSample]) -> None:
    for sample in samples:
        if sample.task_name != task_name:
            raise ValueError(
                f"Sample {sample.sample_id!r} belongs to task {sample.task_name!r}, expected {task_name!r}"
            )
        if sample.split != split:
            raise ValueError(
                f"Sample {sample.sample_id!r} belongs to split {sample.split!r}, expected {split!r}"
            )


def _task_sample_to_json_record(sample: TaskSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "task_name": sample.task_name,
        "split": sample.split,
        "input_payload": sample.input_payload,
        "reference": sample.reference,
        "metadata": sample.metadata,
        "allowed_tools": list(sample.allowed_tools),
    }


def _task_sample_from_json_record(record: Mapping[str, Any]) -> TaskSample:
    return TaskSample(
        sample_id=str(record["sample_id"]),
        task_name=str(record["task_name"]),
        split=str(record["split"]),
        input_payload=dict(record.get("input_payload", {})),
        reference=dict(record.get("reference", {})),
        metadata=dict(record.get("metadata", {})),
        allowed_tools=tuple(record.get("allowed_tools", ())),
    )


def _task_sample_to_parquet_record(sample: TaskSample) -> dict[str, str]:
    return {
        "sample_id": sample.sample_id,
        "task_name": sample.task_name,
        "split": sample.split,
        "input_payload_json": _dump_json(sample.input_payload),
        "reference_json": _dump_json(sample.reference),
        "metadata_json": _dump_json(sample.metadata),
        "allowed_tools_json": _dump_json(list(sample.allowed_tools)),
    }


def _task_sample_from_parquet_record(record: Mapping[str, Any]) -> TaskSample:
    return TaskSample(
        sample_id=str(record["sample_id"]),
        task_name=str(record["task_name"]),
        split=str(record["split"]),
        input_payload=_load_json(record.get("input_payload_json")),
        reference=_load_json(record.get("reference_json")),
        metadata=_load_json(record.get("metadata_json")),
        allowed_tools=tuple(_load_json(record.get("allowed_tools_json"))),
    )


def _write_split_jsonl(path: Path, samples: Sequence[TaskSample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(_task_sample_to_json_record(sample), sort_keys=True))
            handle.write("\n")


def _load_split_jsonl(path: Path) -> list[TaskSample]:
    samples: list[TaskSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(_task_sample_from_json_record(json.loads(line)))
    return samples


def _write_split_parquet(path: Path, samples: Sequence[TaskSample]) -> None:
    pa, pq = _require_pyarrow()
    schema = pa.schema([(name, pa.string()) for name in PARQUET_COLUMNS])
    rows = [_task_sample_to_parquet_record(sample) for sample in samples]
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def _load_split_parquet(path: Path) -> list[TaskSample]:
    _pa, pq = _require_pyarrow()
    table = pq.read_table(path)
    return [_task_sample_from_parquet_record(record) for record in table.to_pylist()]


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet artifact support. Install merl-env[data] or merl-env[dev]."
        ) from exc
    return pa, pq


def _dump_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _load_json(raw: Any) -> Any:
    if raw in (None, ""):
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))

