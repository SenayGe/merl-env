#!/usr/bin/env python3
"""Build one or more task artifact bundles and optionally preview rendered cases."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

if sys.version_info < (3, 10):
    raise SystemExit("merl-env scripts require Python 3.10 or newer")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from merl_env.data import (  # noqa: E402
    BigQueryMimicSource,
    BigQueryMimicSourceConfig,
    DiagnosisBuilderConfig,
    InMemoryMimicSource,
    MimicSource,
    SplitConfig,
    build_diagnosis_artifacts,
    build_discharge_artifacts,
    build_icd_artifacts,
    load_task_artifacts,
)
from merl_env.tasks import DiagnosisTaskSpec, DischargeTaskSpec, IcdTaskSpec  # noqa: E402

TASK_NAMES: tuple[str, ...] = ("diagnosis", "discharge", "icd")
BUILDERS = {
    "diagnosis": build_diagnosis_artifacts,
    "discharge": build_discharge_artifacts,
    "icd": build_icd_artifacts,
}
TASK_SPECS = {
    "diagnosis": DiagnosisTaskSpec,
    "discharge": DischargeTaskSpec,
    "icd": IcdTaskSpec,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local task artifacts for one or more merl-env Phase 1 tasks."
    )
    parser.add_argument(
        "--task",
        choices=("all", *TASK_NAMES),
        default="all",
        help="Task to build. Use 'all' to build every Phase 1 task.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts",
        help="Output directory for built artifact bundles.",
    )
    parser.add_argument(
        "--preview-split",
        choices=("train", "val", "test"),
        default="train",
        help="Split to preview after build.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=1,
        help="Number of rendered samples to preview per built task. Use 0 to skip preview.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1200,
        help="Maximum characters to print per rendered message. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--show-reference",
        action="store_true",
        help="Include the sample reference payload in previews.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Subject-level training split fraction.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Subject-level validation split fraction.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Subject-level test split fraction.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=7,
        help="Deterministic seed for subject-level split assignment.",
    )
    parser.add_argument(
        "--diagnosis-max-samples-per-label",
        type=int,
        help="Optional cap on diagnosis examples per label, applied before split assignment.",
    )
    parser.add_argument(
        "--diagnosis-sampling-seed",
        type=int,
        default=7,
        help="Deterministic seed for diagnosis per-label sampling.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-fixtures",
        type=Path,
        help="Path to a JSON fixture file keyed by builder query name.",
    )
    source_group.add_argument(
        "--source-object",
        type=str,
        help="Import path for a MimicSource instance or factory, e.g. 'pkg.module:build_source' or '/abs/path/source.py:SOURCE'.",
    )
    source_group.add_argument(
        "--source-bigquery",
        action="store_true",
        help="Build directly from MIMIC tables in BigQuery using ADC credentials.",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        help="GCP project id used for BigQuery billing and query execution.",
    )
    parser.add_argument(
        "--bq-location",
        type=str,
        default="US",
        help="BigQuery job location for BigQuery-backed builds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    split_config = SplitConfig(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.split_seed,
    )
    source = load_source(args)
    task_names = list(TASK_NAMES if args.task == "all" else (args.task,))

    print(f"source: {describe_source(args)}")
    print(f"out_dir: {args.out_dir}")
    print(
        "split_config: "
        f"train={split_config.train_frac} val={split_config.val_frac} "
        f"test={split_config.test_frac} seed={split_config.seed}"
    )

    for task_name in task_names:
        if task_name == "diagnosis":
            manifest = build_diagnosis_artifacts(
                str(args.out_dir),
                source,
                config=DiagnosisBuilderConfig(
                    max_samples_per_label=args.diagnosis_max_samples_per_label,
                    sampling_seed=args.diagnosis_sampling_seed,
                ),
                split_config=split_config,
            )
        else:
            manifest = BUILDERS[task_name](
                str(args.out_dir),
                source,
                split_config=split_config,
            )
        task_dir = args.out_dir / task_name
        print(
            f"built task: {task_name} "
            f"(train={manifest.split_counts.get('train', 0)}, "
            f"val={manifest.split_counts.get('val', 0)}, "
            f"test={manifest.split_counts.get('test', 0)}) "
            f"-> {task_dir}"
        )
        if args.preview_limit > 0:
            preview_task(
                task_name,
                out_dir=args.out_dir,
                split=args.preview_split,
                limit=args.preview_limit,
                preview_chars=args.preview_chars,
                show_reference=args.show_reference,
            )
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.diagnosis_max_samples_per_label is not None
        and args.diagnosis_max_samples_per_label <= 0
    ):
        raise ValueError("--diagnosis-max-samples-per-label must be positive")
    if not args.source_bigquery:
        return
    if not args.gcp_project or not args.gcp_project.strip():
        raise ValueError("--gcp-project is required with --source-bigquery")
    if args.task == "all":
        raise ValueError(
            "BigQuery builds currently support only 'diagnosis' or 'discharge'; "
            "task 'all' is not supported"
        )
    if args.task == "icd":
        raise ValueError(
            "BigQuery builds do not yet support task 'icd'; use 'diagnosis' or 'discharge'"
        )


def load_source(args: argparse.Namespace) -> MimicSource:
    if args.source_fixtures is not None:
        return load_fixture_source(args.source_fixtures)
    if args.source_object is not None:
        return load_custom_source(args.source_object)
    if args.source_bigquery:
        return BigQueryMimicSource(
            BigQueryMimicSourceConfig(
                project_id=str(args.gcp_project),
                location=str(args.bq_location),
            )
        )
    raise ValueError("either --source-fixtures or --source-object must be provided")


def describe_source(args: argparse.Namespace) -> str:
    if args.source_fixtures is not None:
        return str(args.source_fixtures)
    if args.source_object is not None:
        return args.source_object
    if args.source_bigquery:
        return f"bigquery project={args.gcp_project} location={args.bq_location}"
    return "unknown"


def load_fixture_source(path: Path) -> MimicSource:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("fixture file must decode to an object keyed by query name")
    return InMemoryMimicSource(payload)


def load_custom_source(spec: str) -> MimicSource:
    candidate = _load_object(spec)
    if isinstance(candidate, MimicSource):
        return candidate
    if isinstance(candidate, type) and issubclass(candidate, MimicSource):
        instance = candidate()
        return instance
    if callable(candidate):
        produced = candidate()
        if isinstance(produced, MimicSource):
            return produced
    raise TypeError(
        f"{spec!r} did not resolve to a MimicSource instance or zero-argument factory"
    )


def _load_object(spec: str) -> Any:
    module_ref, separator, attr_name = spec.partition(":")
    if not separator or not attr_name:
        raise ValueError(
            "source object must look like 'module.path:attribute' or '/path/to/file.py:attribute'"
        )

    module_path = Path(module_ref).expanduser()
    if module_ref.endswith(".py") or module_path.exists():
        resolved_path = module_path.resolve()
        module_spec = importlib.util.spec_from_file_location(
            resolved_path.stem,
            resolved_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"could not load module from {resolved_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"module {module.__name__!r} does not define {attr_name!r}") from exc


def preview_task(
    task_name: str,
    *,
    out_dir: Path,
    split: str,
    limit: int,
    preview_chars: int,
    show_reference: bool,
) -> None:
    artifacts = load_task_artifacts(out_dir, task_name)
    samples = artifacts.get(split, [])
    if not samples:
        print(f"preview task: {task_name} split={split} -> no samples")
        return

    task_spec = TASK_SPECS[task_name]()
    print(f"preview task: {task_name} split={split} count={min(limit, len(samples))}")
    for index, sample in enumerate(samples[:limit], start=1):
        print(f"sample[{index}] id={sample.sample_id} allowed_tools={list(sample.allowed_tools)}")
        if show_reference:
            print(
                "reference: "
                + json.dumps(sample.reference, indent=2, sort_keys=True)
            )
        for message in task_spec.build_messages(sample):
            print(f"[{message.role}]")
            print(_truncate(message.content, preview_chars))


def _truncate(text: str, preview_chars: int) -> str:
    if preview_chars <= 0 or len(text) <= preview_chars:
        return text
    return text[:preview_chars].rstrip() + "\n... [truncated]"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - exercised through CLI integration tests
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
