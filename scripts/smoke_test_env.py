#!/usr/bin/env python3
"""Run a deterministic end-to-end smoke test through a merl-env environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if sys.version_info < (3, 10):
    raise SystemExit("merl-env scripts require Python 3.10 or newer")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from merl_env.core import TaskSample, ToolCall  # noqa: E402
from merl_env.data import load_task_split  # noqa: E402
from merl_env.environments import (  # noqa: E402
    ModelResponse,
    SingleTurnEnvironment,
    ToolEnabledSingleTurnEnvironment,
)
from merl_env.tasks import DiagnosisTaskSpec, DischargeTaskSpec, IcdTaskSpec  # noqa: E402
from merl_env.tools import (  # noqa: E402
    IcdLookupTool,
    InMemoryWebSearchBackend,
    LabRangesTool,
    ToolRegistry,
    WebSearchTool,
)

TASK_SPECS = {
    "diagnosis": DiagnosisTaskSpec,
    "discharge": DischargeTaskSpec,
    "icd": IcdTaskSpec,
}


class DeterministicSmokeModel:
    """Deterministic adapter that exercises the runtime without a live model."""

    def __init__(self, sample: TaskSample, *, prefer_tool_call: bool) -> None:
        self._sample = sample
        self._prefer_tool_call = prefer_tool_call
        self._tool_call_emitted = False

    def generate(self, messages, tools=None, max_new_tokens=None) -> ModelResponse:
        del messages
        del max_new_tokens
        tool_call = self._build_tool_call(tools or [])
        if tool_call is not None:
            self._tool_call_emitted = True
            return ModelResponse(
                text="",
                tool_calls=(tool_call,),
                stop_reason="tool_call",
            )
        return ModelResponse(
            text=_build_final_answer(self._sample),
            stop_reason="stop",
        )

    def _build_tool_call(self, tools) -> ToolCall | None:
        if self._tool_call_emitted or not self._prefer_tool_call or not tools:
            return None

        allowed = list(self._sample.allowed_tools)
        if "lab_ranges" in allowed:
            labs = self._sample.input_payload.get("labs", {})
            if isinstance(labs, dict) and labs:
                lab_name = sorted(str(name) for name in labs)[0]
            else:
                lab_name = "Potassium"
            return ToolCall(
                call_id="smoke-tool-1",
                tool_name="lab_ranges",
                arguments={"lab_name": lab_name},
            )

        if "icd_lookup" in allowed:
            codes = self._sample.reference.get("icd_codes", [])
            if isinstance(codes, list) and codes:
                arguments = {"code": str(codes[0])}
            else:
                arguments = {"query": "discharge diagnosis"}
            return ToolCall(
                call_id="smoke-tool-1",
                tool_name="icd_lookup",
                arguments=arguments,
            )

        if "web_search" in allowed:
            return ToolCall(
                call_id="smoke-tool-1",
                tool_name="web_search",
                arguments={"query": f"{self._sample.task_name} clinical guidance"},
            )
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic smoke test through a merl-env environment."
    )
    parser.add_argument(
        "--task",
        choices=tuple(TASK_SPECS),
        required=True,
        help="Task to load from built artifacts.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing built task artifact bundles.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="Split to sample from.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        help="Explicit sample id to run. Defaults to the sample at --sample-index.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based sample index to use when --sample-id is not provided.",
    )
    parser.add_argument(
        "--environment",
        choices=("single_turn", "tool_enabled"),
        default="single_turn",
        help="Environment runtime to exercise.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=3,
        help="Maximum tool-call budget for tool-enabled smoke runs.",
    )
    parser.add_argument(
        "--trace-chars",
        type=int,
        default=600,
        help="Maximum characters to print per trace message. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--icd-dictionary",
        type=Path,
        help="Optional local ICD dictionary artifact for the icd_lookup tool.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample = load_sample(args)
    task_spec = TASK_SPECS[args.task]()

    if args.environment == "tool_enabled":
        tool_registry = build_tool_registry(sample, icd_dictionary_path=args.icd_dictionary)
        env = ToolEnabledSingleTurnEnvironment(
            [task_spec],
            tool_registry=tool_registry,
            max_tool_calls=args.max_tool_calls,
        )
        model = DeterministicSmokeModel(sample, prefer_tool_call=bool(sample.allowed_tools))
    else:
        env = SingleTurnEnvironment([task_spec])
        model = DeterministicSmokeModel(sample, prefer_tool_call=False)

    result = env.run(sample, model)
    print(f"task: {sample.task_name}")
    print(f"sample_id: {sample.sample_id}")
    print(f"environment: {args.environment}")
    print(f"split: {sample.split}")
    print(f"passed: {result.passed}")
    print(f"stop_reason: {result.stop_reason}")
    if result.score is not None:
        print(f"score: {result.score}")
    if result.reward is not None:
        print(f"reward: {result.reward}")
    if result.error:
        print(f"error: {result.error}")
    if result.parsed_answer is not None:
        print("parsed_answer:")
        print(json.dumps(result.parsed_answer, indent=2, sort_keys=True))
    if result.verification is not None:
        print("verification:")
        print(json.dumps(result.verification, indent=2, sort_keys=True))
    if result.tool_events:
        print("tool_events:")
        print(json.dumps(list(result.tool_events), indent=2, sort_keys=True))
    if result.trace is not None:
        print("trace:")
        for index, message in enumerate(result.trace.messages, start=1):
            header = message.role if message.name is None else f"{message.role}:{message.name}"
            print(f"{index}. [{header}]")
            print(_truncate(message.content, args.trace_chars))
    return 0 if result.passed else 1


def load_sample(args: argparse.Namespace) -> TaskSample:
    task_dir = args.artifacts_dir / args.task
    samples = load_task_split(task_dir, args.split)
    if not samples:
        raise ValueError(f"no samples found for task={args.task!r} split={args.split!r}")

    if args.sample_id is not None:
        for sample in samples:
            if sample.sample_id == args.sample_id:
                return sample
        raise ValueError(f"sample_id {args.sample_id!r} was not found in split {args.split!r}")

    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(
            f"sample_index {args.sample_index} is out of range for split {args.split!r}"
        )
    return samples[args.sample_index]


def build_tool_registry(
    sample: TaskSample,
    *,
    icd_dictionary_path: Path | None,
) -> ToolRegistry:
    tools = [
        LabRangesTool(),
        WebSearchTool(InMemoryWebSearchBackend()),
    ]

    icd_entries = None
    if icd_dictionary_path is None:
        metadata_labels = sample.metadata.get("icd_labels")
        if isinstance(metadata_labels, list):
            icd_entries = metadata_labels
    tools.append(
        IcdLookupTool(
            entries=icd_entries,
            artifact_path=icd_dictionary_path,
        )
    )
    return ToolRegistry(tools)


def _build_final_answer(sample: TaskSample) -> str:
    if sample.task_name == "diagnosis":
        payload = {"primary_diagnosis": sample.reference.get("primary_diagnosis")}
    elif sample.task_name == "discharge":
        payload = {
            "safe_for_discharge_24h": sample.reference.get("safe_for_discharge_24h"),
            "has_hard_barrier": sample.reference.get("has_hard_barrier"),
            "rationale": "Deterministic smoke answer copied from the sample reference.",
        }
    elif sample.task_name == "icd":
        payload = {"icd_codes": sample.reference.get("icd_codes", [])}
    else:
        raise KeyError(f"unsupported task {sample.task_name!r}")
    return f"<final_answer>{json.dumps(payload, sort_keys=True)}</final_answer>"


def _truncate(text: str, trace_chars: int) -> str:
    if trace_chars <= 0 or len(text) <= trace_chars:
        return text
    return text[:trace_chars].rstrip() + "\n... [truncated]"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - exercised through CLI integration tests
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
