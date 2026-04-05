"""Tool-enabled single-turn environment runtime."""

from __future__ import annotations

from merl_env.core.result import EvalResult, EvalTrace
from merl_env.core.sample import TaskSample
from merl_env.environments.base import ModelAdapter, RuntimeEnvironment


class ToolEnabledSingleTurnEnvironment(RuntimeEnvironment):
    """Single-turn runtime with iterative tool use."""

    def __init__(self, *args, max_tool_calls: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_tool_calls = max_tool_calls

    def run(self, sample: TaskSample, model: ModelAdapter) -> EvalResult:
        task_spec = self.get_task_spec(sample)
        messages = task_spec.build_messages(sample)
        trace_messages = list(messages)
        model_outputs: list[str] = []
        tool_events: list[dict[str, object]] = []
        executed_tool_calls = 0
        model_tools = self.get_model_tools(task_spec, sample)

        while True:
            try:
                response = model.generate(
                    messages,
                    tools=model_tools or None,
                    max_new_tokens=self._max_new_tokens,
                )
            except Exception as exc:
                trace = EvalTrace(
                    messages=tuple(trace_messages),
                    tool_events=tuple(tool_events),
                    model_outputs=tuple(model_outputs),
                )
                return EvalResult(
                    sample_id=sample.sample_id,
                    task_name=sample.task_name,
                    passed=False,
                    tool_events=tuple(tool_events),
                    trace=trace,
                    error=f"model error: {exc}",
                    stop_reason="model_error",
                )

            model_outputs.append(response.text)
            assistant_message = self.build_assistant_message(response)
            messages.append(assistant_message)
            trace_messages.append(assistant_message)

            parsed_tool_calls = self.parse_tool_calls(response)
            if parsed_tool_calls.success and parsed_tool_calls.tool_calls:
                if self._tool_executor is None:
                    trace = EvalTrace(
                        messages=tuple(trace_messages),
                        tool_events=tuple(tool_events),
                        model_outputs=tuple(model_outputs),
                    )
                    return EvalResult(
                        sample_id=sample.sample_id,
                        task_name=sample.task_name,
                        passed=False,
                        raw_output=response.text,
                        tool_events=tuple(tool_events),
                        trace=trace,
                        error="tool call requested but no tool registry was configured",
                        stop_reason="tooling_not_configured",
                    )
                if executed_tool_calls + len(parsed_tool_calls.tool_calls) > self._max_tool_calls:
                    trace = EvalTrace(
                        messages=tuple(trace_messages),
                        tool_events=tuple(tool_events),
                        model_outputs=tuple(model_outputs),
                    )
                    return EvalResult(
                        sample_id=sample.sample_id,
                        task_name=sample.task_name,
                        passed=False,
                        raw_output=response.text,
                        tool_events=tuple(tool_events),
                        trace=trace,
                        error="maximum tool-call budget exceeded before final answer",
                        stop_reason="max_tool_calls_exceeded",
                    )

                for tool_call in parsed_tool_calls.tool_calls:
                    tool_result = self._tool_executor.execute(tool_call)
                    executed_tool_calls += 1
                    tool_events.append(tool_result.to_trace_event())
                    tool_message = tool_result.to_message()
                    messages.append(tool_message)
                    trace_messages.append(tool_message)
                continue

            parsed_final = self.parse_final_answer(task_spec, response.text)
            if parsed_final.success:
                try:
                    verification = self.verify(task_spec, sample, parsed_final.answer or {})
                except Exception as exc:
                    trace = EvalTrace(
                        messages=tuple(trace_messages),
                        tool_events=tuple(tool_events),
                        model_outputs=tuple(model_outputs),
                    )
                    return EvalResult(
                        sample_id=sample.sample_id,
                        task_name=sample.task_name,
                        passed=False,
                        parsed_answer=parsed_final.answer,
                        raw_output=response.text,
                        tool_events=tuple(tool_events),
                        trace=trace,
                        error=f"verification error: {exc}",
                        stop_reason="verification_error",
                    )

                trace = EvalTrace(
                    messages=tuple(trace_messages),
                    tool_events=tuple(tool_events),
                    model_outputs=tuple(model_outputs),
                )
                return EvalResult(
                    sample_id=sample.sample_id,
                    task_name=sample.task_name,
                    passed=verification.passed,
                    score=verification.score,
                    reward=verification.reward,
                    parsed_answer=parsed_final.answer,
                    raw_output=response.text,
                    tool_events=tuple(tool_events),
                    verification=verification.to_dict(),
                    trace=trace,
                    stop_reason="completed",
                )

            trace = EvalTrace(
                messages=tuple(trace_messages),
                tool_events=tuple(tool_events),
                model_outputs=tuple(model_outputs),
            )
            error_message = (
                parsed_tool_calls.error.message
                if parsed_tool_calls.error is not None
                else parsed_final.error.message
                if parsed_final.error is not None
                else "model output did not contain a tool call or final answer"
            )
            return EvalResult(
                sample_id=sample.sample_id,
                task_name=sample.task_name,
                passed=False,
                raw_output=response.text,
                tool_events=tuple(tool_events),
                trace=trace,
                error=error_message,
                stop_reason="parse_error",
            )


__all__ = ["ToolEnabledSingleTurnEnvironment"]
