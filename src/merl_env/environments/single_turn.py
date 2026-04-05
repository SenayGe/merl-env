"""Single-turn environment runtime."""

from __future__ import annotations

from merl_env.core.result import EvalResult, EvalTrace
from merl_env.core.sample import TaskSample
from merl_env.environments.base import ModelAdapter, RuntimeEnvironment


class SingleTurnEnvironment(RuntimeEnvironment):
    """Build messages, call model, parse final answer, and verify once."""

    def run(self, sample: TaskSample, model: ModelAdapter) -> EvalResult:
        task_spec = self.get_task_spec(sample)
        messages = task_spec.build_messages(sample)
        trace_messages = list(messages)
        model_outputs: list[str] = []

        try:
            response = model.generate(
                messages,
                max_new_tokens=self._max_new_tokens,
            )
        except Exception as exc:
            trace = EvalTrace(messages=tuple(trace_messages), model_outputs=tuple(model_outputs))
            return EvalResult(
                sample_id=sample.sample_id,
                task_name=sample.task_name,
                passed=False,
                trace=trace,
                error=f"model error: {exc}",
                stop_reason="model_error",
            )

        model_outputs.append(response.text)
        assistant_message = self.build_assistant_message(response)
        trace_messages.append(assistant_message)

        parsed = self.parse_final_answer(task_spec, response.text)
        if not parsed.success:
            trace = EvalTrace(messages=tuple(trace_messages), model_outputs=tuple(model_outputs))
            return EvalResult(
                sample_id=sample.sample_id,
                task_name=sample.task_name,
                passed=False,
                raw_output=response.text,
                trace=trace,
                error=parsed.error.message if parsed.error is not None else "failed to parse final answer",
                stop_reason="parse_error",
            )

        try:
            verification = self.verify(task_spec, sample, parsed.answer or {})
        except Exception as exc:
            trace = EvalTrace(messages=tuple(trace_messages), model_outputs=tuple(model_outputs))
            return EvalResult(
                sample_id=sample.sample_id,
                task_name=sample.task_name,
                passed=False,
                parsed_answer=parsed.answer,
                raw_output=response.text,
                trace=trace,
                error=f"verification error: {exc}",
                stop_reason="verification_error",
            )

        trace = EvalTrace(messages=tuple(trace_messages), model_outputs=tuple(model_outputs))
        return EvalResult(
            sample_id=sample.sample_id,
            task_name=sample.task_name,
            passed=verification.passed,
            score=verification.score,
            reward=verification.reward,
            parsed_answer=parsed.answer,
            raw_output=response.text,
            verification=verification.to_dict(),
            trace=trace,
            stop_reason="completed",
        )


__all__ = ["SingleTurnEnvironment"]
