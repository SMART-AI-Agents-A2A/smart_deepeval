from __future__ import annotations

from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe, update_current_span

from app.api.conect_api import SmartApiConfig, call_smart_chat, extract_tools_called
from app.api.juiz import JudgeConfig


def build_task_completion_metric(config: JudgeConfig) -> TaskCompletionMetric:
    return TaskCompletionMetric(
        threshold=config.threshold,
        model=config.model,
        include_reason=config.include_reason,
    )


def build_observed_smart_agent(api_config: SmartApiConfig, metric: TaskCompletionMetric):
    @observe(type="agent", metrics=[metric])
    def observed_smart_agent(question: str, conversation_id: str, expected_output: str = "") -> str:
        answer, payload = call_smart_chat(question, conversation_id, api_config)
        tools_called = extract_tools_called(payload)

        span_kwargs = {
            "input": question,
            "output": answer,
        }
        if expected_output:
            span_kwargs["expected_output"] = expected_output
        if tools_called:
            span_kwargs["tools_called"] = tools_called
        update_current_span(**span_kwargs)

        return answer

    return observed_smart_agent
