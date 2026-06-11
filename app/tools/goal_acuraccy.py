from __future__ import annotations

from deepeval.metrics import GoalAccuracyMetric
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn

from app.api import JudgeConfig


SMART_AGENT_SCENARIO = (
    "O usuario consulta a API SMART, implementada em Hono, sobre dados ambientais, "
    "meteorologicos e de manejo agricola da Fazenda NSAAB por meio de um "
    "orquestrador multiagente com ferramentas MCP."
)

SMART_EXPECTED_OUTCOME = (
    "O agente deve compreender o objetivo do usuario, acionar os agentes e ferramentas "
    "adequados quando necessario, usar os dados retornados pela API e entregar uma "
    "resposta precisa, fundamentada e util para decisao agricola."
)


def build_goal_accuracy_metric(config: JudgeConfig) -> GoalAccuracyMetric:
    return GoalAccuracyMetric(
        threshold=config.threshold,
        model=config.model,
        include_reason=config.include_reason,
        async_mode=False,
    )


def build_goal_accuracy_case(
    question: str,
    answer: str,
    expected_output: str,
    tools_called: list[ToolCall],
) -> ConversationalTestCase:
    assistant_turn_kwargs = {"role": "assistant", "content": answer}
    if tools_called:
        assistant_turn_kwargs["tools_called"] = tools_called

    return ConversationalTestCase(
        scenario=SMART_AGENT_SCENARIO,
        expected_outcome=(
            f"{SMART_EXPECTED_OUTCOME}\n\n"
            f"Resposta esperada de referencia para este objetivo: {expected_output}"
        ),
        turns=[
            Turn(role="user", content=question),
            Turn(**assistant_turn_kwargs),
        ],
    )
