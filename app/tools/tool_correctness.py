from __future__ import annotations

import os

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from app.api.juiz import JudgeConfig


DEFAULT_SMART_TOOL_NAMES = [
    "weather",
    "forecast",
    "soil_moisture",
    "rainfall",
    "wind",
    "farm_context",
    "agronomic_recommendation",
]

EXPECTED_TOOL_KEYWORDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("umidade", "solo", "seco", "seca", "bulbo"), ("soil_moisture", "farm_context")),
    (("chuva", "acumulada", "choveu"), ("rainfall", "farm_context")),
    (("previsao", "previsão", "proximos", "próximos", "amanha", "amanhã"), ("forecast", "farm_context")),
    (("vento", "ventos", "velocidade", "direcao", "direção", "rajada"), ("wind", "farm_context")),
    (("temperatura", "calor", "frio", "geada", "radiação", "radiacao"), ("weather", "farm_context")),
    (("deriva", "aplicacao", "aplicação"), ("wind", "weather", "agronomic_recommendation")),
    (("nutricao", "nutrição", "fertirrigacao", "fertirrigação", "adubo", "nitrogenio", "nitrogênio"), ("weather", "soil_moisture", "agronomic_recommendation")),
    (("irrigacao", "irrigação", "gotejamento", "lâmina", "lamina", "rega"), ("soil_moisture", "weather", "agronomic_recommendation")),
)


def load_available_smart_tools() -> list[ToolCall]:
    raw_tool_names = os.getenv("SMART_AVAILABLE_TOOLS")
    tool_names = (
        [name.strip() for name in raw_tool_names.split(",") if name.strip()]
        if raw_tool_names
        else DEFAULT_SMART_TOOL_NAMES
    )
    return [ToolCall(name=name) for name in tool_names]


def build_tool_correctness_metric(config: JudgeConfig) -> ToolCorrectnessMetric:
    return ToolCorrectnessMetric(
        available_tools=load_available_smart_tools(),
        threshold=config.threshold,
        model=config.model,
        include_reason=config.include_reason,
    )


def infer_expected_tools(question: str, expected_output: str = "") -> list[ToolCall]:
    normalized = f"{question}\n{expected_output}".lower()
    selected: list[str] = []

    for keywords, tool_names in EXPECTED_TOOL_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            for tool_name in tool_names:
                if tool_name not in selected:
                    selected.append(tool_name)

    if not selected:
        selected = ["weather", "farm_context"]

    return [ToolCall(name=name) for name in selected]


def build_tool_correctness_case(
    question: str,
    answer: str,
    expected_output: str,
    tools_called: list[ToolCall],
    expected_tools: list[ToolCall] | None = None,
) -> LLMTestCase:
    return LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected_output,
        tools_called=tools_called,
        expected_tools=expected_tools or infer_expected_tools(question, expected_output),
    )
