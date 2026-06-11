from __future__ import annotations

import inspect
import os

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from app.api import JudgeConfig


DEFAULT_SMART_TOOL_NAMES = [
    "smart_soil_data",
    "smart_rain_accumulated",
    "smart_rain_forecast",
    "smart_radiation_solar",
    "smart_lightning_incidence",
    "smart_lightning_strikes",
    "smart_lightning_risk",
    "smart_wind_speed",
    "smart_wind_direction",
    "smart_wind_gust",
    "smart_wind_current_weather",
    "smart_wind_forecast",
    "smart_air_temperature",
    "smart_air_humidity",
    "smart_air_pressure",
    "smart_air_conditions",
    "smart_air_current_weather",
]

SMART_AGENT_TOOLS: dict[str, tuple[str, ...]] = {
    "solo": ("smart_soil_data",),
    "chuva": ("smart_rain_accumulated", "smart_rain_forecast"),
    "radiacao": ("smart_radiation_solar",),
    "raio": ("smart_lightning_incidence", "smart_lightning_strikes", "smart_lightning_risk"),
    "ar": (
        "smart_air_temperature",
        "smart_air_humidity",
        "smart_air_pressure",
        "smart_air_conditions",
        "smart_air_current_weather",
    ),
    "vento": (
        "smart_wind_speed",
        "smart_wind_direction",
        "smart_wind_gust",
        "smart_wind_current_weather",
        "smart_wind_forecast",
    ),
}

EXPECTED_TOOL_KEYWORDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("umidade do solo", "solo", "seco", "seca", "bulbo", "teros12"), ("smart_soil_data",)),
    (("temperatura do solo", "condutividade elétrica do solo", "condutividade eletrica do solo"), ("smart_soil_data",)),
    (("chuva acumulada", "choveu", "precipitação acumulada", "precipitacao acumulada"), ("smart_rain_accumulated",)),
    (("previsão de chuva", "previsao de chuva", "risco de chuva"), ("smart_rain_forecast",)),
    (("radiação", "radiacao", "radiação solar", "radiacao solar", "insolação", "insolacao"), ("smart_radiation_solar",)),
    (("raio", "raios", "descarga atmosférica", "descarga atmosferica", "risco elétrico", "risco eletrico"), ("smart_lightning_incidence", "smart_lightning_strikes", "smart_lightning_risk")),
    (("velocidade do vento", "vento forte", "ventos fortes"), ("smart_wind_speed",)),
    (("direção do vento", "direcao do vento"), ("smart_wind_direction",)),
    (("rajada", "rajadas"), ("smart_wind_gust",)),
    (("previsão de vento", "previsao de vento"), ("smart_wind_forecast",)),
    (("vento atual", "deriva", "aplicação agrícola", "aplicacao agricola"), ("smart_wind_current_weather",)),
    (("temperatura do ar", "calor", "frio", "geada"), ("smart_air_temperature",)),
    (("umidade do ar", "umidade relativa"), ("smart_air_humidity",)),
    (("pressão", "pressao", "pressão atmosférica", "pressao atmosferica"), ("smart_air_pressure",)),
    (("condições do ar", "condicoes do ar", "condições gerais", "condicoes gerais"), ("smart_air_conditions",)),
    (("tempo atual", "clima atual", "openweather"), ("smart_air_current_weather",)),
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
    use_llm_tool_selection = os.getenv("DEEPEVAL_TOOL_SELECTION_WITH_LLM", "false").lower() == "true"

    metric_kwargs = {
        "available_tools": load_available_smart_tools() if use_llm_tool_selection else None,
        "threshold": config.threshold,
        "model": config.model,
        "include_reason": config.include_reason,
    }

    if "async_mode" in inspect.signature(ToolCorrectnessMetric).parameters:
        metric_kwargs["async_mode"] = False

    return ToolCorrectnessMetric(**metric_kwargs)


def infer_expected_tools(question: str, expected_output: str = "") -> list[ToolCall]:
    normalized = f"{question}\n{expected_output}".lower()
    selected: list[str] = []

    for keywords, tool_names in EXPECTED_TOOL_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            for tool_name in tool_names:
                if tool_name not in selected:
                    selected.append(tool_name)

    return [ToolCall(name=name) for name in selected]


def build_tool_correctness_case(
    question: str,
    answer: str,
    expected_output: str,
    tools_called: list[ToolCall],
    expected_tools: list[ToolCall] | None = None,
) -> LLMTestCase:
    resolved_expected = expected_tools or infer_expected_tools(question, expected_output)
    if not resolved_expected:
        resolved_expected = tools_called or [ToolCall(name="smart_air_conditions")]

    return LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected_output,
        tools_called=tools_called,
        expected_tools=resolved_expected,
    )