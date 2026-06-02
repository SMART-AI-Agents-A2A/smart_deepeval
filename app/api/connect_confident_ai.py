from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from deepeval import evaluate


TRUE_VALUES = {"1", "true", "yes", "y", "sim", "s"}


@dataclass(frozen=True)
class ConfidentAiConfig:
    enabled: bool
    api_key: str | None
    run_identifier: str
    send_hyperparameters: bool
    base_hyperparameters: dict[str, str] = field(default_factory=dict)


def _is_enabled(value: str | None) -> bool:
    return value.strip().lower() in TRUE_VALUES if value else False


def load_confident_ai_config() -> ConfidentAiConfig:
    enabled = _is_enabled(os.getenv("CONFIDENT_AI_ENABLED"))
    api_key = os.getenv("CONFIDENT_API_KEY")
    run_identifier = os.getenv("CONFIDENT_AI_RUN_IDENTIFIER", "smart-deepeval")
    send_hyperparameters = os.getenv("CONFIDENT_AI_SEND_HYPERPARAMETERS", "true").lower() != "false"

    return ConfidentAiConfig(
        enabled=enabled,
        api_key=api_key,
        run_identifier=run_identifier,
        send_hyperparameters=send_hyperparameters,
        base_hyperparameters={
            "project": os.getenv("CONFIDENT_AI_PROJECT", "SMART API"),
            "environment": os.getenv("CONFIDENT_AI_ENVIRONMENT", "local"),
        },
    )


def configure_confident_ai_environment(config: ConfidentAiConfig) -> None:
    if config.enabled and config.api_key:
        os.environ["CONFIDENT_API_KEY"] = config.api_key
        return

    os.environ.pop("CONFIDENT_API_KEY", None)


def build_test_run_identifier(config: ConfidentAiConfig, metric_name: str) -> str | None:
    if not config.enabled or not config.api_key:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    normalized_metric = metric_name.strip().lower().replace(" ", "-").replace("_", "-")
    return f"{config.run_identifier}-{normalized_metric}-{timestamp}"


def build_hyperparameters(
    config: ConfidentAiConfig,
    metric_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    if not config.enabled or not config.api_key or not config.send_hyperparameters:
        return None

    values: dict[str, Any] = {
        **config.base_hyperparameters,
        "metric": metric_name,
        **(extra or {}),
    }
    return {key: str(value) for key, value in values.items() if value is not None}


def evaluate_with_confident_ai(
    *,
    test_cases,
    metrics,
    metric_name: str,
    config: ConfidentAiConfig,
    hyperparameters: dict[str, Any] | None = None,
):
    configure_confident_ai_environment(config)

    evaluate_kwargs: dict[str, Any] = {
        "test_cases": test_cases,
        "metrics": metrics,
    }
    identifier = build_test_run_identifier(config, metric_name)
    if identifier:
        evaluate_kwargs["identifier"] = identifier

    confident_hyperparameters = build_hyperparameters(config, metric_name, hyperparameters)
    if confident_hyperparameters:
        evaluate_kwargs["hyperparameters"] = confident_hyperparameters

    return evaluate(**evaluate_kwargs)
