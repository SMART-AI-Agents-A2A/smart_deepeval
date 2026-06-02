from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    threshold: float
    include_reason: bool


def load_judge_config() -> JudgeConfig:
    return JudgeConfig(
        model=os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o"),
        threshold=float(os.getenv("DEEPEVAL_THRESHOLD", "0.7")),
        include_reason=os.getenv("DEEPEVAL_INCLUDE_REASON", "true").lower() != "false",
    )


def configure_judge_environment() -> JudgeConfig:
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    return load_judge_config()
