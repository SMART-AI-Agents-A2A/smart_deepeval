from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel


def _clean_optional_env(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    return stripped or None


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 3:
        return "\n".join(lines[1:-1]).strip()
    return stripped


class OpenAICompatibleJudgeModel(DeepEvalBaseLLM):
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str | None,
        timeout: int,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return f"openai-compatible/{self.model_name}"

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        gateway_token = _clean_optional_env(os.getenv("CF_AIG_TOKEN"))
        if gateway_token:
            headers["cf-aig-authorization"] = f"Bearer {gateway_token}"

        byok_alias = _clean_optional_env(os.getenv("CF_AIG_BYOK_ALIAS"))
        if byok_alias:
            headers["cf-aig-byok-alias"] = byok_alias

        return headers

    def _payload(self, prompt: str, schema: type[BaseModel] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            }

        return payload

    def _request(self, payload: dict[str, Any]) -> str:
        response = requests.post(
            self._chat_completions_url(),
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return content if isinstance(content, str) else str(content)

        return data.get("output_text") or data.get("response") or ""

    def _parse_schema_response(self, content: str, schema: type[BaseModel]) -> BaseModel:
        cleaned = _strip_json_fence(content)
        try:
            return schema.model_validate_json(cleaned)
        except Exception:
            return schema.model_validate(json.loads(cleaned))

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        payload = self._payload(prompt, schema)
        try:
            content = self._request(payload)
        except requests.HTTPError:
            if not schema:
                raise

            payload = self._payload(prompt, None)
            payload["response_format"] = {"type": "json_object"}
            content = self._request(payload)

        if schema:
            return self._parse_schema_response(content, schema)
        return content

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return self.generate(prompt, schema)


@dataclass(frozen=True)
class JudgeConfig:
    model: str | OpenAICompatibleJudgeModel
    model_name: str
    threshold: float
    include_reason: bool
    base_url: str | None


def load_judge_config() -> JudgeConfig:
    model_name = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o-mini")
    base_url = _clean_optional_env(os.getenv("DEEPEVAL_JUDGE_BASE_URL"))
    api_key = _clean_optional_env(os.getenv("OPENAI_API_KEY"))
    timeout = int(os.getenv("DEEPEVAL_JUDGE_TIMEOUT", "120"))

    model: str | OpenAICompatibleJudgeModel = model_name
    if base_url:
        model = OpenAICompatibleJudgeModel(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    return JudgeConfig(
        model=model,
        model_name=model_name,
        threshold=float(os.getenv("DEEPEVAL_THRESHOLD", "0.7")),
        include_reason=os.getenv("DEEPEVAL_INCLUDE_REASON", "true").lower() != "false",
        base_url=base_url,
    )


def configure_judge_environment() -> JudgeConfig:
    openai_api_key = _clean_optional_env(os.getenv("OPENAI_API_KEY"))

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    return load_judge_config()
