from __future__ import annotations

import json
import os
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI


class CloudflareGatewayEvalModel(DeepEvalBaseLLM):
    def __init__(self) -> None:
        self.api_key = os.getenv("CF_AIG_TOKEN", "").strip()
        self.base_url = os.getenv("CF_AIG_BASE_URL", "").strip()
        self.model = os.getenv("CF_AIG_MODEL", "openai/gpt-5").strip()

        if not self.api_key:
            raise ValueError("CF_AIG_TOKEN não foi definido no .env.local.")

        if not self.base_url:
            raise ValueError("CF_AIG_BASE_URL não foi definido no .env.local.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "cf-aig-authorization": f"Bearer {self.api_key}",
            },
        )

        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "cf-aig-authorization": f"Bearer {self.api_key}",
            },
        )

    def load_model(self) -> OpenAI:
        return self.client

    def get_model_name(self) -> str:
        return f"Cloudflare AI Gateway - {self.model}"

    def generate(self, prompt: str, schema: Any | None = None) -> Any:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um avaliador de respostas. "
                        "Responda exatamente no formato solicitado."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        text = response.choices[0].message.content or ""

        if schema is not None:
            return self._parse_schema(text, schema)

        return text

    async def a_generate(self, prompt: str, schema: Any | None = None) -> Any:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um avaliador de respostas. "
                        "Responda exatamente no formato solicitado."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        text = response.choices[0].message.content or ""

        if schema is not None:
            return self._parse_schema(text, schema)

        return text

    def _parse_schema(self, text: str, schema: Any) -> Any:
        try:
            return schema.model_validate_json(text)
        except Exception:
            pass

        try:
            parsed = json.loads(self._extract_json(text))
            return schema.model_validate(parsed)
        except Exception as exc:
            raise ValueError(
                "O modelo via Cloudflare AI Gateway não retornou JSON válido para o DeepEval. "
                f"Resposta recebida: {text}"
            ) from exc

    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("Nenhum JSON encontrado na resposta do modelo.")

        return text[start : end + 1]