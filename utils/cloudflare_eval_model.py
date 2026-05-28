from __future__ import annotations

import json
import os
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI


class CloudflareWorkersAIModel(DeepEvalBaseLLM):
    def __init__(self) -> None:
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        self.model = os.getenv(
            "CLOUDFLARE_EVALUATOR_MODEL",
            "@cf/qwen/qwen3-30b-a3b-fp8",
        ).strip()

        if not self.account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID não foi definido no .env.local.")

        if not self.api_token:
            raise ValueError("CLOUDFLARE_API_TOKEN não foi definido no .env.local.")

        self.base_url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/v1"
        )

        self.client = OpenAI(
            api_key=self.api_token,
            base_url=self.base_url,
        )

        self.async_client = AsyncOpenAI(
            api_key=self.api_token,
            base_url=self.base_url,
        )

    def load_model(self) -> OpenAI:
        return self.client

    def get_model_name(self) -> str:
        return f"Cloudflare Workers AI - {self.model}"

    def generate(self, prompt: str, schema: Any | None = None) -> Any:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um avaliador de respostas. "
                        "Responda exatamente no formato solicitado pelo DeepEval."
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
                        "Responda exatamente no formato solicitado pelo DeepEval."
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
                "O modelo da Cloudflare não retornou JSON válido para o DeepEval. "
                f"Resposta recebida: {text}"
            ) from exc

    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("Nenhum JSON encontrado na resposta do modelo.")

        return text[start : end + 1]