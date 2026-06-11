from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _clean_optional_env(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    return stripped or None


def _first_env(*names: str) -> str | None:
    for name in names:
        value = _clean_optional_env(os.getenv(name))
        if value:
            return value
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "sim", "s"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_text(value: Any, limit: int | None = None) -> str:
    if limit is None:
        limit = _env_int("DEEPEVAL_DEBUG_PREVIEW_CHARS", 800)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _preview(value: str, limit: int = 800) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3:
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_object(content: str) -> str:
    cleaned = _strip_json_fence(content)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    if start == -1:
        return cleaned
    depth = 0
    in_string = False
    escape = False
    for index, character in enumerate(cleaned[start:], start=start):
        if escape:
            escape = False
            continue
        if character == "\\":
            escape = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return cleaned


def _extract_between(text: str, start: str, end: str) -> str:
    pattern = re.compile(
        rf"{re.escape(start)}\s*(.*?)(?=\n\s*{re.escape(end)}|\Z)",
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    prompt_text = ""
    if isinstance(messages, list):
        prompt_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
    summary: dict[str, Any] = {
        "model": payload.get("model"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "max_completion_tokens": payload.get("max_completion_tokens"),
        "response_format_active": "response_format" in payload,
        "messages_count": len(messages) if isinstance(messages, list) else 0,
        "prompt_chars": len(prompt_text),
        "prompt_preview": _compact_text(prompt_text),
    }
    if _env_bool("DEEPEVAL_DEBUG_PAYLOAD", False):
        summary["payload_full"] = payload
    return summary


def _rate_limit_headers(response: requests.Response) -> dict[str, str]:
    relevant = {}
    for header in response.headers:
        lower = header.lower()
        if any(k in lower for k in ("retry", "ratelimit", "x-ratelimit", "x-rate-limit")):
            relevant[header] = response.headers[header]
    return relevant


def _append_debug_event(event: dict[str, Any]) -> None:
    debug_path = _clean_optional_env(os.getenv("DEEPEVAL_DEBUG_JSON"))
    if not debug_path:
        return
    path = Path(debug_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any]
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                current = {}
        except Exception:
            current = {}
    else:
        current = {}
    current.setdefault("events", [])
    current["events"].append(event)
    current["updated_at"] = _now_iso()
    path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _provider_error_message(response: requests.Response) -> str:
    parsed = _safe_json_loads(response.text)
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(parsed.get("message"), str):
            return parsed["message"]
    return _compact_text(response.text, 300)


def _compute_sleep(
    attempt: int,
    retry_after: str | None,
    base_sleep: float,
    max_sleep: float,
) -> float:
    if retry_after:
        try:
            wait = float(retry_after)
            jitter = random.uniform(0, 1)
            return min(wait + jitter, max_sleep)
        except ValueError:
            pass
    exponential = base_sleep * (2 ** (attempt - 1))
    jitter = random.uniform(0, base_sleep)
    return min(exponential + jitter, max_sleep)


def _deadline_exceeded(started_at: float, total_deadline: float, next_sleep: float) -> bool:
    if total_deadline <= 0:
        return False
    elapsed = time.monotonic() - started_at
    return (elapsed + next_sleep) >= total_deadline


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

    def _uses_cloudflare_gateway(self) -> bool:
        return "gateway.ai.cloudflare.com" in self.base_url

    def _should_send_response_format(self) -> bool:
        configured = _clean_optional_env(os.getenv("DEEPEVAL_JUDGE_RESPONSE_FORMAT"))
        if configured is not None:
            return configured.lower() in {"1", "true", "yes", "on"}
        return not self._uses_cloudflare_gateway()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        gateway_token = _first_env("CF_AIG_TOKEN", "CLOUDFLARE_API_TOKEN")
        if gateway_token:
            headers["cf-aig-authorization"] = f"Bearer {gateway_token}"
        byok_alias = _first_env("CF_AIG_BYOK_ALIAS", "CLOUDFLARE_AI_GATEWAY_BYOK_ALIAS")
        if byok_alias:
            headers["cf-aig-byok-alias"] = byok_alias
        return headers

    def _payload(
        self,
        prompt: str,
        schema: type[BaseModel] | None,
        *,
        force_no_response_format: bool = False,
    ) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        if schema:
            schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Voce e um juiz de avaliacao. Retorne somente JSON valido, "
                        "sem Markdown, sem explicacao e sem texto fora do objeto JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        "O JSON de resposta deve obedecer exatamente este schema:\n"
                        f"{schema_json}"
                    ),
                },
            ]

        max_tokens = _env_int("DEEPEVAL_JUDGE_MAX_TOKENS", 2048)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        if self.model_name.startswith("gpt-5"):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["temperature"] = 0
            payload["max_tokens"] = max_tokens

        if schema and not force_no_response_format and self._should_send_response_format():
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
        max_attempts = _env_int("DEEPEVAL_JUDGE_RETRY_ATTEMPTS", 6)
        base_sleep = float(os.getenv("DEEPEVAL_JUDGE_RETRY_BASE_SLEEP", "2"))
        max_sleep = float(os.getenv("DEEPEVAL_JUDGE_RETRY_MAX_SLEEP", "90"))
        total_deadline = float(os.getenv("DEEPEVAL_JUDGE_TOTAL_DEADLINE", "0"))
        debug_json_path = os.getenv("DEEPEVAL_DEBUG_JSON", "outputs/deepeval_debug.json")
        url = self._chat_completions_url()
        last_error: Exception | None = None
        started_at = time.monotonic()

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                sleep_time = _compute_sleep(attempt, None, base_sleep, max_sleep)
                if attempt >= max_attempts or _deadline_exceeded(started_at, total_deadline, sleep_time):
                    print(
                        f"[AVISO JUIZ] conexao_falhou model={self.model_name} "
                        f"attempt={attempt}/{max_attempts} sem_retry "
                        f"erro={type(exc).__name__}"
                    )
                    break
                print(
                    f"[AVISO JUIZ] conexao_falhou model={self.model_name} "
                    f"attempt={attempt}/{max_attempts} sleep={sleep_time:.2f}s "
                    f"erro={type(exc).__name__}"
                )
                time.sleep(sleep_time)
                continue

            if response.status_code < 400:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if isinstance(content, list):
                        return "".join(
                            part.get("text", "") for part in content if isinstance(part, dict)
                        )
                    if isinstance(content, str):
                        return content
                    for key in ("reasoning_content", "reasoning", "refusal"):
                        value = message.get(key)
                        if isinstance(value, str) and value.strip():
                            return value
                    if "text" in choices[0] and isinstance(choices[0]["text"], str):
                        return choices[0]["text"]
                    return ""
                return data.get("output_text") or data.get("response") or ""

            provider_message = _provider_error_message(response)
            rl_headers = _rate_limit_headers(response)
            retry_after = response.headers.get("Retry-After")
            sleep_time = _compute_sleep(attempt, retry_after, base_sleep, max_sleep)

            _append_debug_event(
                {
                    "type": "judge_http_error",
                    "timestamp": _now_iso(),
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "status_code": response.status_code,
                    "url": url,
                    "model": self.model_name,
                    "provider_message": provider_message,
                    "rate_limit_headers": rl_headers,
                    "response_preview": _compact_text(response.text),
                    "payload_summary": _summarize_payload(payload),
                }
            )

            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt < max_attempts
                and not _deadline_exceeded(started_at, total_deadline, sleep_time)
            ):
                print(
                    f"[AVISO JUIZ] status={response.status_code} model={self.model_name} "
                    f"attempt={attempt}/{max_attempts} sleep={sleep_time:.2f}s "
                    f"message={provider_message} "
                    f"debug={debug_json_path}"
                )
                time.sleep(sleep_time)
                last_error = requests.HTTPError(response=response)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                print(
                    f"[AVISO JUIZ] status={response.status_code} model={self.model_name} "
                    f"attempt={attempt}/{max_attempts} sem_retry "
                    f"message={provider_message} "
                    f"debug={debug_json_path}"
                )
                last_error = requests.HTTPError(response=response)
                break

            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                detail = response.text.strip()
                if response.status_code == 401:
                    raise RuntimeError(
                        "Falha de autenticacao no provedor usado como juiz. "
                        "Confira MODEL_ACCESS_KEY, DIGITALOCEAN_TOKEN, OPENAI_API_KEY ou OPEN_API_KEY. "
                        f"Resposta do provedor: {detail or '401 Unauthorized'}"
                    ) from error
                raise

        if last_error is not None:
            raise last_error
        return ""

    def _parse_schema_response(self, content: str, schema: type[BaseModel]) -> BaseModel:
        cleaned = _extract_json_object(content)
        try:
            return schema.model_validate_json(cleaned)
        except Exception as first_error:
            try:
                return schema.model_validate(json.loads(cleaned))
            except Exception as second_error:
                raise RuntimeError(
                    "O modelo juiz nao retornou JSON valido para o schema exigido pelo DeepEval. "
                    f"Schema: {schema.__name__}. "
                    f"Conteudo recebido: {_preview(content) or '[vazio]'}"
                ) from second_error or first_error

    def _fallback_schema_response(
        self,
        *,
        prompt: str,
        content: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        if schema.__name__ == "TaskAndOutcome":
            task = _extract_between(prompt, "input:", "tools called:")
            outcome = _extract_between(prompt, "response:", "JSON:")
            if not task:
                task = "Avaliar se a API SMART concluiu a tarefa solicitada pelo usuario."
            if not outcome:
                outcome = _extract_between(prompt, "trace:", "JSON:") or (
                    "O juiz nao retornou JSON valido para extrair o resultado da trace."
                )
            return schema.model_validate({"task": task, "outcome": outcome})

        if schema.__name__ == "TaskCompletionVerdict":
            return schema.model_validate(
                {
                    "verdict": 0.0,
                    "reason": (
                        "O modelo juiz nao retornou JSON valido para calcular Task Completion. "
                        f"Conteudo recebido: {_preview(content) or '[vazio]'}"
                    ),
                }
            )

        schema_fields = getattr(schema, "model_fields", {})
        if "score" in schema_fields and "reason" in schema_fields:
            return schema.model_validate(
                {
                    "score": 0.0,
                    "reason": (
                        "O modelo juiz nao retornou JSON valido para este criterio. "
                        f"Conteudo recebido: {_preview(content) or '[vazio]'}"
                    ),
                }
            )

        raise RuntimeError(
            "O modelo juiz nao retornou JSON valido para o schema exigido pelo DeepEval. "
            f"Schema: {schema.__name__}. Conteudo recebido: {_preview(content) or '[vazio]'}"
        )

    def _repair_schema_response(
        self,
        *,
        prompt: str,
        content: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        repair_prompt = (
            "Converta a resposta abaixo para JSON valido que siga exatamente o schema informado. "
            "Retorne somente o JSON, sem Markdown, sem explicacao e sem texto antes ou depois.\n\n"
            f"SCHEMA:\n{json.dumps(schema.model_json_schema(), ensure_ascii=False)}\n\n"
            f"PROMPT ORIGINAL:\n{prompt}\n\n"
            f"RESPOSTA A CONVERTER:\n{content or '[resposta vazia]'}"
        )
        payload = self._payload(repair_prompt, None)
        if self._should_send_response_format():
            payload["response_format"] = {"type": "json_object"}
        repaired_content = self._request(payload)
        if not repaired_content.strip():
            fallback_payload = self._payload(repair_prompt, None, force_no_response_format=True)
            repaired_content = self._request(fallback_payload)
        try:
            return self._parse_schema_response(repaired_content, schema)
        except Exception:
            return self._fallback_schema_response(
                prompt=prompt,
                content=repaired_content,
                schema=schema,
            )

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        payload = self._payload(prompt, schema)
        try:
            content = self._request(payload)
        except requests.HTTPError:
            if not schema:
                raise
            payload = self._payload(prompt, None, force_no_response_format=True)
            if self._should_send_response_format():
                payload["response_format"] = {"type": "json_object"}
            content = self._request(payload)

        if schema:
            if not content.strip() and self._should_send_response_format():
                payload = self._payload(prompt, schema, force_no_response_format=True)
                content = self._request(payload)
            try:
                return self._parse_schema_response(content, schema)
            except Exception:
                try:
                    return self._repair_schema_response(
                        prompt=prompt,
                        content=content,
                        schema=schema,
                    )
                except Exception:
                    return self._fallback_schema_response(
                        prompt=prompt,
                        content=content,
                        schema=schema,
                    )
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
    model_name = os.getenv("DEEPEVAL_JUDGE_MODEL")
    base_url = _clean_optional_env(os.getenv("DEEPEVAL_JUDGE_BASE_URL"))
    api_key = _first_env(
        "MODEL_ACCESS_KEY",
        "DIGITALOCEAN_TOKEN",
        "OPENAI_API_KEY",
        "OPEN_API_KEY",
    )
    timeout = _env_int("DEEPEVAL_JUDGE_TIMEOUT", 120)

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
    judge_api_key = _first_env(
        "MODEL_ACCESS_KEY",
        "DIGITALOCEAN_TOKEN",
        "OPENAI_API_KEY",
        "OPEN_API_KEY",
    )
    if judge_api_key:
        os.environ["OPENAI_API_KEY"] = judge_api_key
    return load_judge_config()