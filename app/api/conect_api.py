from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests
from deepeval.test_case import ToolCall


SMART_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
TOOL_LIST_KEYS = ("mcpTools", "toolsCalled", "toolCalls", "tools_called")


@dataclass(frozen=True)
class SmartApiConfig:
    base_url: str
    chat_path: str
    token: str | None
    cookie: str | None
    origin: str | None
    timeout: int
    retry_attempts: int
    retry_base_sleep: float
    retry_max_sleep: float


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def load_api_config() -> SmartApiConfig:
    base_url = os.getenv("HONO_BASE_URL", "http://127.0.0.1:8787").rstrip("/")
    chat_path = os.getenv("HONO_CHAT_PATH", "/v1/ai/chat")
    if not chat_path.startswith("/"):
        chat_path = f"/{chat_path}"
    token = os.getenv("HONO_TOKEN")
    cookie = os.getenv("HONO_COOKIE")
    origin = os.getenv("HONO_ORIGIN", "http://localhost:5173")
    timeout = _env_int("HONO_TIMEOUT", 60)
    return SmartApiConfig(
        base_url=base_url,
        chat_path=chat_path,
        token=token,
        cookie=cookie,
        origin=origin,
        timeout=timeout,
        retry_attempts=_env_int("HONO_RETRY_ATTEMPTS", 4),
        retry_base_sleep=_env_float("HONO_RETRY_BASE_SLEEP", 2.0),
        retry_max_sleep=_env_float("HONO_RETRY_MAX_SLEEP", 60.0),
    )


def _compute_sleep(attempt: int, retry_after: str | None, base: float, maximum: float) -> float:
    if retry_after:
        try:
            return min(float(retry_after) + random.uniform(0, 1), maximum)
        except ValueError:
            pass
    exponential = base * (2 ** (attempt - 1))
    return min(exponential + random.uniform(0, base), maximum)


def call_smart_chat(question: str, conversation_id: str, config: SmartApiConfig | None = None) -> tuple[str, dict[str, Any]]:
    config = config or load_api_config()
    headers: dict[str, str] = {}
    if config.origin:
        headers["Origin"] = config.origin
    if config.cookie:
        headers["Cookie"] = config.cookie
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"

    url = f"{config.base_url}{config.chat_path}"
    body = {
        "conversationId": conversation_id,
        "messages": [{"role": "user", "content": question}],
    }

    last_error: Exception | None = None
    for attempt in range(1, config.retry_attempts + 1):
        try:
            response = requests.post(url, json=body, headers=headers, timeout=config.timeout)
        except requests.RequestException as exc:
            last_error = exc
            if attempt < config.retry_attempts:
                sleep = _compute_sleep(attempt, None, config.retry_base_sleep, config.retry_max_sleep)
                print(
                    f"[AVISO SMART] conexao_falhou conv={conversation_id} "
                    f"attempt={attempt}/{config.retry_attempts} sleep={sleep:.2f}s "
                    f"erro={type(exc).__name__}"
                )
                time.sleep(sleep)
                continue
            raise

        if response.status_code in {401, 403}:
            response.raise_for_status()

        if response.status_code in SMART_RETRYABLE_STATUS and attempt < config.retry_attempts:
            sleep = _compute_sleep(
                attempt,
                response.headers.get("Retry-After"),
                config.retry_base_sleep,
                config.retry_max_sleep,
            )
            print(
                f"[AVISO SMART] status={response.status_code} conv={conversation_id} "
                f"attempt={attempt}/{config.retry_attempts} sleep={sleep:.2f}s"
            )
            last_error = requests.HTTPError(response=response)
            time.sleep(sleep)
            continue

        response.raise_for_status()
        payload = parse_chat_response(response)
        return extract_answer(payload), payload

    if last_error is not None:
        raise last_error
    raise RuntimeError("Falha ao chamar a API SMART sem erro registrado.")


def parse_chat_response(response: requests.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return parse_sse_response(response.text)
    try:
        return response.json()
    except ValueError:
        text = response.text.strip()
        return {
            "response": text,
            "raw_response": text,
            "content_type": content_type,
        }


def parse_sse_response(text: str) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    full_response_parts: list[str] = []
    latest_trace: dict[str, Any] | None = None
    done_payload: dict[str, Any] | None = None
    error_messages: list[str] = []

    for block in text.replace("\r\n", "\n").split("\n\n"):
        block = block.strip()
        if not block:
            continue

        event_name = "message"
        data_lines: list[str] = []

        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

        if not data_lines:
            continue

        raw_data = "\n".join(data_lines)
        try:
            data: Any = json.loads(raw_data)
        except ValueError:
            data = raw_data

        events.append({"event": event_name, "data": data})

        if event_name == "delta" and isinstance(data, dict) and isinstance(data.get("delta"), str):
            full_response_parts.append(data["delta"])
        elif event_name == "trace" and isinstance(data, dict):
            latest_trace = data
        elif event_name == "done" and isinstance(data, dict):
            done_payload = data
        elif event_name == "error" and isinstance(data, dict) and isinstance(data.get("message"), str):
            error_messages.append(data["message"])

    payload = dict(done_payload or {})
    if latest_trace and "trace" not in payload:
        payload["trace"] = latest_trace

    response_text = payload.get("response")
    if not isinstance(response_text, str) or not response_text.strip():
        fallback = "".join(full_response_parts).strip()
        if not fallback:
            fallback = build_agent_evidence_response(payload, latest_trace)
        if not fallback and error_messages:
            fallback = "Erro retornado pela API SMART: " + " | ".join(error_messages)
        payload["response"] = fallback

    payload["events"] = events
    payload["raw_response"] = text
    return payload


def build_agent_evidence_response(
    payload: dict[str, Any],
    trace: dict[str, Any] | None,
) -> str:
    agent_results = payload.get("agentResults")
    if not isinstance(agent_results, list):
        return ""

    lines: list[str] = []
    route = trace.get("route") if isinstance(trace, dict) else payload.get("route")
    if isinstance(route, str):
        lines.append(f"Rota usada: {route}.")

    for result in agent_results:
        if not isinstance(result, dict):
            continue
        agent_name = result.get("agentName") or result.get("agentId")
        summary = result.get("summary")
        action = result.get("action")
        if isinstance(agent_name, str):
            lines.append(f"Agente acionado: {agent_name}.")
        if isinstance(action, str):
            lines.append(f"Acao: {action}.")
        if isinstance(summary, str):
            lines.append(summary)

        evidence = result.get("evidence")
        if isinstance(evidence, dict):
            tools = evidence.get("mcpTools")
            if isinstance(tools, list) and tools:
                lines.append("Ferramentas MCP: " + ", ".join(str(tool) for tool in tools) + ".")
            values = evidence.get("values")
            if isinstance(values, list):
                for value in values[:12]:
                    if not isinstance(value, dict):
                        continue
                    label = value.get("label")
                    amount = value.get("value")
                    unit = value.get("unit")
                    timestamp = value.get("timestamp")
                    if label is not None and amount is not None:
                        suffix = f" {unit}" if isinstance(unit, str) and unit else ""
                        when = f" em {timestamp}" if isinstance(timestamp, str) and timestamp else ""
                        lines.append(f"{label}: {amount}{suffix}{when}.")

    return "\n".join(lines).strip()


def extract_answer(payload: dict[str, Any]) -> str:
    for key in ("response", "answer", "output", "content", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

    return ""


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def _tool_from_value(value: Any) -> ToolCall | None:
    if isinstance(value, str) and value.strip():
        return ToolCall(name=value.strip())

    if not isinstance(value, dict):
        return None

    name = value.get("name") or value.get("toolName") or value.get("functionName")
    function = value.get("function")
    if not name and isinstance(function, dict):
        name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    parameters = (
        value.get("input_parameters")
        or value.get("inputParameters")
        or value.get("arguments")
        or value.get("args")
        or value.get("input")
    )
    output = value.get("output") or value.get("result")

    kwargs: dict[str, Any] = {"name": name.strip()}
    if isinstance(parameters, dict):
        kwargs["input_parameters"] = parameters
    if output is not None:
        kwargs["output"] = output

    return ToolCall(**kwargs)


def extract_tools_called(payload: dict[str, Any]) -> list[ToolCall]:
    tools: list[ToolCall] = []
    seen: set[str] = set()

    for item in _iter_dicts(payload):
        for key in TOOL_LIST_KEYS:
            raw_tools = item.get(key)
            if not isinstance(raw_tools, list):
                continue
            for raw_tool in raw_tools:
                tool = _tool_from_value(raw_tool)
                if not tool or tool.name in seen:
                    continue
                seen.add(tool.name)
                tools.append(tool)

    return tools


def extract_agents_called(payload: dict[str, Any]) -> list[str]:
    agents: list[str] = []
    seen: set[str] = set()

    for item in _iter_dicts(payload):
        agent_name = item.get("agentName") or item.get("agent_name") or item.get("agentId")
        agent = item.get("agent")
        if not agent_name and isinstance(agent, str):
            agent_name = agent
        if not agent_name and isinstance(agent, dict):
            agent_name = agent.get("name") or agent.get("id")
        if isinstance(agent_name, str) and _is_valid_agent_name(agent_name) and agent_name not in seen:
            seen.add(agent_name)
            agents.append(agent_name)

    return agents


def _is_valid_agent_name(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    if re.fullmatch(r"\d+\s+agentes?", normalized, flags=re.IGNORECASE):
        return False
    return True