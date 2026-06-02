from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import requests
from deepeval.test_case import ToolCall


@dataclass(frozen=True)
class SmartApiConfig:
    base_url: str
    chat_path: str
    token: str | None
    timeout: int


def load_api_config() -> SmartApiConfig:
    base_url = os.getenv("HONO_BASE_URL", "http://127.0.0.1:8787").rstrip("/")
    chat_path = os.getenv("HONO_CHAT_PATH", "/v1/ai/chat")
    if not chat_path.startswith("/"):
        chat_path = f"/{chat_path}"
    token = os.getenv("HONO_TOKEN")
    timeout = int(os.getenv("HONO_TIMEOUT", "60"))
    return SmartApiConfig(base_url=base_url, chat_path=chat_path, token=token, timeout=timeout)


def call_smart_chat(question: str, conversation_id: str, config: SmartApiConfig | None = None) -> tuple[str, dict[str, Any]]:
    config = config or load_api_config()
    headers: dict[str, str] = {}
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"

    response = requests.post(
        f"{config.base_url}{config.chat_path}",
        json={
            "conversationId": conversation_id,
            "messages": [{"role": "user", "content": question}],
        },
        headers=headers,
        timeout=config.timeout,
    )
    response.raise_for_status()

    payload: dict[str, Any] = response.json()
    return extract_answer(payload), payload


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
        for key in ("mcpTools", "toolsCalled", "toolCalls", "tools_called", "tools"):
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
        if isinstance(agent_name, str) and agent_name and agent_name not in seen:
            seen.add(agent_name)
            agents.append(agent_name)

    return agents
