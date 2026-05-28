from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv(".env.local")
load_dotenv(".env")


def _get_by_path(data: Any, path: str) -> Any:
    current = data

    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue

        raise KeyError(path)

    return current


def _extract_output(data: Any) -> str:
    preferred_path = os.getenv("SMART_RESPONSE_JSON_PATH", "data.answer").strip()

    candidate_paths = [
        preferred_path,
        "data.answer",
        "data.response",
        "data.message",
        "answer",
        "response",
        "message",
        "output",
        "result",
    ]

    for path in candidate_paths:
        if not path:
            continue

        try:
            value = _get_by_path(data, path)

            if isinstance(value, str):
                return value

            if value is not None:
                return str(value)

        except KeyError:
            continue

    raise ValueError(
        "Não encontrei a resposta no JSON da API. "
        "Ajuste SMART_RESPONSE_JSON_PATH no .env.local."
    )


def ask_smart_api(
    question: str,
    endpoint: str | None = None,
    method: str | None = None,
) -> tuple[str, dict[str, Any]]:
    base_url = os.getenv("SMART_API_BASE_URL", "http://127.0.0.1:8787").rstrip("/")
    api_endpoint = endpoint or os.getenv("SMART_API_ENDPOINT", "/v1/orchestrator")
    http_method = (method or os.getenv("SMART_API_METHOD", "POST")).upper()
    timeout = int(os.getenv("SMART_API_TIMEOUT", "120"))
    request_field = os.getenv("SMART_REQUEST_FIELD", "message")

    url = f"{base_url}/{api_endpoint.lstrip('/')}"

    headers = {
        "Content-Type": "application/json",
    }

    auth_token = os.getenv("SMART_AUTH_TOKEN", "").strip()

    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    if http_method == "GET":
        response = requests.get(
            url,
            params={request_field: question},
            headers=headers,
            timeout=timeout,
        )
    else:
        response = requests.request(
            http_method,
            url,
            json={request_field: question},
            headers=headers,
            timeout=timeout,
        )

    if not response.ok:
        raise RuntimeError(
            f"Erro ao chamar API SMART: {response.status_code} - {response.text}"
        )

    try:
        raw_json = response.json()
    except ValueError as exc:
        raise ValueError(f"A API não retornou JSON válido: {response.text}") from exc

    actual_output = _extract_output(raw_json)

    return actual_output, raw_json