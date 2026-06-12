from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SMART_REPO = ROOT_DIR.parent / "smart"
DEFAULT_OUT = (
    DEFAULT_SMART_REPO
    / "apps"
    / "api"
    / "src"
    / "features"
    / "ai"
    / "ai.response-snapshot.generated.ts"
)


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def split_pipe(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def parse_json_cell(value: str | None) -> dict[str, Any]:
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_entry(row: dict[str, str]) -> dict[str, Any]:
    question = normalize_question(row.get("question", ""))
    trace = parse_json_cell(row.get("trace_json"))
    rag = parse_json_cell(row.get("rag_json"))
    agents_called = split_pipe(row.get("agents_called"))
    tools_called = split_pipe(row.get("tools_called"))
    route = row.get("route") or trace.get("route") or "snapshot-replay"

    if tools_called:
        trace["toolsCalled"] = tools_called
        trace["mcpTools"] = tools_called
    if agents_called:
        trace["agentsCalled"] = agents_called

    agent_results = [
        {
            "agentName": agent,
            "agentId": agent.lower(),
            "summary": "Replay de snapshot de resposta coletada para avaliacao.",
            "evidence": {
                "mcpTools": tools_called,
            },
        }
        for agent in agents_called
    ]

    return {
        "numero": int(row["numero"]) if str(row.get("numero", "")).isdigit() else row.get("numero"),
        "conversationId": row.get("conversation_id") or None,
        "question": question,
        "response": row.get("actual_output") or "",
        "route": route,
        "agentsCalled": agents_called,
        "toolsCalled": tools_called,
        "trace": trace,
        "rag": rag,
        "agentResults": agent_results,
    }


def write_typescript_snapshot(entries: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entries, ensure_ascii=False, indent=2)
    output.write_text(
        "\n".join(
            [
                "/* eslint-disable */",
                "// Arquivo gerado por DeepEval/scripts/build_response_snapshot.py.",
                "// Nao edite manualmente; gere novamente a partir de um *_response.csv.",
                "",
                "import type { ResponseSnapshotEntry } from './ai.response-snapshot';",
                "",
                f"export const generatedResponseSnapshot = {payload} satisfies readonly ResponseSnapshotEntry[];",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera snapshot replayable da API SMART a partir de um *_response.csv."
    )
    parser.add_argument("--response-csv", type=Path, required=True, help="CSV exportado pelo DeepEval.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Arquivo TS gerado no repo smart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.response_csv.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    entries = [build_entry(row) for row in rows if normalize_question(row.get("question", ""))]
    if not entries:
        raise ValueError(f"Nenhuma linha valida encontrada em {args.response_csv}.")

    write_typescript_snapshot(entries, args.out)
    print(f"Snapshot gerado: {args.out}")
    print(f"Entradas: {len(entries)}")


if __name__ == "__main__":
    main()
