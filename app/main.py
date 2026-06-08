from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall

from app.api import (
    ConfidentAiConfig,
    call_smart_chat,
    configure_confident_ai_environment,
    configure_judge_environment,
    evaluate_with_confident_ai,
    extract_agents_called,
    extract_tools_called,
    load_confident_ai_config,
    load_api_config,
)
from app.tools import (
    SMART_AGENT_TOOLS,
    build_goal_accuracy_case,
    build_goal_accuracy_metric,
    build_observed_smart_agent,
    build_task_completion_metric,
    build_tool_correctness_case,
    build_tool_correctness_metric,
    infer_expected_tools,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_FILE = ROOT_DIR / "app" / "db" / "db.json"


@dataclass(frozen=True)
class EvalSample:
    question: str
    expected_output: str
    number: int | None = None
    expected_tools: list[str] = field(default_factory=list)


def _find_numbered_value(item: dict[str, Any], prefix: str) -> str | None:
    for key, value in item.items():
        if key.startswith(prefix) and isinstance(value, str):
            return value
    return None


def load_dataset(path: Path = DEFAULT_DATASET_FILE) -> list[EvalSample]:
    with path.open(encoding="utf-8") as file:
        data: Any = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{path} deve conter uma lista JSON.")

    samples: list[EvalSample] = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            samples.append(EvalSample(question=item.strip(), expected_output="", number=index))
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Item {index} de {path} deve ser string ou objeto JSON.")

        question = (
            item.get("question")
            or item.get("pergunta")
            or item.get("input")
            or _find_numbered_value(item, "pergunta_")
        )
        expected_output = (
            item.get("expected_output")
            or item.get("resposta_completa")
            or item.get("resposta")
            or item.get("answer")
            or item.get("expected")
            or _find_numbered_value(item, "resposta_")
            or ""
        )
        expected_tools = item.get("expected_tools") or item.get("ferramentas_esperadas") or []
        number = item.get("numero") or item.get("number") or index

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Item {index} de {path} nao contem pergunta valida.")
        if not isinstance(expected_output, str):
            raise ValueError(f"Item {index} de {path} contem resposta esperada invalida.")
        if not isinstance(expected_tools, list) or not all(isinstance(tool, str) for tool in expected_tools):
            raise ValueError(f"Item {index} de {path} contem expected_tools invalido.")

        samples.append(
            EvalSample(
                question=question.strip(),
                expected_output=expected_output.strip(),
                number=number if isinstance(number, int) else index,
                expected_tools=[tool.strip() for tool in expected_tools if tool.strip()],
            )
        )

    if not samples:
        raise ValueError(f"{path} nao contem casos de avaliacao.")

    return samples


def collect_smart_responses(samples: list[EvalSample]):
    api_config = load_api_config()
    collected = []

    print(f"Coletando respostas da API SMART para {len(samples)} cenarios...\n")

    for index, sample in enumerate(samples, start=1):
        conversation_id = f"deepeval-smart-{index}"
        print(f"  [{index}/{len(samples)}] {sample.question[:90]}")

        try:
            answer, payload = call_smart_chat(sample.question, conversation_id, api_config)
        except Exception as error:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if status_code in {401, 403}:
                raise RuntimeError(
                    "Falha de autenticacao ao chamar a API SMART. "
                    "Confira HONO_TOKEN, HONO_COOKIE e HONO_ORIGIN no .env."
                ) from error

            print(f"      [AVISO] Falha ao chamar API: {error}")
            answer, payload = "Erro ao consultar a API SMART.", {}

        agents_called = extract_agents_called(payload)
        tools_called = extract_tools_called(payload)
        if not tools_called:
            tools_called = infer_tools_from_agents(agents_called)
        route = payload.get("trace", {}).get("route", "desconhecida")

        print(f"      route={route} | agentes={agents_called} | tools={[tool.name for tool in tools_called]}")

        collected.append(
            {
                "sample": sample,
                "answer": answer,
                "payload": payload,
                "tools_called": tools_called,
                "conversation_id": conversation_id,
            }
        )

    return collected


def infer_tools_from_agents(agents_called: list[str]) -> list[ToolCall]:
    tool_names: list[str] = []
    for agent_name in agents_called:
        normalized = agent_name.strip().lower()
        for known_agent, mapped_tools in SMART_AGENT_TOOLS.items():
            if normalized in {known_agent, known_agent.lower()} or normalized == known_agent.lower():
                for tool_name in mapped_tools:
                    if tool_name not in tool_names:
                        tool_names.append(tool_name)

    return [ToolCall(name=tool_name) for tool_name in tool_names]


def _expected_tool_calls(sample: EvalSample) -> list[ToolCall] | None:
    if not sample.expected_tools:
        return None
    return [ToolCall(name=tool_name) for tool_name in sample.expected_tools]


def _tool_names(tools: list[ToolCall]) -> list[str]:
    return [tool.name for tool in tools]


def _join(values: list[str]) -> str:
    return "|".join(values)


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _simple_tool_score(expected_tools: list[str], called_tools: list[str]) -> float:
    if not expected_tools:
        return 1.0 if called_tools else 0.0

    expected = set(expected_tools)
    called = set(called_tools)
    true_positive = len(expected & called)
    false_positive = len(called - expected)
    false_negative = len(expected - called)
    denominator = true_positive + false_positive + false_negative

    return round(true_positive / denominator, 4) if denominator else 1.0


def export_collected_csv(collected, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "numero",
        "conversation_id",
        "route",
        "agents_called",
        "tools_called",
        "expected_tools",
        "simple_tool_score",
        "question",
        "expected_output",
        "actual_output",
        "trace_json",
        "rag_json",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in collected:
            sample = item["sample"]
            payload = item["payload"]
            agents_called = extract_agents_called(payload)
            called_tools = _tool_names(item["tools_called"])
            expected_tool_calls = _expected_tool_calls(sample) or []
            expected_tools = _tool_names(expected_tool_calls)

            if not expected_tools:
                expected_tools = _tool_names(infer_expected_tools(sample.question, sample.expected_output))

            writer.writerow(
                {
                    "numero": sample.number or "",
                    "conversation_id": item["conversation_id"],
                    "route": payload.get("trace", {}).get("route", payload.get("route", "")),
                    "agents_called": _join(agents_called),
                    "tools_called": _join(called_tools),
                    "expected_tools": _join(expected_tools),
                    "simple_tool_score": _simple_tool_score(expected_tools, called_tools),
                    "question": sample.question,
                    "expected_output": sample.expected_output,
                    "actual_output": item["answer"],
                    "trace_json": _json_cell(payload.get("trace", {})),
                    "rag_json": _json_cell(payload.get("rag", {})),
                }
            )

    print(f"\nCSV exportado em: {output_path}")


def run_goal_accuracy(collected, judge_config, confident_config: ConfidentAiConfig) -> None:
    metric = build_goal_accuracy_metric(judge_config)
    test_cases = [
        build_goal_accuracy_case(
            item["sample"].question,
            item["answer"],
            item["sample"].expected_output,
            item["tools_called"],
        )
        for item in collected
    ]

    print("\n[1/3] Avaliando Goal Accuracy...")
    evaluate_with_confident_ai(
        test_cases=test_cases,
        metrics=[metric],
        metric_name="goal_accuracy",
        config=confident_config,
        hyperparameters={
            "judge_model": judge_config.model_name,
            "judge_base_url": judge_config.base_url,
            "threshold": judge_config.threshold,
            "dataset_size": len(test_cases),
        },
    )


def run_tool_correctness(collected, judge_config, confident_config: ConfidentAiConfig) -> None:
    metric = build_tool_correctness_metric(judge_config)
    test_cases = [
        build_tool_correctness_case(
            item["sample"].question,
            item["answer"],
            item["sample"].expected_output,
            item["tools_called"],
            expected_tools=_expected_tool_calls(item["sample"]),
        )
        for item in collected
    ]

    print("\n[2/3] Avaliando Tool Correctness...")
    evaluate_with_confident_ai(
        test_cases=test_cases,
        metrics=[metric],
        metric_name="tool_correctness",
        config=confident_config,
        hyperparameters={
            "judge_model": judge_config.model_name,
            "judge_base_url": judge_config.base_url,
            "threshold": judge_config.threshold,
            "dataset_size": len(test_cases),
        },
    )


def run_task_completion(samples: list[EvalSample], judge_config, confident_config: ConfidentAiConfig) -> None:
    configure_confident_ai_environment(confident_config)
    metric = build_task_completion_metric(judge_config)
    api_config = load_api_config()
    observed_smart_agent = build_observed_smart_agent(api_config, metric)
    dataset = EvaluationDataset(
        goldens=[
            Golden(input=sample.question, expected_output=sample.expected_output)
            for sample in samples
        ]
    )

    print("\n[3/3] Avaliando Task Completion via tracing...")
    for index, golden in enumerate(dataset.evals_iterator(metrics=[metric]), start=1):
        observed_smart_agent(
            golden.input,
            f"deepeval-task-completion-{index}",
            golden.expected_output or "",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia a API SMART com DeepEval: Goal Accuracy, Tool Correctness e Task Completion."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="JSON de perguntas/respostas esperadas. Padrao: app/db/db.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita a quantidade de casos carregados do dataset.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Exporta respostas e metadados coletados em CSV para analise posterior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT_DIR / ".env")
    judge_config = configure_judge_environment()
    confident_config = load_confident_ai_config()
    samples = load_dataset(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    collected = collect_smart_responses(samples)

    if args.export_csv:
        export_collected_csv(collected, args.export_csv)

    run_goal_accuracy(collected, judge_config, confident_config)
    run_tool_correctness(collected, judge_config, confident_config)
    run_task_completion(samples, judge_config, confident_config)

    if args.export_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_collected_csv(collected, ROOT_DIR / "outputs" / f"smart_deepeval_{timestamp}.csv")

    print("\nAvaliacao concluida.")


if __name__ == "__main__":
    main()
