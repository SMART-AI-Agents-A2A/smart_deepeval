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
DEFAULT_OUTPUTS_DIR = ROOT_DIR / "outputs"


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


def select_top_samples_by_csv(samples: list[EvalSample], csv_path: Path, limit: int) -> list[EvalSample]:
    if limit <= 0:
        return samples

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    ranked_numbers: list[int] = []
    for row in sorted(rows, key=lambda item: float(item.get("simple_tool_score") or 0), reverse=True):
        try:
            number = int(row.get("numero") or "")
        except ValueError:
            continue
        if number not in ranked_numbers:
            ranked_numbers.append(number)

    selected_numbers = set(ranked_numbers[:limit])
    return [sample for sample in samples if sample.number in selected_numbers]


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


def _base_metric_row(
    *,
    metric_name: str,
    score: float | None,
    threshold: float,
    passed: bool | None,
    reason: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    sample = item["sample"]
    payload = item["payload"]
    agents_called = extract_agents_called(payload)
    called_tools = _tool_names(item["tools_called"])
    expected_tool_calls = _expected_tool_calls(sample) or infer_expected_tools(
        sample.question,
        sample.expected_output,
    )
    expected_tools = _tool_names(expected_tool_calls)

    return {
        "numero": sample.number or "",
        "metric_name": metric_name,
        "score": "" if score is None else round(float(score), 4),
        "threshold": threshold,
        "passed": "" if passed is None else bool(passed),
        "reason": reason,
        "question": sample.question,
        "actual_output": item["answer"],
        "expected_output": sample.expected_output,
        "route": payload.get("trace", {}).get("route", payload.get("route", "")),
        "agents_called": _join(agents_called),
        "tools_called": _join(called_tools),
        "expected_tools": _join(expected_tools),
    }


def _get_attr(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _metric_data_rows(evaluation_result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_results = _get_attr(evaluation_result, "test_results") or []

    for test_result in test_results:
        input_text = _get_attr(test_result, "input") or ""
        actual_output = _get_attr(test_result, "actual_output") or ""
        expected_output = _get_attr(test_result, "expected_output") or ""
        metrics_data = _get_attr(test_result, "metrics_data") or []

        if not input_text:
            turns = _get_attr(test_result, "turns") or []
            for turn in turns:
                role = _get_attr(turn, "role")
                content = _get_attr(turn, "content") or ""
                if role == "user" and not input_text:
                    input_text = content
                elif role == "assistant" and not actual_output:
                    actual_output = content

        for metric_data in metrics_data:
            rows.append(
                {
                    "metric_name": _get_attr(metric_data, "name") or "",
                    "score": _get_attr(metric_data, "score"),
                    "threshold": _get_attr(metric_data, "threshold"),
                    "passed": _get_attr(metric_data, "success"),
                    "reason": _get_attr(metric_data, "reason") or _get_attr(metric_data, "error") or "",
                    "question": input_text,
                    "actual_output": actual_output,
                    "expected_output": expected_output,
                }
            )

    return rows


def _merge_metric_rows(metric_rows: list[dict[str, Any]], collected) -> list[dict[str, Any]]:
    by_question = {item["sample"].question: item for item in collected}
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for metric_row in metric_rows:
        question = str(metric_row.get("question") or "")
        item = by_question.get(question)
        if item is None:
            continue

        metric_name = str(metric_row.get("metric_name") or "")
        key = (metric_name, question)
        if key in seen:
            continue
        seen.add(key)

        threshold = metric_row.get("threshold")
        if threshold is None:
            threshold = 0.0
        merged.append(
            _base_metric_row(
                metric_name=metric_name,
                score=metric_row.get("score"),
                threshold=float(threshold),
                passed=metric_row.get("passed"),
                reason=str(metric_row.get("reason") or ""),
                item=item,
            )
        )

    return merged


def fallback_tool_metric_rows(collected, threshold: float) -> list[dict[str, Any]]:
    rows = []
    for item in collected:
        sample = item["sample"]
        called_tools = _tool_names(item["tools_called"])
        expected_tool_calls = _expected_tool_calls(sample) or infer_expected_tools(
            sample.question,
            sample.expected_output,
        )
        expected_tools = _tool_names(expected_tool_calls)
        score = _simple_tool_score(expected_tools, called_tools)
        rows.append(
            _base_metric_row(
                metric_name="Tool Correctness",
                score=score,
                threshold=threshold,
                passed=score >= threshold,
                reason="Score local calculado por cobertura Jaccard entre expected_tools e tools_called.",
                item=item,
            )
        )
    return rows


def export_metric_results_csv(metric_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "numero",
        "metric_name",
        "score",
        "threshold",
        "passed",
        "reason",
        "question",
        "actual_output",
        "expected_output",
        "route",
        "agents_called",
        "tools_called",
        "expected_tools",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"\nCSV de metricas exportado em: {output_path}")


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


def run_goal_accuracy(collected, judge_config, confident_config: ConfidentAiConfig):
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
    return evaluate_with_confident_ai(
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


def run_tool_correctness(collected, judge_config, confident_config: ConfidentAiConfig):
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
    return evaluate_with_confident_ai(
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


def run_task_completion(samples: list[EvalSample], judge_config, confident_config: ConfidentAiConfig):
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
    iterator = dataset.evals_iterator(metrics=[metric])
    index = 1
    while True:
        try:
            golden = next(iterator)
        except StopIteration as finished:
            return finished.value

        observed_smart_agent(
            golden.input,
            f"deepeval-task-completion-{index}",
            golden.expected_output or "",
        )
        index += 1


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
    parser.add_argument(
        "--export-metrics-csv",
        type=Path,
        default=None,
        help="Exporta scores oficiais por metrica/pergunta em CSV.",
    )
    parser.add_argument(
        "--top-tool-score",
        type=int,
        default=None,
        help="Seleciona as N melhores perguntas com base no simple_tool_score de um CSV anterior.",
    )
    parser.add_argument(
        "--top-source-csv",
        type=Path,
        default=None,
        help="CSV anterior usado por --top-tool-score. Padrao: outputs/smart_deepeval_90.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT_DIR / ".env", override=True)
    judge_config = configure_judge_environment()
    confident_config = load_confident_ai_config()
    samples = load_dataset(args.dataset)
    if args.top_tool_score is not None:
        source_csv = args.top_source_csv or DEFAULT_OUTPUTS_DIR / "smart_deepeval_90.csv"
        samples = select_top_samples_by_csv(samples, source_csv, args.top_tool_score)
    if args.limit is not None:
        samples = samples[: args.limit]
    collected = collect_smart_responses(samples)

    if args.export_csv:
        export_collected_csv(collected, args.export_csv)

    all_metric_rows: list[dict[str, Any]] = []

    goal_result = run_goal_accuracy(collected, judge_config, confident_config)
    all_metric_rows.extend(_merge_metric_rows(_metric_data_rows(goal_result), collected))

    tool_result = run_tool_correctness(collected, judge_config, confident_config)
    tool_rows = _merge_metric_rows(_metric_data_rows(tool_result), collected)
    all_metric_rows.extend(tool_rows or fallback_tool_metric_rows(collected, judge_config.threshold))

    task_result = run_task_completion(samples, judge_config, confident_config)
    all_metric_rows.extend(_merge_metric_rows(_metric_data_rows(task_result), collected))

    if args.export_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_collected_csv(collected, ROOT_DIR / "outputs" / f"smart_deepeval_{timestamp}.csv")

    metrics_csv = args.export_metrics_csv
    if metrics_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        metrics_csv = DEFAULT_OUTPUTS_DIR / f"smart_deepeval_metrics_{timestamp}.csv"
    export_metric_results_csv(all_metric_rows, metrics_csv)

    print("\nAvaliacao concluida.")


if __name__ == "__main__":
    main()
