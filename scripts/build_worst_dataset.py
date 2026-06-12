from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _matches_metric(metric_name: str, wanted: str) -> bool:
    return wanted.lower() in metric_name.lower()


def _read_worst_numbers(metrics_path: Path, metric: str, limit: int) -> list[int]:
    rows: list[tuple[float, int]] = []
    with metrics_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if not _matches_metric(row.get("metric_name") or "", metric):
                continue
            raw_number = str(row.get("numero") or "").strip()
            raw_score = str(row.get("score") or "").strip().replace(",", ".")
            if not raw_number or not raw_score:
                continue
            try:
                rows.append((float(raw_score), int(raw_number)))
            except ValueError:
                continue

    numbers: list[int] = []
    for _, number in sorted(rows, key=lambda item: (item[0], item[1])):
        if number not in numbers:
            numbers.append(number)
        if len(numbers) >= limit:
            break

    return numbers


def _load_dataset_by_number(dataset_path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset deve conter uma lista JSON.")

    by_number: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        number = item.get("numero") or item.get("number") or index
        if isinstance(number, int):
            by_number[number] = item

    return by_number


def build_worst_dataset(
    *,
    dataset_path: Path,
    metrics_path: Path,
    output_path: Path,
    metric: str,
    limit: int,
) -> None:
    numbers = _read_worst_numbers(metrics_path, metric, limit)
    if not numbers:
        raise ValueError(f"Nenhuma linha encontrada para metric={metric!r} em {metrics_path}.")

    by_number = _load_dataset_by_number(dataset_path)
    selected = [by_number[number] for number in numbers if number in by_number]

    if not selected:
        raise ValueError("Nenhuma questao do CSV foi encontrada no dataset.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dataset reduzido criado em: {output_path}")
    print("Questoes:", ", ".join(str(item.get("numero")) for item in selected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria dataset pequeno com as piores questoes de um CSV de metricas."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset base JSON.")
    parser.add_argument("--metrics", type=Path, required=True, help="CSV de metricas.")
    parser.add_argument("--output", type=Path, required=True, help="JSON reduzido de saida.")
    parser.add_argument("--metric", default="G-Eval", help="Nome parcial da metrica.")
    parser.add_argument("--limit", type=int, default=8, help="Quantidade de questoes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_worst_dataset(
        dataset_path=args.dataset,
        metrics_path=args.metrics,
        output_path=args.output,
        metric=args.metric,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
