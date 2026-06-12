from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _split_tools(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]


def _read_tool_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_number: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                metric_name = (row.get("metric_name") or "").lower()
                if metric_name and "tool" not in metric_name:
                    continue

                number = str(row.get("numero") or "").strip()
                if not number:
                    continue

                item = by_number.setdefault(
                    number,
                    {
                        "numero": number,
                        "question": row.get("question") or "",
                        "expected_tools": _split_tools(row.get("expected_tools")),
                        "missing": Counter(),
                        "extra": Counter(),
                        "called": Counter(),
                        "files": set(),
                    },
                )
                item["files"].add(path.name)
                item["question"] = item["question"] or row.get("question") or ""
                if not item["expected_tools"]:
                    item["expected_tools"] = _split_tools(row.get("expected_tools"))
                item["missing"].update(_split_tools(row.get("tools_missing")))
                item["extra"].update(_split_tools(row.get("tools_extra")))
                item["called"].update(_split_tools(row.get("tools_called")))
    return by_number


def review(paths: list[Path], output_path: Path) -> None:
    rows = _read_tool_rows(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "numero",
        "question",
        "expected_tools",
        "missing_counts",
        "extra_counts",
        "called_counts",
        "suggestion",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for number in sorted(rows, key=lambda value: int(value) if value.isdigit() else value):
            row = rows[number]
            expected = set(row["expected_tools"])
            missing = row["missing"]
            extra = row["extra"]
            suggestion_parts: list[str] = []
            if missing:
                suggestion_parts.append("Verificar se expected_tools esta amplo demais ou se a API deixou de chamar ferramenta obrigatoria.")
            if extra:
                suggestion_parts.append("Classificar cada extra como optional_tools ou corrigir roteador para nao chamar.")
            if not missing and not extra:
                suggestion_parts.append("Contrato de ferramentas esta aderente nos CSVs analisados.")

            writer.writerow(
                {
                    "numero": number,
                    "question": row["question"],
                    "expected_tools": "|".join(row["expected_tools"]),
                    "missing_counts": "|".join(f"{tool}:{count}" for tool, count in missing.most_common()),
                    "extra_counts": "|".join(f"{tool}:{count}" for tool, count in extra.most_common() if tool not in expected),
                    "called_counts": "|".join(f"{tool}:{count}" for tool, count in row["called"].most_common()),
                    "suggestion": " ".join(suggestion_parts),
                }
            )

    print(f"Revisao de ferramentas criada em: {output_path}")
    print(f"Questoes analisadas: {len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolida tools_missing/tools_extra dos CSVs para revisar expected_tools e optional_tools."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=Path,
        required=True,
        help="Um ou mais CSVs de metricas gerados pelo DeepEval.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tool_contract_review.csv"),
        help="CSV de revisao gerado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review(args.metrics, args.output)


if __name__ == "__main__":
    main()
