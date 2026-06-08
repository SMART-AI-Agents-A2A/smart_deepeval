from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "analises" / "graficos"


def split_pipe(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []

    return [item.strip() for item in value.split("|") if item.strip()]


def latest_csv() -> Path:
    csv_files = sorted(
        (ROOT_DIR / "outputs").glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError("Nenhum CSV encontrado na pasta outputs.")

    return csv_files[0]


def is_metric_csv(df: pd.DataFrame) -> bool:
    return {"metric_name", "score", "passed"}.issubset(df.columns)


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if is_metric_csv(df):
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce").fillna(0.7)
        df["passed"] = df["passed"].astype(str).str.lower().isin(["true", "1", "yes"])
        df["route"] = df.get("route", "").fillna("").replace("", "sem_rota")
        df["answer_length"] = df.get("actual_output", "").fillna("").astype(str).str.len()
        df["tools_called_list"] = df.get("tools_called", "").apply(split_pipe)
        df["expected_tools_list"] = df.get("expected_tools", "").apply(split_pipe)
        df["agents_called_list"] = df.get("agents_called", "").apply(split_pipe)
        return df

    if "simple_tool_score" in df.columns:
        df["simple_tool_score"] = pd.to_numeric(df["simple_tool_score"], errors="coerce").fillna(0)
    else:
        df["simple_tool_score"] = 0.0

    df["route"] = df.get("route", "").fillna("").replace("", "sem_rota")
    df["answer_length"] = df.get("actual_output", "").fillna("").astype(str).str.len()
    df["tools_called_list"] = df.get("tools_called", "").apply(split_pipe)
    df["expected_tools_list"] = df.get("expected_tools", "").apply(split_pipe)
    df["agents_called_list"] = df.get("agents_called", "").apply(split_pipe)
    df["rag_source_count"] = df.get("rag_json", "").apply(extract_rag_source_count)

    return df


def plot_metric_score_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = sorted(df["metric_name"].dropna().unique())
    if not metrics:
        return

    plt.figure(figsize=(10, 5))
    for metric_name in metrics:
        values = df.loc[df["metric_name"] == metric_name, "score"].dropna()
        if values.empty:
            continue
        plt.hist(values, bins=10, alpha=0.55, label=metric_name)

    plt.axvline(0.7, color="#cc3333", linestyle="--", linewidth=2, label="threshold 0.70")
    plt.title("Distribuicao de Scores por Metrica")
    plt.xlabel("Score")
    plt.ylabel("Quantidade de casos")
    plt.legend()
    save_current(output_dir, "01_metricas_distribuicao_scores.png")


def plot_metric_average_scores(df: pd.DataFrame, output_dir: Path) -> None:
    scores = df.groupby("metric_name")["score"].mean().sort_values(ascending=False)
    ax = scores.plot(kind="bar", figsize=(9, 5), color="#4c78a8")
    plt.axhline(0.7, color="#cc3333", linestyle="--", linewidth=2, label="threshold 0.70")
    plt.title("Score Medio por Metrica")
    plt.xlabel("Metrica")
    plt.ylabel("Score medio")
    plt.ylim(0, 1)
    plt.legend()
    annotate_bars(ax)
    save_current(output_dir, "02_metricas_score_medio.png")


def plot_metric_pass_rate(df: pd.DataFrame, output_dir: Path) -> None:
    pass_rate = (df.groupby("metric_name")["passed"].mean() * 100).sort_values(ascending=False)
    ax = pass_rate.plot(kind="bar", figsize=(9, 5), color="#59a14f")
    plt.title("Pass Rate por Metrica")
    plt.xlabel("Metrica")
    plt.ylabel("Pass rate (%)")
    plt.ylim(0, 100)
    annotate_bars(ax)
    save_current(output_dir, "03_metricas_pass_rate.png")


def plot_metric_pass_fail(df: pd.DataFrame, output_dir: Path) -> None:
    pivot = (
        df.assign(status=df["passed"].map({True: "passou", False: "falhou"}))
        .pivot_table(index="metric_name", columns="status", values="question", aggfunc="count", fill_value=0)
    )
    ax = pivot.plot(kind="bar", stacked=True, figsize=(9, 5), color=["#e15759", "#59a14f"])
    plt.title("Passou vs Falhou por Metrica")
    plt.xlabel("Metrica")
    plt.ylabel("Quantidade de casos")
    annotate_bars(ax)
    save_current(output_dir, "04_metricas_passou_vs_falhou.png")


def plot_metric_score_by_route(df: pd.DataFrame, output_dir: Path) -> None:
    pivot = df.pivot_table(index="route", columns="metric_name", values="score", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    plt.axhline(0.7, color="#cc3333", linestyle="--", linewidth=2, label="threshold 0.70")
    plt.title("Score Medio por Rota e Metrica")
    plt.xlabel("Rota")
    plt.ylabel("Score medio")
    plt.ylim(0, 1)
    plt.legend()
    save_current(output_dir, "05_metricas_score_por_rota.png")


def plot_worst_cases(df: pd.DataFrame, output_dir: Path) -> None:
    worst = df.dropna(subset=["score"]).sort_values("score").head(20).copy()
    if worst.empty:
        return

    worst["label"] = worst["numero"].astype(str) + " - " + worst["metric_name"].astype(str)
    ax = worst.set_index("label")["score"].sort_values().plot(kind="barh", figsize=(11, 7), color="#e15759")
    plt.title("20 Piores Casos por Score")
    plt.xlabel("Score")
    plt.ylabel("Caso")
    plt.xlim(0, 1)
    save_current(output_dir, "06_metricas_piores_casos.png")


def plot_metric_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    pivot = df.pivot_table(index="numero", columns="metric_name", values="score", aggfunc="mean").sort_index()
    if pivot.empty:
        return

    plt.figure(figsize=(9, max(8, len(pivot) * 0.12)))
    plt.imshow(pivot.fillna(0), aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Heatmap Pergunta x Metrica")
    plt.xlabel("Metrica")
    plt.ylabel("Numero da pergunta")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=6)
    plt.colorbar(label="Score")
    save_current(output_dir, "07_metricas_heatmap_pergunta_metrica.png")


def plot_metric_tools_for_failures(df: pd.DataFrame, output_dir: Path) -> None:
    failures = df[(df["metric_name"].str.lower() == "tool correctness") & (~df["passed"])]
    if failures.empty:
        return

    missing: dict[str, int] = {}
    extra: dict[str, int] = {}
    for _, row in failures.iterrows():
        called = set(row["tools_called_list"])
        expected = set(row["expected_tools_list"])
        for tool in expected - called:
            missing[tool] = missing.get(tool, 0) + 1
        for tool in called - expected:
            extra[tool] = extra.get(tool, 0) + 1

    comparison = pd.DataFrame(
        {
            "faltou": pd.Series(missing),
            "sobrou": pd.Series(extra),
        }
    ).fillna(0).sort_values("faltou", ascending=True)

    if comparison.empty:
        return

    ax = comparison.plot(kind="barh", figsize=(12, max(6, len(comparison) * 0.35)), color=["#e15759", "#76b7b2"])
    plt.title("Falhas de Tool Correctness: Ferramentas Faltantes vs Extras")
    plt.xlabel("Quantidade de falhas")
    plt.ylabel("Ferramenta")
    save_current(output_dir, "08_metricas_tool_faltou_vs_sobrou.png")


def plot_metric_csv(df: pd.DataFrame, output_dir: Path) -> None:
    plot_metric_score_distribution(df, output_dir)
    plot_metric_average_scores(df, output_dir)
    plot_metric_pass_rate(df, output_dir)
    plot_metric_pass_fail(df, output_dir)
    plot_metric_score_by_route(df, output_dir)
    plot_worst_cases(df, output_dir)
    plot_metric_heatmap(df, output_dir)
    plot_metric_tools_for_failures(df, output_dir)


def extract_rag_source_count(value: object) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return 0

    source_count = parsed.get("sourceCount")
    if isinstance(source_count, int):
        return source_count

    sources = parsed.get("sources")
    return len(sources) if isinstance(sources, list) else 0


def explode_counts(values: Iterable[list[str]]) -> pd.Series:
    counter: dict[str, int] = {}
    for items in values:
        for item in items:
            counter[item] = counter.get(item, 0) + 1

    return pd.Series(counter).sort_values(ascending=False)


def save_current(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
    plt.close()


def annotate_bars(ax) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f" if any(bar.get_height() % 1 for bar in container) else "%.0f")


def plot_score_distribution(df: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(df["simple_tool_score"], bins=10, color="#3366cc", edgecolor="white")
    plt.axvline(threshold, color="#cc3333", linestyle="--", linewidth=2, label=f"threshold {threshold:.2f}")
    plt.title("Distribuicao do Simple Tool Score")
    plt.xlabel("Simple Tool Score")
    plt.ylabel("Quantidade de perguntas")
    plt.legend()
    save_current(output_dir, "01_distribuicao_simple_tool_score.png")


def plot_route_counts(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["route"].value_counts()
    ax = counts.plot(kind="bar", figsize=(9, 5), color="#4c78a8")
    plt.title("Quantidade de Perguntas por Rota")
    plt.xlabel("Rota")
    plt.ylabel("Quantidade")
    annotate_bars(ax)
    save_current(output_dir, "02_quantidade_por_rota.png")


def plot_score_by_route(df: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    scores = df.groupby("route")["simple_tool_score"].mean().sort_values(ascending=False)
    ax = scores.plot(kind="bar", figsize=(9, 5), color="#59a14f")
    plt.axhline(threshold, color="#cc3333", linestyle="--", linewidth=2, label=f"threshold {threshold:.2f}")
    plt.title("Score Medio por Rota")
    plt.xlabel("Rota")
    plt.ylabel("Simple Tool Score medio")
    plt.ylim(0, 1)
    plt.legend()
    annotate_bars(ax)
    save_current(output_dir, "03_score_medio_por_rota.png")


def plot_agents(df: pd.DataFrame, output_dir: Path) -> None:
    counts = explode_counts(df["agents_called_list"])
    if counts.empty:
        return

    ax = counts.plot(kind="bar", figsize=(10, 5), color="#f28e2b")
    plt.title("Agentes Mais Acionados")
    plt.xlabel("Agente")
    plt.ylabel("Quantidade de chamadas")
    annotate_bars(ax)
    save_current(output_dir, "04_agentes_mais_acionados.png")


def plot_called_tools(df: pd.DataFrame, output_dir: Path) -> None:
    counts = explode_counts(df["tools_called_list"]).head(20)
    if counts.empty:
        return

    ax = counts.sort_values().plot(kind="barh", figsize=(11, 7), color="#76b7b2")
    plt.title("Top Ferramentas Chamadas")
    plt.xlabel("Quantidade")
    plt.ylabel("Ferramenta")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    save_current(output_dir, "05_top_ferramentas_chamadas.png")


def plot_expected_vs_called_tools(df: pd.DataFrame, output_dir: Path) -> None:
    called = explode_counts(df["tools_called_list"])
    expected = explode_counts(df["expected_tools_list"])
    tools = sorted(set(called.index) | set(expected.index))
    if not tools:
        return

    comparison = pd.DataFrame(
        {
            "esperadas": [expected.get(tool, 0) for tool in tools],
            "chamadas": [called.get(tool, 0) for tool in tools],
        },
        index=tools,
    ).sort_values("esperadas", ascending=True)

    ax = comparison.plot(kind="barh", figsize=(12, max(6, len(comparison) * 0.35)), width=0.8)
    plt.title("Ferramentas Esperadas vs Chamadas")
    plt.xlabel("Quantidade de perguntas")
    plt.ylabel("Ferramenta")
    save_current(output_dir, "06_ferramentas_esperadas_vs_chamadas.png")


def tool_precision_recall(df: pd.DataFrame) -> pd.DataFrame:
    tools = sorted(
        set(tool for row in df["tools_called_list"] for tool in row)
        | set(tool for row in df["expected_tools_list"] for tool in row)
    )
    rows = []

    for tool in tools:
        tp = fp = fn = 0
        for _, result in df.iterrows():
            called = set(result["tools_called_list"])
            expected = set(result["expected_tools_list"])
            tp += int(tool in called and tool in expected)
            fp += int(tool in called and tool not in expected)
            fn += int(tool not in called and tool in expected)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        rows.append({"tool": tool, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn})

    return pd.DataFrame(rows).sort_values("recall")


def plot_tool_precision_recall(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = tool_precision_recall(df)
    if metrics.empty:
        return

    chart = metrics.set_index("tool")[["precision", "recall"]]
    ax = chart.plot(kind="barh", figsize=(12, max(6, len(chart) * 0.35)), width=0.8)
    plt.title("Precision e Recall por Ferramenta")
    plt.xlabel("Score")
    plt.ylabel("Ferramenta")
    plt.xlim(0, 1)
    save_current(output_dir, "07_precision_recall_por_ferramenta.png")


def plot_question_tool_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    tools = sorted(set(tool for row in df["tools_called_list"] for tool in row))
    if not tools:
        return

    matrix = []
    labels = []
    for _, result in df.iterrows():
        called = set(result["tools_called_list"])
        matrix.append([1 if tool in called else 0 for tool in tools])
        labels.append(str(result.get("numero", "")))

    fig_height = max(8, len(df) * 0.12)
    plt.figure(figsize=(12, fig_height))
    plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues")
    plt.title("Heatmap Pergunta x Ferramenta Chamada")
    plt.xlabel("Ferramenta")
    plt.ylabel("Numero da pergunta")
    plt.xticks(range(len(tools)), tools, rotation=75, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.colorbar(label="Chamada")
    save_current(output_dir, "08_heatmap_pergunta_ferramenta.png")


def plot_answer_length_vs_score(df: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    plt.figure(figsize=(9, 5))
    plt.scatter(df["answer_length"], df["simple_tool_score"], alpha=0.75, color="#8f63b8")
    plt.axhline(threshold, color="#cc3333", linestyle="--", linewidth=2, label=f"threshold {threshold:.2f}")
    plt.title("Tamanho da Resposta vs Simple Tool Score")
    plt.xlabel("Caracteres na resposta")
    plt.ylabel("Simple Tool Score")
    plt.ylim(0, 1)
    plt.legend()
    save_current(output_dir, "09_tamanho_resposta_vs_score.png")


def plot_rag_by_route(df: pd.DataFrame, output_dir: Path) -> None:
    rag = df.groupby("route")["rag_source_count"].mean().sort_values(ascending=False)
    ax = rag.plot(kind="bar", figsize=(9, 5), color="#edc948")
    plt.title("Media de Fontes RAG por Rota")
    plt.xlabel("Rota")
    plt.ylabel("Media de fontes RAG")
    annotate_bars(ax)
    save_current(output_dir, "10_fontes_rag_por_rota.png")


def print_summary(df: pd.DataFrame, csv_path: Path, output_dir: Path, threshold: float) -> None:
    if is_metric_csv(df):
        print(f"CSV de metricas analisado: {csv_path}")
        print(f"Linhas: {len(df)}")
        print("Score medio por metrica:")
        for metric_name, score in df.groupby("metric_name")["score"].mean().sort_values(ascending=False).items():
            pass_rate = df.loc[df["metric_name"] == metric_name, "passed"].mean() * 100
            print(f"  - {metric_name}: score={score:.4f} pass_rate={pass_rate:.2f}%")
        print(f"Graficos exportados em: {output_dir}")
        return

    pass_rate = (df["simple_tool_score"] >= threshold).mean() * 100
    print(f"CSV analisado: {csv_path}")
    print(f"Perguntas: {len(df)}")
    print(f"Simple Tool Score medio: {df['simple_tool_score'].mean():.4f}")
    print(f"Pass rate simples por threshold {threshold:.2f}: {pass_rate:.2f}%")
    print(f"Graficos exportados em: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera graficos PNG a partir do CSV exportado pela avaliacao DeepEval da API SMART."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Caminho do CSV. Se omitido, usa o CSV mais recente da pasta outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta onde os PNGs serao exportados.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold visual usado nos graficos de score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv or latest_csv()
    output_dir = args.out_dir
    df = load_results(csv_path)

    if is_metric_csv(df):
        plot_metric_csv(df, output_dir)
        print_summary(df, csv_path, output_dir, args.threshold)
        return

    plot_score_distribution(df, output_dir, args.threshold)
    plot_route_counts(df, output_dir)
    plot_score_by_route(df, output_dir, args.threshold)
    plot_agents(df, output_dir)
    plot_called_tools(df, output_dir)
    plot_expected_vs_called_tools(df, output_dir)
    plot_tool_precision_recall(df, output_dir)
    plot_question_tool_heatmap(df, output_dir)
    plot_answer_length_vs_score(df, output_dir, args.threshold)
    plot_rag_by_route(df, output_dir)
    print_summary(df, csv_path, output_dir, args.threshold)


if __name__ == "__main__":
    main()
