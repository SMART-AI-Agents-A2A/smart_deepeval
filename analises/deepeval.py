from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "analises" / "graficos"

METRIC_COLORS = [
    "#185FA5",
    "#3B6D11",
    "#854F0B",
    "#993556",
    "#533BA7",
    "#0F6E56",
    "#993C1D",
    "#27500A",
]

PASS_COLOR = "#59a14f"
FAIL_COLOR = "#e15759"
NEUTRAL_COLOR = "#4c78a8"
THRESHOLD_COLOR = "#cc3333"


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
        df["route"] = df.get("route", pd.Series(dtype=str)).fillna("").replace("", "sem_rota")
        df["answer_length"] = df.get("actual_output", pd.Series(dtype=str)).fillna("").astype(str).str.len()
        df["tools_called_list"] = df.get("tools_called", pd.Series(dtype=str)).apply(split_pipe)
        df["expected_tools_list"] = df.get("expected_tools", pd.Series(dtype=str)).apply(split_pipe)
        df["agents_called_list"] = df.get("agents_called", pd.Series(dtype=str)).apply(split_pipe)
        df["tools_correct_list"] = df.get("tools_correct", pd.Series(dtype=str)).apply(split_pipe)
        df["tools_missing_list"] = df.get("tools_missing", pd.Series(dtype=str)).apply(split_pipe)
        df["tools_extra_list"] = df.get("tools_extra", pd.Series(dtype=str)).apply(split_pipe)
        return df

    df["route"] = df.get("route", pd.Series(dtype=str)).fillna("").replace("", "sem_rota")
    df["answer_length"] = df.get("actual_output", pd.Series(dtype=str)).fillna("").astype(str).str.len()
    df["tools_called_list"] = df.get("tools_called", pd.Series(dtype=str)).apply(split_pipe)
    df["expected_tools_list"] = df.get("expected_tools", pd.Series(dtype=str)).apply(split_pipe)
    df["agents_called_list"] = df.get("agents_called", pd.Series(dtype=str)).apply(split_pipe)
    df["tools_correct_list"] = df.get("tools_correct", pd.Series(dtype=str)).apply(split_pipe)
    df["tools_missing_list"] = df.get("tools_missing", pd.Series(dtype=str)).apply(split_pipe)
    df["tools_extra_list"] = df.get("tools_extra", pd.Series(dtype=str)).apply(split_pipe)
    df["rag_source_count"] = df.get("rag_json", pd.Series(dtype=str)).apply(extract_rag_source_count)
    return df


def load_multi(csv_paths: list[Path], labels: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path, label in zip(csv_paths, labels):
        df = load_results(path)
        df["modelo"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


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


def annotate_bars(ax) -> None:
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f" if any(bar.get_height() % 1 for bar in container) else "%.0f",
            padding=2,
        )


def fig_width_for_models(n_models: int, base: float = 9.0, per_model: float = 1.5) -> float:
    return max(base, base + (n_models - 1) * per_model)


class FigureSaver:
    def __init__(self, output_dir: Path, pdf_path: Path):
        self.output_dir = output_dir
        self.pdf_path = pdf_path
        self._pdf: pdf_backend.PdfPages | None = None

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pdf = pdf_backend.PdfPages(self.pdf_path)
        return self

    def __exit__(self, *_):
        if self._pdf:
            self._pdf.close()

    def save(self, filename: str) -> None:
        plt.tight_layout()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output_dir / filename, dpi=160, bbox_inches="tight")
        if self._pdf:
            self._pdf.savefig(bbox_inches="tight")
        plt.close()


def plot_metric_score_distribution(df: pd.DataFrame, saver: FigureSaver, threshold: float) -> None:
    metrics = sorted(df["metric_name"].dropna().unique())
    if not metrics:
        return
    plt.figure(figsize=(10, 5))
    for metric_name in metrics:
        values = df.loc[df["metric_name"] == metric_name, "score"].dropna()
        if values.empty:
            continue
        plt.hist(values, bins=10, alpha=0.55, label=metric_name)
    plt.axvline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=2, label=f"threshold {threshold:.2f}")
    plt.title("Distribuicao de Scores por Metrica")
    plt.xlabel("Score")
    plt.ylabel("Quantidade de casos")
    plt.legend()
    saver.save("01_metricas_distribuicao_scores.png")


def plot_metric_average_scores(df: pd.DataFrame, saver: FigureSaver, threshold: float) -> None:
    scores = df.groupby("metric_name")["score"].mean().sort_values(ascending=False)
    ax = scores.plot(kind="bar", figsize=(9, 5), color=NEUTRAL_COLOR)
    plt.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=2, label=f"threshold {threshold:.2f}")
    plt.title("Score Medio por Metrica")
    plt.xlabel("Metrica")
    plt.ylabel("Score medio")
    plt.ylim(0, 1)
    plt.legend()
    annotate_bars(ax)
    saver.save("02_metricas_score_medio.png")


def plot_metric_pass_rate(df: pd.DataFrame, saver: FigureSaver) -> None:
    pass_rate = (df.groupby("metric_name")["passed"].mean() * 100).sort_values(ascending=False)
    ax = pass_rate.plot(kind="bar", figsize=(9, 5), color=PASS_COLOR)
    plt.title("Pass Rate por Metrica (%)")
    plt.xlabel("Metrica")
    plt.ylabel("Pass rate (%)")
    plt.ylim(0, 100)
    annotate_bars(ax)
    saver.save("03_metricas_pass_rate.png")


def plot_metric_pass_fail(df: pd.DataFrame, saver: FigureSaver) -> None:
    pivot = (
        df.assign(status=df["passed"].map({True: "passou", False: "falhou"}))
        .pivot_table(index="metric_name", columns="status", values="question", aggfunc="count", fill_value=0)
    )
    for col in ["falhou", "passou"]:
        if col not in pivot.columns:
            pivot[col] = 0
    ax = pivot[["passou", "falhou"]].plot(kind="bar", stacked=True, figsize=(9, 5), color=[PASS_COLOR, FAIL_COLOR])
    plt.title("Passou vs Falhou por Metrica")
    plt.xlabel("Metrica")
    plt.ylabel("Quantidade de casos")
    annotate_bars(ax)
    saver.save("04_metricas_passou_vs_falhou.png")


def plot_metric_score_by_route(df: pd.DataFrame, saver: FigureSaver, threshold: float) -> None:
    pivot = df.pivot_table(index="route", columns="metric_name", values="score", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    plt.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=2, label=f"threshold {threshold:.2f}")
    plt.title("Score Medio por Rota e Metrica")
    plt.xlabel("Rota")
    plt.ylabel("Score medio")
    plt.ylim(0, 1)
    plt.legend()
    saver.save("05_metricas_score_por_rota.png")


def plot_worst_cases(df: pd.DataFrame, saver: FigureSaver) -> None:
    worst = df.dropna(subset=["score"]).sort_values("score").head(20).copy()
    if worst.empty:
        return
    worst["label"] = worst["numero"].astype(str) + " - " + worst["metric_name"].astype(str)
    worst.set_index("label")["score"].sort_values().plot(kind="barh", figsize=(11, 7), color=FAIL_COLOR)
    plt.title("20 Piores Casos por Score")
    plt.xlabel("Score")
    plt.ylabel("Caso")
    plt.xlim(0, 1)
    saver.save("06_metricas_piores_casos.png")


def plot_metric_heatmap(df: pd.DataFrame, saver: FigureSaver) -> None:
    pivot = df.pivot_table(index="numero", columns="metric_name", values="score", aggfunc="mean").sort_index()
    if pivot.empty:
        return
    plt.figure(figsize=(9, max(8, len(pivot) * 0.35)))
    plt.imshow(pivot.fillna(0), aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Heatmap Pergunta x Metrica")
    plt.xlabel("Metrica")
    plt.ylabel("Numero da pergunta")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    plt.colorbar(label="Score")
    saver.save("07_metricas_heatmap_pergunta_metrica.png")


def plot_metric_tools_breakdown(df: pd.DataFrame, saver: FigureSaver) -> None:
    tc = df[df["metric_name"].str.lower().str.contains("tool correctness", na=False)]
    if tc.empty:
        return

    missing: dict[str, int] = {}
    extra: dict[str, int] = {}
    for _, row in tc.iterrows():
        for tool in row["tools_missing_list"]:
            missing[tool] = missing.get(tool, 0) + 1
        for tool in row["tools_extra_list"]:
            extra[tool] = extra.get(tool, 0) + 1

    comparison = pd.DataFrame({"faltou": pd.Series(missing), "sobrou": pd.Series(extra)}).fillna(0).sort_values("faltou", ascending=True)
    if comparison.empty:
        return

    comparison.plot(kind="barh", figsize=(12, max(6, len(comparison) * 0.4)), color=["#e15759", "#76b7b2"])
    plt.title("Tool Correctness: Ferramentas Faltantes vs Extras")
    plt.xlabel("Quantidade de ocorrencias")
    plt.ylabel("Ferramenta")
    saver.save("08_metricas_tool_faltou_vs_sobrou.png")


def plot_multi_score_comparison(df: pd.DataFrame, saver: FigureSaver, threshold: float) -> None:
    models = df["modelo"].unique()
    n = len(models)
    w = fig_width_for_models(n)
    pivot = df.groupby(["modelo", "metric_name"])["score"].mean().unstack("metric_name")

    ax = pivot.plot(kind="bar", figsize=(w, 5), color=METRIC_COLORS[:len(pivot.columns)], width=0.7)
    plt.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.5, label=f"threshold {threshold:.2f}")
    plt.title("Score Medio por Modelo e Metrica")
    plt.xlabel("Modelo")
    plt.ylabel("Score medio")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    annotate_bars(ax)
    saver.save("09_multi_score_por_modelo.png")


def plot_multi_pass_rate(df: pd.DataFrame, saver: FigureSaver) -> None:
    models = df["modelo"].unique()
    n = len(models)
    w = fig_width_for_models(n)
    pivot = (df.groupby(["modelo", "metric_name"])["passed"].mean() * 100).unstack("metric_name")

    ax = pivot.plot(kind="bar", figsize=(w, 5), color=METRIC_COLORS[:len(pivot.columns)], width=0.7)
    plt.title("Pass Rate por Modelo e Metrica (%)")
    plt.xlabel("Modelo")
    plt.ylabel("Pass rate (%)")
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    annotate_bars(ax)
    saver.save("10_multi_pass_rate_por_modelo.png")


def plot_multi_overall_score(df: pd.DataFrame, saver: FigureSaver, threshold: float) -> None:
    models = df["modelo"].unique()
    n = len(models)
    w = fig_width_for_models(n)
    overall = df.groupby("modelo")["score"].mean().sort_values(ascending=False)

    colors = [PASS_COLOR if v >= threshold else FAIL_COLOR for v in overall]
    ax = overall.plot(kind="bar", figsize=(w, 5), color=colors, width=0.6)
    plt.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.5, label=f"threshold {threshold:.2f}")
    plt.title("Score Geral por Modelo (media de todas as metricas)")
    plt.xlabel("Modelo")
    plt.ylabel("Score medio")
    plt.ylim(0, 1)
    plt.legend()
    plt.xticks(rotation=25, ha="right")
    annotate_bars(ax)
    saver.save("11_multi_score_geral.png")


def plot_multi_score_distribution(df: pd.DataFrame, saver: FigureSaver, threshold: float) -> None:
    metrics = sorted(df["metric_name"].dropna().unique())
    n_metrics = len(metrics)
    if not n_metrics:
        return

    models = sorted(df["modelo"].dropna().unique())
    n_models = len(models)
    w = fig_width_for_models(n_models, base=12)
    fig, axes = plt.subplots(1, n_metrics, figsize=(w, 4), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for idx, model in enumerate(models):
            values = df.loc[(df["metric_name"] == metric) & (df["modelo"] == model), "score"].dropna()
            if values.empty:
                continue
            ax.hist(values, bins=8, alpha=0.55, label=model, color=METRIC_COLORS[idx % len(METRIC_COLORS)])
        ax.axvline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2)
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Casos", fontsize=8)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=min(n_models, 4), fontsize=8, bbox_to_anchor=(0.5, -0.08))
    plt.suptitle("Distribuicao de Scores por Metrica e Modelo", y=1.02)
    saver.save("12_multi_distribuicao_scores.png")


def plot_multi_tool_breakdown(df: pd.DataFrame, saver: FigureSaver) -> None:
    tc = df[df["metric_name"].str.lower().str.contains("tool correctness", na=False)]
    if tc.empty:
        return

    models = sorted(tc["modelo"].unique())
    n = len(models)
    w = fig_width_for_models(n, base=12)
    fig, axes = plt.subplots(1, n, figsize=(w, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = tc[tc["modelo"] == model]
        missing: dict[str, int] = {}
        extra: dict[str, int] = {}
        for _, row in sub.iterrows():
            for tool in row["tools_missing_list"]:
                missing[tool] = missing.get(tool, 0) + 1
            for tool in row["tools_extra_list"]:
                extra[tool] = extra.get(tool, 0) + 1
        comparison = pd.DataFrame({"faltou": pd.Series(missing), "sobrou": pd.Series(extra)}).fillna(0).sort_values("faltou", ascending=True)
        if comparison.empty:
            ax.set_title(model, fontsize=9)
            continue
        comparison.plot(kind="barh", ax=ax, color=["#e15759", "#76b7b2"], width=0.7, legend=(model == models[0]))
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("Ocorrencias", fontsize=8)

    plt.suptitle("Ferramentas Faltantes vs Extras por Modelo", y=1.02)
    saver.save("13_multi_tool_breakdown.png")


def plot_metric_csv(df: pd.DataFrame, saver: FigureSaver, threshold: float) -> None:
    plot_metric_score_distribution(df, saver, threshold)
    plot_metric_average_scores(df, saver, threshold)
    plot_metric_pass_rate(df, saver)
    plot_metric_pass_fail(df, saver)
    plot_metric_score_by_route(df, saver, threshold)
    plot_worst_cases(df, saver)
    plot_metric_heatmap(df, saver)
    plot_metric_tools_breakdown(df, saver)


def plot_collected_csv(df: pd.DataFrame, saver: FigureSaver, threshold: float) -> None:
    plot_route_counts(df, saver)
    plot_agents(df, saver)
    plot_called_tools(df, saver)
    plot_expected_vs_called_tools(df, saver)
    plot_tool_precision_recall_chart(df, saver)
    plot_question_tool_heatmap(df, saver)
    plot_rag_by_route(df, saver)


def plot_route_counts(df: pd.DataFrame, saver: FigureSaver) -> None:
    counts = df["route"].value_counts()
    ax = counts.plot(kind="bar", figsize=(9, 5), color=NEUTRAL_COLOR)
    plt.title("Quantidade de Perguntas por Rota")
    plt.xlabel("Rota")
    plt.ylabel("Quantidade")
    annotate_bars(ax)
    saver.save("02_quantidade_por_rota.png")


def plot_agents(df: pd.DataFrame, saver: FigureSaver) -> None:
    counts = explode_counts(df["agents_called_list"])
    if counts.empty:
        return
    ax = counts.plot(kind="bar", figsize=(10, 5), color="#f28e2b")
    plt.title("Agentes Mais Acionados")
    plt.xlabel("Agente")
    plt.ylabel("Quantidade de chamadas")
    annotate_bars(ax)
    saver.save("04_agentes_mais_acionados.png")


def plot_called_tools(df: pd.DataFrame, saver: FigureSaver) -> None:
    counts = explode_counts(df["tools_called_list"]).head(20)
    if counts.empty:
        return
    ax = counts.sort_values().plot(kind="barh", figsize=(11, 7), color="#76b7b2")
    plt.title("Top Ferramentas Chamadas")
    plt.xlabel("Quantidade")
    plt.ylabel("Ferramenta")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    saver.save("05_top_ferramentas_chamadas.png")


def plot_expected_vs_called_tools(df: pd.DataFrame, saver: FigureSaver) -> None:
    called = explode_counts(df["tools_called_list"])
    expected = explode_counts(df["expected_tools_list"])
    tools = sorted(set(called.index) | set(expected.index))
    if not tools:
        return
    comparison = pd.DataFrame(
        {"esperadas": [expected.get(t, 0) for t in tools], "chamadas": [called.get(t, 0) for t in tools]},
        index=tools,
    ).sort_values("esperadas", ascending=True)
    comparison.plot(kind="barh", figsize=(12, max(6, len(comparison) * 0.4)), width=0.8)
    plt.title("Ferramentas Esperadas vs Chamadas")
    plt.xlabel("Quantidade de perguntas")
    plt.ylabel("Ferramenta")
    saver.save("06_ferramentas_esperadas_vs_chamadas.png")


def tool_precision_recall(df: pd.DataFrame) -> pd.DataFrame:
    tools = sorted(
        {t for row in df["tools_called_list"] for t in row}
        | {t for row in df["expected_tools_list"] for t in row}
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


def plot_tool_precision_recall_chart(df: pd.DataFrame, saver: FigureSaver) -> None:
    metrics = tool_precision_recall(df)
    if metrics.empty:
        return
    chart = metrics.set_index("tool")[["precision", "recall"]]
    chart.plot(kind="barh", figsize=(12, max(6, len(chart) * 0.4)), width=0.8)
    plt.title("Precision e Recall por Ferramenta")
    plt.xlabel("Score")
    plt.ylabel("Ferramenta")
    plt.xlim(0, 1)
    saver.save("07_precision_recall_por_ferramenta.png")


def plot_question_tool_heatmap(df: pd.DataFrame, saver: FigureSaver) -> None:
    tools = sorted({t for row in df["tools_called_list"] for t in row})
    if not tools:
        return
    matrix = []
    labels = []
    for _, result in df.iterrows():
        called = set(result["tools_called_list"])
        matrix.append([1 if t in called else 0 for t in tools])
        labels.append(str(result.get("numero", "")))
    plt.figure(figsize=(12, max(8, len(df) * 0.3)))
    plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues")
    plt.title("Heatmap Pergunta x Ferramenta Chamada")
    plt.xlabel("Ferramenta")
    plt.ylabel("Numero da pergunta")
    plt.xticks(range(len(tools)), tools, rotation=75, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.colorbar(label="Chamada")
    saver.save("08_heatmap_pergunta_ferramenta.png")


def plot_rag_by_route(df: pd.DataFrame, saver: FigureSaver) -> None:
    if "rag_source_count" not in df.columns:
        return
    rag = df.groupby("route")["rag_source_count"].mean().sort_values(ascending=False)
    ax = rag.plot(kind="bar", figsize=(9, 5), color="#edc948")
    plt.title("Media de Fontes RAG por Rota")
    plt.xlabel("Rota")
    plt.ylabel("Media de fontes RAG")
    annotate_bars(ax)
    saver.save("10_fontes_rag_por_rota.png")


def print_summary(df: pd.DataFrame, csv_path: Path | None, output_dir: Path, threshold: float) -> None:
    if csv_path:
        print(f"CSV analisado: {csv_path}")
    print(f"Linhas: {len(df)}")
    if is_metric_csv(df):
        print("Score medio por metrica:")
        for metric_name, score in df.groupby("metric_name")["score"].mean().sort_values(ascending=False).items():
            pass_rate = df.loc[df["metric_name"] == metric_name, "passed"].mean() * 100
            print(f"  {metric_name}: score={score:.4f}  pass={pass_rate:.1f}%")
    print(f"Graficos exportados em: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera graficos PNG + PDF a partir dos CSVs exportados pela avaliacao DeepEval da API SMART. "
            "Aceita um ou varios CSVs (modo multi-modelo)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=None,
        metavar="CSV",
        help=(
            "Um ou mais caminhos de CSV. "
            "Ex.: --csv outputs/gpt.csv outputs/deepseek.csv. "
            "Se omitido, usa o CSV mais recente da pasta outputs."
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Rotulos para cada CSV em --csv. "
            "Ex.: --labels GPT-5.4-mini DeepSeek-V4. "
            "Se omitido, usa o nome do arquivo sem extensao."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta onde os PNGs e o PDF serao exportados.",
    )
    parser.add_argument(
        "--pdf-name",
        type=str,
        default="relatorio_metricas.pdf",
        help="Nome do arquivo PDF gerado (dentro de --out-dir).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold visual usado nos graficos de score (padrao: 0.7).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.out_dir
    pdf_path = output_dir / args.pdf_name

    if args.csv:
        csv_paths: list[Path] = args.csv
        if args.labels:
            if len(args.labels) != len(csv_paths):
                raise ValueError(
                    f"--labels tem {len(args.labels)} valores mas --csv tem {len(csv_paths)}. "
                    "Devem ter o mesmo numero."
                )
            labels = args.labels
        else:
            labels = [p.stem for p in csv_paths]
    else:
        csv_paths = [latest_csv()]
        labels = [csv_paths[0].stem]

    multi_mode = len(csv_paths) > 1

    with FigureSaver(output_dir, pdf_path) as saver:
        if multi_mode:
            df_all = load_multi(csv_paths, labels)
            first_df = load_results(csv_paths[0])
            if is_metric_csv(first_df):
                for path, label in zip(csv_paths, labels):
                    df_single = load_results(path)
                    plot_metric_csv(df_single, saver, args.threshold)
                plot_multi_score_comparison(df_all, saver, args.threshold)
                plot_multi_pass_rate(df_all, saver)
                plot_multi_overall_score(df_all, saver, args.threshold)
                plot_multi_score_distribution(df_all, saver, args.threshold)
                plot_multi_tool_breakdown(df_all, saver)
            else:
                for path, label in zip(csv_paths, labels):
                    df_single = load_results(path)
                    plot_collected_csv(df_single, saver, args.threshold)
                plot_multi_score_comparison(df_all, saver, args.threshold)
            print_summary(df_all, None, output_dir, args.threshold)
        else:
            df = load_results(csv_paths[0])
            if is_metric_csv(df):
                plot_metric_csv(df, saver, args.threshold)
            else:
                plot_collected_csv(df, saver, args.threshold)
            print_summary(df, csv_paths[0], output_dir, args.threshold)

    print(f"PDF consolidado: {pdf_path}")


if __name__ == "__main__":
    main()