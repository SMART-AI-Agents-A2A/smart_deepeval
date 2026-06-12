from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "analises" / "graficos"

PALETTE = ["#185FA5", "#3B6D11", "#854F0B", "#993556", "#533BA7", "#0F6E56"]
THRESHOLD_COLOR = "#CC3333"
GRID_COLOR = "#E8E8E8"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 130,
    }
)


TEXT = {
    "pt": {
        "threshold": "limite",
        "avg_score": "Score medio",
        "avg_score_tool": "Score medio (Tool Correctness)",
        "avg_by_metric_model": "Score medio por metrica - {model}",
        "avg_by_metric_all": "Score medio por metrica e modelo - agregado",
        "score_by_metric_model": "Score por metrica - {model}",
        "extras_title": "Score medio vs ferramentas a mais",
        "missing_title": "Score medio vs ferramentas faltando",
        "extras_all_title": "Score medio vs ferramentas a mais - todos os modelos",
        "missing_all_title": "Score medio vs ferramentas faltando - todos os modelos",
        "extras_x": "Numero de ferramentas extras chamadas",
        "missing_x": "Numero de ferramentas esperadas nao chamadas",
        "tool_impact": "Tool Correctness - impacto de ferramentas extras e faltando | {model}",
        "bubble_note": "Tamanho do ponto proporcional ao numero de casos no grupo.",
        "tools_simple_model": "Resumo de uso de ferramentas - {model}",
        "tools_simple_all": "Resumo de uso de ferramentas por modelo",
        "tools_y": "Numero de ferramentas",
        "expected_called": "Esperadas chamadas",
        "expected_missing": "Esperadas faltando",
        "extra_called": "Extras chamadas",
        "tools_question_model": "Uso de ferramentas por questao - {model}",
        "question_score": "Score por questao - {model}",
        "question_score_all": "Score por questao e modelo - agregado",
        "question_x": "Questao",
        "pass_fail_model": "Passed vs failed por metrica - {model}",
        "pass_fail_all": "Passed vs failed por metrica e modelo - agregado",
        "pass_fail_y": "Numero de casos",
        "pass_fail_question_model": "Passed vs failed por questao - {model}",
        "pass_fail_question_all": "Passed vs failed por questao e modelo - agregado",
        "passed": "Passed",
        "failed": "Failed",
        "pass_rate": "Pass rate",
        "summary_title": "RESUMO DA AVALIACAO - API SMART com DeepEval",
        "metrics_meaning": "O QUE CADA METRICA SIGNIFICA",
        "results_by_model": "RESULTADOS POR MODELO",
        "method_note": "NOTA METODOLOGICA",
    },
    "en": {
        "threshold": "threshold",
        "avg_score": "Average score",
        "avg_score_tool": "Average score (Tool Correctness)",
        "avg_by_metric_model": "Average score by metric - {model}",
        "avg_by_metric_all": "Average score by metric and model - aggregate",
        "score_by_metric_model": "Score by metric - {model}",
        "extras_title": "Average score vs extra tools",
        "missing_title": "Average score vs missing tools",
        "extras_all_title": "Average score vs extra tools - all models",
        "missing_all_title": "Average score vs missing tools - all models",
        "extras_x": "Number of extra tools called",
        "missing_x": "Number of expected tools not called",
        "tool_impact": "Tool Correctness - impact of extra and missing tools | {model}",
        "bubble_note": "Point size is proportional to the number of cases in the group.",
        "tools_simple_model": "Tool usage summary - {model}",
        "tools_simple_all": "Tool usage summary by model",
        "tools_y": "Number of tools",
        "expected_called": "Expected called",
        "expected_missing": "Expected missing",
        "extra_called": "Extra called",
        "tools_question_model": "Tool usage by question - {model}",
        "question_score": "Score by question - {model}",
        "question_score_all": "Score by question and model - aggregate",
        "question_x": "Question",
        "pass_fail_model": "Passed vs failed by metric - {model}",
        "pass_fail_all": "Passed vs failed by metric and model - aggregate",
        "pass_fail_y": "Number of cases",
        "pass_fail_question_model": "Passed vs failed by question - {model}",
        "pass_fail_question_all": "Passed vs failed by question and model - aggregate",
        "passed": "Passed",
        "failed": "Failed",
        "pass_rate": "Pass rate",
        "summary_title": "EVALUATION SUMMARY - SMART API with DeepEval",
        "metrics_meaning": "WHAT EACH METRIC MEANS",
        "results_by_model": "RESULTS BY MODEL",
        "method_note": "METHODOLOGICAL NOTE",
    },
}


def split_pipe(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [x.strip() for x in value.split("|") if x.strip()]


def safe_name(value: str) -> str:
    keep = []
    for char in value.strip():
        keep.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(keep).strip("_") or "model"


def get_score_column(df: pd.DataFrame) -> str:
    for candidate in ("score", "simple_tool_score", "selection_score"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "CSV sem coluna de score. Esperado: score, simple_tool_score ou selection_score."
    )


def get_question_column(df: pd.DataFrame) -> str | None:
    for candidate in ("numero", "question", "question_id", "case_id", "index"):
        if candidate in df.columns:
            return candidate
    return None


def question_order(df: pd.DataFrame, question_col: str) -> list[object]:
    return list(pd.unique(df[question_col].dropna()))


def question_axis_labels(order: list[object]) -> list[str]:
    return [str(idx) for idx in range(1, len(order) + 1)]


def write_question_mapping(df: pd.DataFrame, output_dir: Path) -> None:
    question_col = get_question_column(df)
    if question_col is None:
        return

    order = question_order(df, question_col)
    mapping = pd.DataFrame(
        {
            "questao_no_grafico": range(1, len(order) + 1),
            "questao_original_csv": order,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_dir / "mapeamento_questoes.csv", index=False, encoding="utf-8-sig")
    print("  mapeamento_questoes.csv")


def load_csv(path: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    score_col = get_score_column(df)
    df["score"] = pd.to_numeric(df[score_col], errors="coerce")

    if "passed" in df.columns:
        df["passed"] = df["passed"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["passed"] = df["score"] >= threshold

    if "metric_name" not in df.columns:
        df["metric_name"] = "Score"

    for column in ("tools_missing", "tools_extra", "tools_called", "expected_tools"):
        if column not in df.columns:
            df[column] = ""

    df["tools_missing_list"] = df["tools_missing"].apply(split_pipe)
    df["tools_extra_list"] = df["tools_extra"].apply(split_pipe)
    df["tools_called_list"] = df["tools_called"].apply(split_pipe)
    df["expected_tools_list"] = df["expected_tools"].apply(split_pipe)
    return df


def load_all(csv_paths: list[Path], labels: list[str], threshold: float) -> pd.DataFrame:
    frames = []
    for path, label in zip(csv_paths, labels):
        df = load_csv(path, threshold)
        df["modelo"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def resolve_output_dir(base_dir: Path, overwrite: bool, run_name: str | None) -> Path:
    if overwrite:
        return base_dir

    stem = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / stem
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{stem}_{suffix}"
        suffix += 1
    return candidate


class Saver:
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
        plt.savefig(self.output_dir / filename, dpi=160, bbox_inches="tight")
        if self._pdf:
            self._pdf.savefig(bbox_inches="tight")
        plt.close()
        print(f"  {filename}")


def _label(language: str, key: str, **kwargs: object) -> str:
    return TEXT[language][key].format(**kwargs)


def _annotate_bars(ax, fmt: str = "{:.2f}", color: str | None = None) -> None:
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if np.isnan(h) or h == 0:
                continue
            c = color or bar.get_facecolor()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
                color=c,
            )


def barras_por_modelo(
    df: pd.DataFrame, saver: Saver, threshold: float, modelo: str, idx: int, language: str
) -> None:
    sub = df[df["modelo"] == modelo]
    metrics = sorted(sub["metric_name"].dropna().unique())
    means = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
    color = PALETTE[idx % len(PALETTE)]

    _, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(metrics, means, color=color, width=0.5, zorder=3)
    ax.axhline(
        threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.4,
        label=f"{_label(language, 'threshold')} {threshold:.1f}",
        zorder=4,
    )
    _annotate_bars(ax, color=color)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(_label(language, "avg_score"))
    ax.set_title(_label(language, "avg_by_metric_model", model=modelo))
    ax.legend(fontsize=9, framealpha=0)
    saver.save(f"barras_{safe_name(modelo)}.png")


def barras_agregado(df: pd.DataFrame, saver: Saver, threshold: float, language: str) -> None:
    models = list(df["modelo"].unique())
    metrics = sorted(df["metric_name"].dropna().unique())
    width = 0.7 / max(len(models), 1)
    x = np.arange(len(metrics))

    _, ax = plt.subplots(figsize=(max(8, 3 * len(metrics) + len(models)), 5))
    for i, model in enumerate(models):
        color = PALETTE[i % len(PALETTE)]
        sub = df[df["modelo"] == model]
        means = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width=width * 0.92, color=color, label=model, zorder=3)
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.015,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                )

    ax.axhline(
        threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.4,
        label=f"{_label(language, 'threshold')} {threshold:.1f}",
        zorder=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(_label(language, "avg_score"))
    ax.set_title(_label(language, "avg_by_metric_all"))
    ax.legend(fontsize=9, framealpha=0)
    saver.save("barras_agregado.png")


def slope_por_modelo(
    df: pd.DataFrame, saver: Saver, threshold: float, modelo: str, idx: int, language: str
) -> None:
    sub = df[df["modelo"] == modelo]
    metrics = sorted(sub["metric_name"].dropna().unique())
    scores = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
    color = PALETTE[idx % len(PALETTE)]
    x = list(range(len(metrics)))

    _, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        x,
        scores,
        color=color,
        linewidth=2.2,
        zorder=3,
        marker="o",
        markersize=13,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=2,
    )
    for xi, score in zip(x, scores):
        ax.text(xi, score + 0.03, f"{score:.2f}", ha="center", va="bottom", fontsize=10, color=color)

    ax.axhline(
        threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.3,
        zorder=2,
        label=f"{_label(language, 'threshold')} {threshold:.1f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylabel(_label(language, "avg_score"))
    ax.set_title(_label(language, "score_by_metric_model", model=modelo))
    ax.legend(fontsize=9, framealpha=0)
    saver.save(f"slope_{safe_name(modelo)}.png")


def slope_agregado(df: pd.DataFrame, saver: Saver, threshold: float, language: str) -> None:
    models = list(df["modelo"].unique())
    metrics = sorted(df["metric_name"].dropna().unique())
    x = list(range(len(metrics)))
    if not x:
        return
    last_x = x[-1]

    _, ax = plt.subplots(figsize=(9, 5))
    for idx, model in enumerate(models):
        color = PALETTE[idx % len(PALETTE)]
        sub = df[df["modelo"] == model]
        scores = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
        ax.plot(
            x,
            scores,
            color=color,
            linewidth=2.2,
            zorder=3,
            marker="o",
            markersize=13,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=2,
            label=model,
        )
        for xi, score in zip(x, scores):
            if not np.isnan(score):
                ax.text(
                    xi,
                    score + 0.05,
                    f"{score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                )

    ax.axhline(
        threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.3,
        zorder=2,
        label=f"{_label(language, 'threshold')} {threshold:.1f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.5, last_x + 0.5)
    ax.set_ylabel(_label(language, "avg_score"))
    ax.set_title(_label(language, "avg_by_metric_all"))
    ax.legend(fontsize=9, framealpha=0, loc="upper left")
    saver.save("slope_agregado.png")


def _tool_avg_by_count(tc: pd.DataFrame, col: str) -> dict[int, tuple[float, int]]:
    from collections import defaultdict

    groups: dict[int, list[float]] = defaultdict(list)
    for _, row in tc.iterrows():
        groups[int(row[col])].append(float(row["score"]))
    return {k: (sum(v) / len(v), len(v)) for k, v in groups.items()}


def dispersao_por_modelo(
    df: pd.DataFrame, saver: Saver, threshold: float, modelo: str, idx: int, language: str
) -> None:
    tc = df[
        (df["modelo"] == modelo)
        & df["metric_name"].str.lower().str.contains("tool correctness", na=False)
    ].copy()
    if tc.empty:
        return

    tc["extras"] = tc["tools_extra_list"].apply(len)
    tc["missing"] = tc["tools_missing_list"].apply(len)
    color = PALETTE[idx % len(PALETTE)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title_key, x_key in zip(
        axes,
        ["extras", "missing"],
        ["extras_title", "missing_title"],
        ["extras_x", "missing_x"],
    ):
        stats = _tool_avg_by_count(tc, col)
        xs = sorted(stats.keys())
        ys = [stats[x][0] for x in xs]
        ns = [stats[x][1] for x in xs]
        sizes = [max(60, n * 40) for n in ns]

        ax.plot(xs, ys, color=color, linewidth=1.8, zorder=2, alpha=0.6)
        ax.scatter(xs, ys, s=sizes, color=color, zorder=3, edgecolors="white", linewidths=1.5, alpha=0.9)

        for x_val, y_val, n_val in zip(xs, ys, ns):
            ax.text(x_val, y_val + 0.04, f"{y_val:.2f}\n(n={n_val})", ha="center", va="bottom", fontsize=7.5, color=color)

        ax.axhline(
            threshold,
            color=THRESHOLD_COLOR,
            linestyle="--",
            linewidth=1.2,
            zorder=1,
            label=f"{_label(language, 'threshold')} {threshold:.1f}",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=9)
        ax.set_xlabel(_label(language, x_key), fontsize=9, labelpad=8)
        ax.set_ylabel(_label(language, "avg_score_tool"), fontsize=9)
        ax.set_title(_label(language, title_key), fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, framealpha=0)

    fig.suptitle(_label(language, "tool_impact", model=modelo), fontsize=11)
    saver.save(f"dispersao_{safe_name(modelo)}.png")


def dispersao_agregado(df: pd.DataFrame, saver: Saver, threshold: float, language: str) -> None:
    tc = df[df["metric_name"].str.lower().str.contains("tool correctness", na=False)].copy()
    if tc.empty:
        return

    tc["extras"] = tc["tools_extra_list"].apply(len)
    tc["missing"] = tc["tools_missing_list"].apply(len)
    models = list(tc["modelo"].unique())

    for col, title_key, x_key, filename in [
        ("extras", "extras_all_title", "extras_x", "dispersao_extras_agregado.png"),
        ("missing", "missing_all_title", "missing_x", "dispersao_missing_agregado.png"),
    ]:
        all_xs: set[int] = set()
        for model in models:
            sub = tc[tc["modelo"] == model]
            all_xs.update(int(v) for v in sub[col].unique())
        xs_sorted = sorted(all_xs)

        _, ax = plt.subplots(figsize=(max(9, len(xs_sorted) * 1.2 + 2), 5))
        for idx, model in enumerate(models):
            color = PALETTE[idx % len(PALETTE)]
            sub = tc[tc["modelo"] == model]
            stats = _tool_avg_by_count(sub, col)
            xs = sorted(stats.keys())
            ys = [stats[x][0] for x in xs]
            ns = [stats[x][1] for x in xs]
            sizes = [max(55, n * 38) for n in ns]

            ax.plot(xs, ys, color=color, linewidth=1.8, zorder=2, alpha=0.6, label=model)
            ax.scatter(xs, ys, s=sizes, color=color, zorder=3, edgecolors="white", linewidths=1.5, alpha=0.9)

            for x_val, y_val in zip(xs, ys):
                ax.annotate(
                    f"{y_val:.2f}",
                    xy=(x_val, y_val),
                    xytext=(0, 11),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=color,
                    fontweight="bold",
                )

        ax.axhline(
            threshold,
            color=THRESHOLD_COLOR,
            linestyle="--",
            linewidth=1.2,
            zorder=1,
            label=f"{_label(language, 'threshold')} {threshold:.1f}",
        )
        ax.set_xticks(xs_sorted)
        ax.set_xticklabels([str(x) for x in xs_sorted], fontsize=10)
        ax.set_xlabel(_label(language, x_key), fontsize=9, labelpad=10)
        ax.set_ylabel(_label(language, "avg_score_tool"), fontsize=10)
        ax.set_title(_label(language, title_key), fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=9, framealpha=0, loc="lower left")
        ax.text(0.99, 0.02, _label(language, "bubble_note"), transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#777777")
        saver.save(filename)


def simple_tool_charts(df: pd.DataFrame, saver: Saver, labels: list[str], language: str) -> None:
    tc = df[df["metric_name"].str.lower().str.contains("tool correctness", na=False)].copy()
    if tc.empty:
        print("  grafico simples de tools ignorado: sem metrica Tool Correctness.")
        return

    series_keys = ["expected_called", "expected_missing", "extra_called"]
    colors = {
        "expected_called": "#2E7D32",
        "expected_missing": "#C62828",
        "extra_called": "#EF6C00",
    }

    rows: list[dict[str, object]] = []
    for model in labels:
        sub = tc[tc["modelo"] == model]
        expected_total = int(sub["expected_tools_list"].apply(len).sum())
        missing_total = int(sub["tools_missing_list"].apply(len).sum())
        extra_total = int(sub["tools_extra_list"].apply(len).sum())
        called_expected = max(expected_total - missing_total, 0)
        rows.append(
            {
                "model": model,
                "expected_called": called_expected,
                "expected_missing": missing_total,
                "extra_called": extra_total,
            }
        )

        values = [called_expected, missing_total, extra_total]
        x = np.arange(len(series_keys))
        _, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            x,
            values,
            color=[colors[key] for key in series_keys],
            width=0.58,
            zorder=3,
        )
        ax.bar_label(bars, labels=[str(v) for v in values], padding=4, fontsize=9, fontweight="bold")
        ax.set_title(_label(language, "tools_simple_model", model=model))
        ax.set_ylabel(_label(language, "tools_y"))
        ax.set_xticks(x)
        ax.set_xticklabels([_label(language, key) for key in series_keys], fontsize=9)
        ax.set_ylim(0, max(values + [1]) * 1.18)
        saver.save(f"tools_resumo_{safe_name(model)}.png")

    if len(rows) <= 1:
        return

    x = np.arange(len(rows))
    width = 0.22
    _, ax = plt.subplots(figsize=(max(10, len(rows) * 2.4), 5.5))
    for idx, key in enumerate(series_keys):
        values = [int(row[key]) for row in rows]
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=colors[key],
            zorder=3,
            label=_label(language, key),
        )
        ax.bar_label(bars, labels=[str(v) for v in values], padding=3, fontsize=8)

    ax.set_title(_label(language, "tools_simple_all"))
    ax.set_ylabel(_label(language, "tools_y"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["model"]) for row in rows], fontsize=9)
    max_value = max([int(row[key]) for row in rows for key in series_keys] + [1])
    ax.set_ylim(0, max_value * 1.22)
    ax.legend(fontsize=9, framealpha=0, loc="upper right")
    saver.save("tools_resumo_agregado.png")


def tool_charts_by_question(df: pd.DataFrame, saver: Saver, labels: list[str], language: str) -> None:
    question_col = get_question_column(df)
    if question_col is None:
        print("  tools por questao ignorado: CSV sem coluna numero/question/question_id/case_id.")
        return

    tc = df[df["metric_name"].str.lower().str.contains("tool correctness", na=False)].copy()
    if tc.empty:
        print("  tools por questao ignorado: sem metrica Tool Correctness.")
        return

    order = question_order(df, question_col)
    if not order:
        return

    labels_x = question_axis_labels(order)
    series_keys = ["expected_called", "expected_missing", "extra_called"]
    colors = {
        "expected_called": "#2E7D32",
        "expected_missing": "#C62828",
        "extra_called": "#EF6C00",
    }

    for model in labels:
        sub_model = tc[tc["modelo"] == model]
        if sub_model.empty:
            continue

        values_by_key: dict[str, list[int]] = {key: [] for key in series_keys}
        for question in order:
            sub_question = sub_model[sub_model[question_col] == question]
            expected_total = int(sub_question["expected_tools_list"].apply(len).sum())
            missing_total = int(sub_question["tools_missing_list"].apply(len).sum())
            extra_total = int(sub_question["tools_extra_list"].apply(len).sum())
            values_by_key["expected_called"].append(max(expected_total - missing_total, 0))
            values_by_key["expected_missing"].append(missing_total)
            values_by_key["extra_called"].append(extra_total)

        x = np.arange(len(order))
        width = 0.24
        _, ax = plt.subplots(figsize=(18, 9))

        for idx, key in enumerate(series_keys):
            offset = (idx - 1) * width
            values = values_by_key[key]
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[key],
                zorder=3,
                label=_label(language, key),
            )
            ax.bar_label(bars, labels=[str(v) if v else "" for v in values], padding=2, fontsize=6.5)

        ax.set_title(_label(language, "tools_question_model", model=model))
        ax.set_xlabel(_label(language, "question_x"))
        ax.set_ylabel(_label(language, "tools_y"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=8)
        max_value = max([value for values in values_by_key.values() for value in values] + [1])
        ax.set_ylim(0, max_value * 1.25)
        ax.legend(fontsize=9, framealpha=0, loc="upper right", ncols=3)
        saver.save(f"tools_questoes_{safe_name(model)}.png")


def question_scores(
    df: pd.DataFrame,
    saver: Saver,
    threshold: float,
    labels: list[str],
    language: str,
    metric_filter: str | None,
) -> None:
    question_col = get_question_column(df)
    if question_col is None:
        print("  grafico por questao ignorado: CSV sem coluna numero/question/question_id/case_id.")
        return

    plot_df = df.copy()
    if metric_filter:
        plot_df = plot_df[plot_df["metric_name"].str.lower() == metric_filter.lower()]
        if plot_df.empty:
            print(f"  grafico por questao ignorado: metrica nao encontrada ({metric_filter}).")
            return

    order = question_order(plot_df, question_col)
    if not order:
        return
    labels_x = question_axis_labels(order)

    for idx, model in enumerate(labels):
        sub = plot_df[plot_df["modelo"] == model]
        grouped = sub.groupby(question_col, dropna=False)["score"].mean().reindex(order)
        x = np.arange(len(order))
        color = PALETTE[idx % len(PALETTE)]

        _, ax = plt.subplots(figsize=(16, 9))
        ax.bar(x, grouped.values, color=color, width=0.72, zorder=3)
        ax.axhline(
            threshold,
            color=THRESHOLD_COLOR,
            linestyle="--",
            linewidth=1.4,
            zorder=4,
            label=f"{_label(language, 'threshold')} {threshold:.1f}",
        )
        ax.set_title(_label(language, "question_score", model=model))
        ax.set_xlabel(_label(language, "question_x"))
        ax.set_ylabel(_label(language, "avg_score"))
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, rotation=0, fontsize=8)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        _annotate_bars(ax, color=color)
        ax.legend(fontsize=9, framealpha=0)
        saver.save(f"questoes_30_{safe_name(model)}.png")

    if len(labels) <= 1:
        return

    width = min(0.8 / len(labels), 0.22)
    x = np.arange(len(order))
    _, ax = plt.subplots(figsize=(18, 9))
    for idx, model in enumerate(labels):
        color = PALETTE[idx % len(PALETTE)]
        sub = plot_df[plot_df["modelo"] == model]
        grouped = sub.groupby(question_col, dropna=False)["score"].mean().reindex(order)
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, grouped.values, color=color, width=width * 0.92, zorder=3, label=model)

    ax.axhline(
        threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.4,
        zorder=4,
        label=f"{_label(language, 'threshold')} {threshold:.1f}",
    )
    ax.set_title(_label(language, "question_score_all"))
    ax.set_xlabel(_label(language, "question_x"))
    ax.set_ylabel(_label(language, "avg_score"))
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=0, fontsize=8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.legend(fontsize=9, framealpha=0, loc="upper left", ncols=min(len(labels), 3))
    saver.save("questoes_30_agregado.png")


def pass_fail_charts(df: pd.DataFrame, saver: Saver, labels: list[str], language: str) -> None:
    metrics = sorted(df["metric_name"].dropna().unique())
    if not metrics:
        return

    pass_color = "#2E7D32"
    fail_color = "#C62828"

    for model in labels:
        sub = df[df["modelo"] == model]
        if sub.empty:
            continue

        passed_counts = []
        failed_counts = []
        pass_rates = []
        for metric in metrics:
            metric_rows = sub[sub["metric_name"] == metric]
            passed = int(metric_rows["passed"].sum())
            total = len(metric_rows)
            failed = total - passed
            passed_counts.append(passed)
            failed_counts.append(failed)
            pass_rates.append((passed / total * 100) if total else 0.0)

        x = np.arange(len(metrics))
        _, ax = plt.subplots(figsize=(max(8, len(metrics) * 2.6), 5))
        passed_bars = ax.bar(x, passed_counts, color=pass_color, width=0.62, zorder=3, label=_label(language, "passed"))
        failed_bars = ax.bar(x, failed_counts, bottom=passed_counts, color=fail_color, width=0.62, zorder=3, label=_label(language, "failed"))

        for xi, passed, failed, rate in zip(x, passed_counts, failed_counts, pass_rates):
            total = passed + failed
            ax.text(
                xi,
                total + max(0.25, total * 0.025),
                f"{passed}/{total}\n{rate:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
                fontweight="bold",
            )

        ax.bar_label(passed_bars, labels=[str(v) if v else "" for v in passed_counts], label_type="center", color="white", fontsize=8, fontweight="bold")
        ax.bar_label(failed_bars, labels=[str(v) if v else "" for v in failed_counts], label_type="center", color="white", fontsize=8, fontweight="bold")
        ax.set_title(_label(language, "pass_fail_model", model=model))
        ax.set_ylabel(_label(language, "pass_fail_y"))
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylim(0, max([p + f for p, f in zip(passed_counts, failed_counts)] + [1]) * 1.22)
        ax.legend(fontsize=9, framealpha=0, loc="upper right")
        saver.save(f"pass_fail_{safe_name(model)}.png")

    rows: list[dict[str, object]] = []
    for model in labels:
        sub = df[df["modelo"] == model]
        for metric in metrics:
            metric_rows = sub[sub["metric_name"] == metric]
            passed = int(metric_rows["passed"].sum())
            total = len(metric_rows)
            rows.append(
                {
                    "label": f"{model}\n{metric}",
                    "passed": passed,
                    "failed": total - passed,
                    "rate": (passed / total * 100) if total else 0.0,
                    "total": total,
                }
            )

    if not rows:
        return

    x = np.arange(len(rows))
    width = 0.68
    passed_counts = [int(row["passed"]) for row in rows]
    failed_counts = [int(row["failed"]) for row in rows]

    _, ax = plt.subplots(figsize=(max(12, len(rows) * 1.25), 6))
    passed_bars = ax.bar(x, passed_counts, color=pass_color, width=width, zorder=3, label=_label(language, "passed"))
    failed_bars = ax.bar(x, failed_counts, bottom=passed_counts, color=fail_color, width=width, zorder=3, label=_label(language, "failed"))

    for xi, row in zip(x, rows):
        total = int(row["total"])
        ax.text(
            xi,
            total + max(0.25, total * 0.025),
            f"{int(row['passed'])}/{total}\n{float(row['rate']):.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#333333",
        )

    ax.bar_label(passed_bars, labels=[str(v) if v else "" for v in passed_counts], label_type="center", color="white", fontsize=7, fontweight="bold")
    ax.bar_label(failed_bars, labels=[str(v) if v else "" for v in failed_counts], label_type="center", color="white", fontsize=7, fontweight="bold")
    ax.set_title(_label(language, "pass_fail_all"))
    ax.set_ylabel(_label(language, "pass_fail_y"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["label"]) for row in rows], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, max([p + f for p, f in zip(passed_counts, failed_counts)] + [1]) * 1.25)
    ax.legend(fontsize=9, framealpha=0, loc="upper right")
    saver.save("pass_fail_agregado.png")


def pass_fail_by_question_charts(
    df: pd.DataFrame, saver: Saver, labels: list[str], language: str
) -> None:
    question_col = get_question_column(df)
    if question_col is None:
        print("  pass/fail por questao ignorado: CSV sem coluna numero/question/question_id/case_id.")
        return

    order = question_order(df, question_col)
    if not order:
        return
    labels_x = question_axis_labels(order)

    pass_color = "#2E7D32"
    fail_color = "#C62828"

    rows: list[dict[str, object]] = []
    for model in labels:
        sub = df[df["modelo"] == model]
        if sub.empty:
            continue

        passed_counts = []
        failed_counts = []
        pass_rates = []
        for question in order:
            question_rows = sub[sub[question_col] == question]
            passed = int(question_rows["passed"].sum())
            total = len(question_rows)
            failed = total - passed
            passed_counts.append(passed)
            failed_counts.append(failed)
            pass_rates.append((passed / total * 100) if total else 0.0)
            rows.append(
                {
                    "label": f"{model}\n{labels_x[len(passed_counts) - 1]}",
                    "passed": passed,
                    "failed": failed,
                    "total": total,
                    "rate": (passed / total * 100) if total else 0.0,
                }
            )

        x = np.arange(len(order))
        _, ax = plt.subplots(figsize=(18, 9))
        passed_bars = ax.bar(
            x,
            passed_counts,
            color=pass_color,
            width=0.72,
            zorder=3,
            label=_label(language, "passed"),
        )
        failed_bars = ax.bar(
            x,
            failed_counts,
            bottom=passed_counts,
            color=fail_color,
            width=0.72,
            zorder=3,
            label=_label(language, "failed"),
        )

        for xi, passed, failed, rate in zip(x, passed_counts, failed_counts, pass_rates):
            total = passed + failed
            if total == 0:
                continue
            ax.text(
                xi,
                total + max(0.15, total * 0.035),
                f"{passed}/{total}\n{rate:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#333333",
            )

        ax.bar_label(
            passed_bars,
            labels=[str(v) if v else "" for v in passed_counts],
            label_type="center",
            color="white",
            fontsize=7,
            fontweight="bold",
        )
        ax.bar_label(
            failed_bars,
            labels=[str(v) if v else "" for v in failed_counts],
            label_type="center",
            color="white",
            fontsize=7,
            fontweight="bold",
        )
        ax.set_title(_label(language, "pass_fail_question_model", model=model))
        ax.set_xlabel(_label(language, "question_x"))
        ax.set_ylabel(_label(language, "pass_fail_y"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=8)
        ax.set_ylim(0, max([p + f for p, f in zip(passed_counts, failed_counts)] + [1]) * 1.25)
        ax.legend(fontsize=9, framealpha=0, loc="upper right")
        saver.save(f"pass_fail_questoes_{safe_name(model)}.png")

    if len(labels) <= 1 or not rows:
        return

    x = np.arange(len(rows))
    passed_counts = [int(row["passed"]) for row in rows]
    failed_counts = [int(row["failed"]) for row in rows]

    _, ax = plt.subplots(figsize=(max(18, len(rows) * 0.7), 8))
    passed_bars = ax.bar(
        x,
        passed_counts,
        color=pass_color,
        width=0.68,
        zorder=3,
        label=_label(language, "passed"),
    )
    failed_bars = ax.bar(
        x,
        failed_counts,
        bottom=passed_counts,
        color=fail_color,
        width=0.68,
        zorder=3,
        label=_label(language, "failed"),
    )

    for xi, row in zip(x, rows):
        total = int(row["total"])
        if total == 0:
            continue
        ax.text(
            xi,
            total + max(0.15, total * 0.035),
            f"{int(row['passed'])}/{total}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#333333",
        )

    ax.bar_label(
        passed_bars,
        labels=[str(v) if v else "" for v in passed_counts],
        label_type="center",
        color="white",
        fontsize=6,
        fontweight="bold",
    )
    ax.bar_label(
        failed_bars,
        labels=[str(v) if v else "" for v in failed_counts],
        label_type="center",
        color="white",
        fontsize=6,
        fontweight="bold",
    )
    ax.set_title(_label(language, "pass_fail_question_all"))
    ax.set_ylabel(_label(language, "pass_fail_y"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["label"]) for row in rows], rotation=60, ha="right", fontsize=6.5)
    ax.set_ylim(0, max([p + f for p, f in zip(passed_counts, failed_counts)] + [1]) * 1.28)
    ax.legend(fontsize=9, framealpha=0, loc="upper right")
    saver.save("pass_fail_questoes_agregado.png")


def write_summary(df: pd.DataFrame, output_dir: Path, threshold: float, labels: list[str], language: str) -> None:
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append(_label(language, "summary_title"))
    lines.append("=" * 62)
    lines.append("")
    lines.append(_label(language, "metrics_meaning"))
    lines.append("-" * 40)
    for metric in sorted(df["metric_name"].dropna().unique()):
        lines.append(f"\n{metric}")
        if language == "en":
            lines.append("  Evaluates one aspect of the model execution or answer quality.")
        else:
            lines.append("  Avalia um aspecto da execucao do modelo ou da qualidade da resposta.")

    lines.append("")
    lines.append("=" * 62)
    lines.append(_label(language, "results_by_model"))
    lines.append("=" * 62)

    for label in labels:
        sub = df[df["modelo"] == label]
        if sub.empty:
            continue
        lines.append(f"\nModelo: {label}" if language == "pt" else f"\nModel: {label}")
        lines.append("-" * 40)
        for metric in sorted(sub["metric_name"].dropna().unique()):
            ms = sub[sub["metric_name"] == metric]
            avg = ms["score"].mean()
            pr = ms["passed"].mean() * 100
            n_pass = int(ms["passed"].sum())
            n_total = len(ms)
            lines.append(f"  {metric}")
            lines.append(f"    average score : {avg:.3f}")
            lines.append(f"    pass rate     : {pr:.0f}% ({n_pass}/{n_total})")

    lines.append("")
    lines.append("=" * 62)
    lines.append(_label(language, "method_note"))
    lines.append("=" * 62)
    if language == "en":
        lines.append("Report judge models separately and discuss variance as a methodological limitation.")
    else:
        lines.append("Reporte os modelos juizes separadamente e discuta a variancia como limitacao metodologica.")

    out_path = output_dir / "resumo_metricas.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("  resumo_metricas.txt")


def latest_csv() -> Path:
    files = sorted((ROOT_DIR / "outputs").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("Nenhum CSV encontrado na pasta outputs.")
    return files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera graficos PNG + PDF a partir dos CSVs de metricas DeepEval.")
    parser.add_argument("--csv", type=Path, nargs="+", default=None, metavar="CSV", help="Um ou mais CSVs de metricas.")
    parser.add_argument("--labels", type=str, nargs="+", default=None, metavar="LABEL", help="Rotulo para cada CSV/modelo.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Pasta base de saida.")
    parser.add_argument("--run-name", type=str, default=None, help="Nome da subpasta desta execucao.")
    parser.add_argument("--overwrite", action="store_true", help="Usa exatamente --out-dir e permite sobrescrever arquivos.")
    parser.add_argument("--pdf-name", type=str, default="relatorio.pdf", help="Nome do PDF consolidado.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold de corte visual.")
    parser.add_argument("--language", choices=["pt", "en"], default="pt", help="Idioma dos titulos e legendas.")
    parser.add_argument("--question-metric", type=str, default=None, help="Metrica usada no grafico por questao. Padrao: media das metricas.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        csv_paths = args.csv
        labels = args.labels if args.labels else [p.stem for p in csv_paths]
        if len(labels) != len(csv_paths):
            raise ValueError("--labels deve ter o mesmo numero de itens que --csv.")
    else:
        csv_paths = [latest_csv()]
        labels = [csv_paths[0].stem]

    output_dir = resolve_output_dir(args.out_dir, args.overwrite, args.run_name)
    pdf_path = output_dir / args.pdf_name

    print(f"Modelos: {labels}")
    print(f"Saida:   {output_dir}")

    df = load_all(csv_paths, labels, args.threshold)
    write_question_mapping(df, output_dir)

    with Saver(output_dir, pdf_path) as saver:
        for idx, label in enumerate(labels):
            barras_por_modelo(df, saver, args.threshold, label, idx, args.language)
        barras_agregado(df, saver, args.threshold, args.language)

        for idx, label in enumerate(labels):
            slope_por_modelo(df, saver, args.threshold, label, idx, args.language)
        slope_agregado(df, saver, args.threshold, args.language)

        simple_tool_charts(df, saver, labels, args.language)
        tool_charts_by_question(df, saver, labels, args.language)

        question_scores(df, saver, args.threshold, labels, args.language, args.question_metric)
        pass_fail_charts(df, saver, labels, args.language)
        pass_fail_by_question_charts(df, saver, labels, args.language)
        write_summary(df, output_dir, args.threshold, labels, args.language)

    print(f"\nPDF: {pdf_path}")


if __name__ == "__main__":
    main()
