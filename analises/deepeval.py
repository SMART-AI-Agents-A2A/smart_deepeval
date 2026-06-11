from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "analises" / "graficos"

PALETTE = ["#185FA5", "#3B6D11", "#854F0B", "#993556", "#533BA7", "#0F6E56"]
THRESHOLD_COLOR = "#cc3333"
GRID_COLOR = "#e8e8e8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 130,
})


def split_pipe(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [x.strip() for x in value.split("|") if x.strip()]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["passed"] = df["passed"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["tools_missing_list"] = df.get("tools_missing", pd.Series(dtype=str)).apply(split_pipe)
    df["tools_extra_list"] = df.get("tools_extra", pd.Series(dtype=str)).apply(split_pipe)
    df["tools_called_list"] = df.get("tools_called", pd.Series(dtype=str)).apply(split_pipe)
    df["expected_tools_list"] = df.get("expected_tools", pd.Series(dtype=str)).apply(split_pipe)
    return df


def load_all(csv_paths: list[Path], labels: list[str]) -> pd.DataFrame:
    frames = []
    for path, label in zip(csv_paths, labels):
        df = load_csv(path)
        df["modelo"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fig_w(n_models: int, base: float = 9.0) -> float:
    return max(base, base + (n_models - 2) * 1.8)


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


def chart_01_score_medio(df: pd.DataFrame, saver: Saver, threshold: float) -> None:
    models = df["modelo"].unique()
    metrics = sorted(df["metric_name"].dropna().unique())
    n_models = len(models)
    n_metrics = len(metrics)
    w = fig_w(n_models)

    x = np.arange(n_metrics)
    width = 0.7 / n_models
    fig, ax = plt.subplots(figsize=(w, 5))

    for i, (model, color) in enumerate(zip(models, PALETTE)):
        sub = df[df["modelo"] == model]
        means = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, means, width=width * 0.92, color=color, label=model, zorder=3)
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=color)

    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.4,
               label=f"threshold {threshold:.1f}", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score médio")
    ax.set_title("Score médio por métrica e modelo")
    ax.legend(fontsize=9, framealpha=0)
    saver.save("01_score_medio_por_metrica.png")


def chart_02_pass_rate(df: pd.DataFrame, saver: Saver) -> None:
    models = df["modelo"].unique()
    metrics = sorted(df["metric_name"].dropna().unique())
    n_models = len(models)
    n_metrics = len(metrics)
    w = fig_w(n_models)

    x = np.arange(n_metrics)
    width = 0.7 / n_models
    fig, ax = plt.subplots(figsize=(w, 5))

    for i, (model, color) in enumerate(zip(models, PALETTE)):
        sub = df[df["modelo"] == model]
        rates = [sub.loc[sub["metric_name"] == m, "passed"].mean() * 100 for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, rates, width=width * 0.92, color=color, label=model, zorder=3)
        for bar, val in zip(bars, rates):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Pass rate")
    ax.set_title("Pass rate por métrica e modelo")
    ax.legend(fontsize=9, framealpha=0)
    saver.save("02_pass_rate_por_metrica.png")


def chart_03_linha_por_caso(df: pd.DataFrame, saver: Saver, threshold: float) -> None:
    metrics = sorted(df["metric_name"].dropna().unique())
    n_metrics = len(metrics)
    models = df["modelo"].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for model, color in zip(models, PALETTE):
            sub = (df[(df["modelo"] == model) & (df["metric_name"] == metric)]
                   .sort_values("numero"))
            if sub.empty:
                continue
            ax.plot(sub["numero"], sub["score"], color=color, linewidth=1.4,
                    label=model, zorder=3)
            ax.scatter(sub["numero"], sub["score"], color=color, s=22, zorder=4)

        ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2, zorder=2)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Nº do caso")
        ax.set_title(metric, fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Score")

    handles = [plt.Line2D([0], [0], color=c, linewidth=2, label=m)
               for m, c in zip(models, PALETTE)]
    handles.append(plt.Line2D([0], [0], color=THRESHOLD_COLOR, linewidth=1.4,
                               linestyle="--", label=f"threshold {threshold:.1f}"))
    fig.legend(handles=handles, loc="lower center", ncol=min(n_models + 1, 5),
               fontsize=9, framealpha=0, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Score por caso — linha com data points", fontsize=12, y=1.02)
    saver.save("03_score_por_caso_linha.png")


def chart_04_dispersao_score_tools(df: pd.DataFrame, saver: Saver, threshold: float) -> None:
    tc = df[df["metric_name"].str.lower().str.contains("tool correctness", na=False)].copy()
    if tc.empty:
        return

    tc["tools_extras_count"] = tc["tools_extra_list"].apply(len)
    tc["tools_missing_count"] = tc["tools_missing_list"].apply(len)

    models = tc["modelo"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, col, xlabel in zip(axes,
                                ["tools_extras_count", "tools_missing_count"],
                                ["Ferramentas a mais", "Ferramentas faltando"]):
        for model, color in zip(models, PALETTE):
            sub = tc[tc["modelo"] == model]
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(sub))
            ax.scatter(sub[col] + jitter, sub["score"],
                       color=color, alpha=0.65, s=40, label=model, zorder=3)

        ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--",
                   linewidth=1.2, zorder=2)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[0].set_ylabel("Tool Correctness score")
    axes[0].legend(fontsize=9, framealpha=0)
    fig.suptitle("Dispersão: Tool Correctness score vs excesso / falta de ferramentas", fontsize=11)
    saver.save("04_dispersao_tool_correctness.png")


def chart_05_heatmap_casos(df: pd.DataFrame, saver: Saver) -> None:
    models = list(df["modelo"].unique())
    metrics = sorted(df["metric_name"].dropna().unique())
    cases = sorted(df["numero"].dropna().unique())

    n_models = len(models)
    n_metrics = len(metrics)
    n_cases = len(cases)

    fig, axes = plt.subplots(1, n_models,
                              figsize=(max(4, n_metrics * 1.4) * n_models, max(8, n_cases * 0.32)),
                              sharey=True)
    if n_models == 1:
        axes = [axes]

    im = None
    for ax, model in zip(axes, models):
        sub = df[df["modelo"] == model]
        pivot = (sub.pivot_table(index="numero", columns="metric_name",
                                  values="score", aggfunc="mean")
                 .reindex(index=cases, columns=metrics)
                 .fillna(0))

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=8)
        ax.set_title(model, fontsize=10)
        if ax == axes[0]:
            ax.set_yticks(range(n_cases))
            ax.set_yticklabels([str(int(c)) for c in cases], fontsize=7)
            ax.set_ylabel("Nº do caso")

    if im is not None:
        fig.colorbar(im, ax=axes[-1], fraction=0.03, label="Score")
    fig.suptitle("Heatmap de scores por caso e métrica", fontsize=12, y=1.01)
    saver.save("05_heatmap_scores.png")


def chart_06_ferramentas_faltou_sobrou(df: pd.DataFrame, saver: Saver) -> None:
    tc = df[df["metric_name"].str.lower().str.contains("tool correctness", na=False)]
    if tc.empty:
        return

    models = list(tc["modelo"].unique())
    n = len(models)
    w = fig_w(n, base=11)
    fig, axes = plt.subplots(1, n, figsize=(w, 6), sharey=False)
    if n == 1:
        axes = [axes]

    all_tools: set[str] = set()
    for _, row in tc.iterrows():
        all_tools.update(row["tools_missing_list"])
        all_tools.update(row["tools_extra_list"])

    for ax, model in zip(axes, models):
        sub = tc[tc["modelo"] == model]
        missing: dict[str, int] = {}
        extra: dict[str, int] = {}
        for _, row in sub.iterrows():
            for t in row["tools_missing_list"]:
                missing[t] = missing.get(t, 0) + 1
            for t in row["tools_extra_list"]:
                extra[t] = extra.get(t, 0) + 1

        tools = sorted(all_tools)
        miss_vals = [missing.get(t, 0) for t in tools]
        extra_vals = [extra.get(t, 0) for t in tools]
        y = np.arange(len(tools))

        ax.barh(y, [-v for v in miss_vals], color="#e15759", label="faltou", zorder=3)
        ax.barh(y, extra_vals, color="#76b7b2", label="sobrou", zorder=3)
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([t.replace("smart_", "") for t in tools], fontsize=8)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("← faltou  |  sobrou →", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: str(abs(int(v)))))
        if ax == axes[0]:
            ax.legend(fontsize=8, framealpha=0)

    fig.suptitle("Ferramentas faltantes vs extras por modelo", fontsize=12)
    saver.save("06_ferramentas_faltou_sobrou.png")


def latest_csv() -> Path:
    files = sorted((ROOT_DIR / "outputs").glob("*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("Nenhum CSV encontrado na pasta outputs.")
    return files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera 6 gráficos PNG + PDF a partir dos CSVs de métricas DeepEval."
    )
    parser.add_argument("--csv", type=Path, nargs="+", default=None, metavar="CSV",
                        help="Um ou mais CSVs de métricas. Se omitido, usa o mais recente em outputs/.")
    parser.add_argument("--labels", type=str, nargs="+", default=None, metavar="LABEL",
                        help="Rótulo para cada CSV (mesmo número que --csv).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Pasta de saída.")
    parser.add_argument("--pdf-name", type=str, default="relatorio.pdf",
                        help="Nome do PDF consolidado.")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Threshold de corte visual (padrão: 0.7).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        csv_paths = args.csv
        labels = args.labels if args.labels else [p.stem for p in csv_paths]
        if len(labels) != len(csv_paths):
            raise ValueError("--labels deve ter o mesmo número de itens que --csv.")
    else:
        csv_paths = [latest_csv()]
        labels = [csv_paths[0].stem]

    print(f"Modelos: {labels}")
    print(f"Saída:   {args.out_dir}")

    df = load_all(csv_paths, labels)
    pdf_path = args.out_dir / args.pdf_name

    with Saver(args.out_dir, pdf_path) as saver:
        chart_01_score_medio(df, saver, args.threshold)
        chart_02_pass_rate(df, saver)
        chart_03_linha_por_caso(df, saver, args.threshold)
        chart_04_dispersao_score_tools(df, saver, args.threshold)
        chart_05_heatmap_casos(df, saver)
        chart_06_ferramentas_faltou_sobrou(df, saver)

    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()