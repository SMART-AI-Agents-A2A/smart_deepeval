from __future__ import annotations

import argparse
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
                ha="center", va="bottom", fontsize=8, color=c,
            )


def barras_por_modelo(df: pd.DataFrame, saver: Saver, threshold: float, modelo: str, idx: int) -> None:
    sub = df[df["modelo"] == modelo]
    metrics = sorted(sub["metric_name"].dropna().unique())
    means = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
    color = PALETTE[idx % len(PALETTE)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(metrics, means, color=color, width=0.5, zorder=3)
    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.4,
               label=f"threshold {threshold:.1f}", zorder=4)
    _annotate_bars(ax, color=color)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score médio")
    ax.set_title(f"Score médio por métrica — {modelo}")
    ax.legend(fontsize=9, framealpha=0)
    safe = modelo.replace("/", "-").replace(" ", "_")
    saver.save(f"barras_{safe}.png")


def barras_agregado(df: pd.DataFrame, saver: Saver, threshold: float) -> None:
    models = list(df["modelo"].unique())
    metrics = sorted(df["metric_name"].dropna().unique())
    n_models = len(models)
    n_metrics = len(metrics)
    width = 0.7 / n_models
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(max(8, 3 * n_metrics + n_models), 5))
    for i, (model, color) in enumerate(zip(models, PALETTE)):
        sub = df[df["modelo"] == model]
        means = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, means, width=width * 0.92, color=color, label=model, zorder=3)
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=color)

    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.4,
               label=f"threshold {threshold:.1f}", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score médio")
    ax.set_title("Score médio por métrica e modelo — agregado")
    ax.legend(fontsize=9, framealpha=0)
    saver.save("barras_agregado.png")


def slope_por_modelo(df: pd.DataFrame, saver: Saver, threshold: float, modelo: str, idx: int) -> None:
    sub = df[df["modelo"] == modelo]
    metrics = sorted(sub["metric_name"].dropna().unique())
    scores = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
    color = PALETTE[idx % len(PALETTE)]
    x = list(range(len(metrics)))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, scores, color=color, linewidth=2.2, zorder=3,
            marker="o", markersize=13, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=2)
    for xi, score in zip(x, scores):
        ax.text(xi, score + 0.03, f"{score:.2f}",
                ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")

    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=1.3, zorder=2, label=f"threshold {threshold:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylabel("Score médio")
    ax.set_title(f"Score por métrica — {modelo}")
    ax.legend(fontsize=9, framealpha=0)
    safe = modelo.replace("/", "-").replace(" ", "_")
    saver.save(f"slope_{safe}.png")


def slope_agregado(df: pd.DataFrame, saver: Saver, threshold: float) -> None:
    models = list(df["modelo"].unique())
    metrics = sorted(df["metric_name"].dropna().unique())
    x = list(range(len(metrics)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, color in zip(models, PALETTE):
        sub = df[df["modelo"] == model]
        scores = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
        ax.plot(x, scores, color=color, linewidth=2.2, zorder=3,
                marker="o", markersize=13, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=2, label=model)
        for xi, score in zip(x, scores):
            ax.text(xi, score + 0.03, f"{score:.2f}",
                    ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=1.3, zorder=2, label=f"threshold {threshold:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylabel("Score médio")
    ax.set_title("Score por métrica — todos os modelos")
    ax.legend(fontsize=9, framealpha=0, loc="lower right")
    saver.save("slope_agregado.png")


def dispersao_por_modelo(df: pd.DataFrame, saver: Saver, threshold: float, modelo: str, idx: int) -> None:
    tc = df[(df["modelo"] == modelo) &
            df["metric_name"].str.lower().str.contains("tool correctness", na=False)].copy()
    if tc.empty:
        return

    tc["extras"] = tc["tools_extra_list"].apply(len)
    tc["missing"] = tc["tools_missing_list"].apply(len)
    color = PALETTE[idx % len(PALETTE)]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, col, xlabel in zip(axes,
                                ["extras", "missing"],
                                ["Ferramentas a mais", "Ferramentas faltando"]):
        jitter = rng.uniform(-0.12, 0.12, len(tc))
        ax.scatter(tc[col] + jitter, tc["score"],
                   color=color, alpha=0.7, s=50, zorder=3)
        ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2, zorder=2)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[0].set_ylabel("Tool Correctness score")
    fig.suptitle(f"Dispersão: Tool Correctness — {modelo}", fontsize=11)
    safe = modelo.replace("/", "-").replace(" ", "_")
    saver.save(f"dispersao_{safe}.png")


def dispersao_agregado(df: pd.DataFrame, saver: Saver, threshold: float) -> None:
    tc = df[df["metric_name"].str.lower().str.contains("tool correctness", na=False)].copy()
    if tc.empty:
        return

    tc["extras"] = tc["tools_extra_list"].apply(len)
    tc["missing"] = tc["tools_missing_list"].apply(len)
    models = list(tc["modelo"].unique())
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, col, xlabel in zip(axes,
                                ["extras", "missing"],
                                ["Ferramentas a mais", "Ferramentas faltando"]):
        for model, color in zip(models, PALETTE):
            sub = tc[tc["modelo"] == model]
            jitter = rng.uniform(-0.12, 0.12, len(sub))
            ax.scatter(sub[col] + jitter, sub["score"],
                       color=color, alpha=0.65, s=45, label=model, zorder=3)
        ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2, zorder=2)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[0].set_ylabel("Tool Correctness score")
    axes[0].legend(fontsize=9, framealpha=0)
    fig.suptitle("Dispersão: Tool Correctness — todos os modelos", fontsize=11)
    saver.save("dispersao_agregado.png")


def write_summary(df: pd.DataFrame, output_dir: Path, threshold: float, labels: list[str]) -> None:
    metrics_info: dict[str, str] = {}
    for m in sorted(df["metric_name"].dropna().unique()):
        if "geval" in m.lower() or "g-eval" in m.lower():
            metrics_info[m] = (
                "Mede se a resposta final atingiu o objetivo do usuario.\n"
                "  Um LLM compara a resposta com a esperada e avalia se a conclusao\n"
                "  foi correta, direta e tecnicamente justificada.\n"
                "  Score alto = resposta atingiu o objetivo.\n"
                "  Score baixo = resposta vaga, errada ou sem conclusao clara."
            )
        elif "tool" in m.lower():
            metrics_info[m] = (
                "Mede se o agente chamou as ferramentas certas.\n"
                "  Combina dois criterios: (1) deterministico — quantas esperadas foram\n"
                "  chamadas vs. total chamado (excesso penaliza); (2) por LLM — se a\n"
                "  selecao foi otima para a tarefa. Score final = minimo dos dois."
            )
        elif "task" in m.lower():
            metrics_info[m] = (
                "Mede se a tarefa foi cumprida, analisando o trace de execucao.\n"
                "  Um LLM extrai o objetivo e o resultado do trace e avalia se a tarefa\n"
                "  foi satisfeita. Diferente do GEval, foca em 'a tarefa foi feita?'\n"
                "  e nao em 'a resposta bate com o gabarito?'"
            )
        else:
            metrics_info[m] = "Metrica customizada."

    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("RESUMO DA AVALIACAO — API SMART com DeepEval")
    lines.append("=" * 62)
    lines.append("")
    lines.append("O QUE CADA METRICA SIGNIFICA")
    lines.append("-" * 40)
    for name, desc in metrics_info.items():
        lines.append(f"\n{name}")
        lines.append(f"  {desc}")

    lines.append("")
    lines.append("-" * 40)
    lines.append("PASS RATE")
    lines.append(f"  Percentual de casos com score >= threshold ({threshold:.1f}).")
    lines.append("  Exemplo: 47% = 14 de 30 casos aprovados.")
    lines.append("")
    lines.append("SCORE MEDIO")
    lines.append("  Media aritmetica dos scores de todos os casos.")
    lines.append("  Varia de 0.0 (pessimo) a 1.0 (perfeito).")
    lines.append("")
    lines.append("=" * 62)
    lines.append("RESULTADOS POR MODELO")
    lines.append("=" * 62)

    for label in labels:
        sub = df[df["modelo"] == label]
        if sub.empty:
            continue
        lines.append(f"\nModelo: {label}")
        lines.append("-" * 40)
        for metric in sorted(sub["metric_name"].dropna().unique()):
            ms = sub[sub["metric_name"] == metric]
            avg = ms["score"].mean()
            pr = ms["passed"].mean() * 100
            n_pass = int(ms["passed"].sum())
            n_total = len(ms)
            status = "PASSOU" if avg >= threshold else "abaixo do threshold"
            lines.append(f"  {metric}")
            lines.append(f"    score medio : {avg:.3f}  ({status})")
            lines.append(f"    pass rate   : {pr:.0f}%  ({n_pass}/{n_total} casos)")

    lines.append("")
    lines.append("=" * 62)
    lines.append("NOTA METODOLOGICA")
    lines.append("=" * 62)
    lines.append("")
    lines.append("A escolha do modelo juiz influencia os resultados.")
    lines.append("Para o artigo: reporte os modelos separadamente e discuta a")
    lines.append("variancia entre eles como limitacao metodologica.")

    out_path = output_dir / "resumo_metricas.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  resumo_metricas.txt")
    print("")
    print("\n".join(lines))


def latest_csv() -> Path:
    files = sorted((ROOT_DIR / "outputs").glob("*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("Nenhum CSV encontrado na pasta outputs.")
    return files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera graficos PNG + PDF a partir dos CSVs de metricas DeepEval."
    )
    parser.add_argument("--csv", type=Path, nargs="+", default=None, metavar="CSV",
                        help="Um ou mais CSVs de metricas.")
    parser.add_argument("--labels", type=str, nargs="+", default=None, metavar="LABEL",
                        help="Rotulo para cada CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Pasta de saida.")
    parser.add_argument("--pdf-name", type=str, default="relatorio.pdf",
                        help="Nome do PDF consolidado.")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Threshold de corte visual (padrao: 0.7).")
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

    print(f"Modelos: {labels}")
    print(f"Saida:   {args.out_dir}")

    df = load_all(csv_paths, labels)
    pdf_path = args.out_dir / args.pdf_name

    with Saver(args.out_dir, pdf_path) as saver:
        for idx, label in enumerate(labels):
            barras_por_modelo(df, saver, args.threshold, label, idx)
        barras_agregado(df, saver, args.threshold)

        for idx, label in enumerate(labels):
            slope_por_modelo(df, saver, args.threshold, label, idx)
        slope_agregado(df, saver, args.threshold)

        for idx, label in enumerate(labels):
            dispersao_por_modelo(df, saver, args.threshold, label, idx)
        dispersao_agregado(df, saver, args.threshold)

        write_summary(df, args.out_dir, args.threshold, labels)

    print(f"\nPDF: {pdf_path}")


if __name__ == "__main__":
    main()