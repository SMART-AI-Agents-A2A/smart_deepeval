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
    last_x = x[-1]

    fig, ax = plt.subplots(figsize=(9, 5))

    end_scores: list[tuple[float, int]] = []
    all_scores: list[list[float]] = []

    for idx, (model, color) in enumerate(zip(models, PALETTE)):
        sub = df[df["modelo"] == model]
        scores = [sub.loc[sub["metric_name"] == m, "score"].mean() for m in metrics]
        all_scores.append(scores)
        if not np.isnan(scores[-1]):
            end_scores.append((scores[-1], idx))

    end_scores_sorted = sorted(end_scores, key=lambda t: t[0])
    label_y: dict[int, float] = {}
    min_gap = 0.06
    placed: list[float] = []
    for score, idx in end_scores_sorted:
        y = score
        for p in sorted(placed):
            if abs(y - p) < min_gap:
                y = p + min_gap
        placed.append(y)
        label_y[idx] = y

    for idx, (model, color) in enumerate(zip(models, PALETTE)):
        scores = all_scores[idx]
        ax.plot(x, scores, color=color, linewidth=2.2, zorder=3,
                marker="o", markersize=13, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=2)

        for xi, score in zip(x, scores):
            if not np.isnan(score):
                ax.text(xi, score + 0.05, f"{score:.2f}",
                        ha="center", va="bottom", fontsize=8.5,
                        color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

        if idx in label_y:
            ly = label_y[idx]
            end_score = scores[-1]
            if abs(ly - end_score) > 0.01:
                ax.plot([last_x + 0.05, last_x + 0.10], [end_score, ly],
                        color=color, linewidth=0.8, zorder=2)
            ax.text(last_x + 0.12, ly, model,
                    ha="left", va="center", fontsize=8.5,
                    color=color, fontweight="bold")

    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=1.3, zorder=2,
               label=f"threshold {threshold:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.5, last_x + 1.2)
    ax.set_ylabel("Score médio")
    ax.set_title("Score médio por métrica e modelo — agregado")
    ax.legend(fontsize=9, framealpha=0, loc="upper left")
    saver.save("slope_agregado.png")


def _tool_avg_by_count(tc: pd.DataFrame, col: str) -> dict[int, tuple[float, int]]:
    from collections import defaultdict
    groups: dict[int, list[float]] = defaultdict(list)
    for _, row in tc.iterrows():
        groups[int(row[col])].append(float(row["score"]))
    return {k: (sum(v) / len(v), len(v)) for k, v in groups.items()}


def dispersao_por_modelo(df: pd.DataFrame, saver: Saver, threshold: float, modelo: str, idx: int) -> None:
    tc = df[(df["modelo"] == modelo) &
            df["metric_name"].str.lower().str.contains("tool correctness", na=False)].copy()
    if tc.empty:
        return

    tc["extras"] = tc["tools_extra_list"].apply(len)
    tc["missing"] = tc["tools_missing_list"].apply(len)
    color = PALETTE[idx % len(PALETTE)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, titulo, legenda_x in zip(
        axes,
        ["extras", "missing"],
        [
            "Score médio vs ferramentas a mais",
            "Score médio vs ferramentas faltando",
        ],
        [
            "Nº de ferramentas extras chamadas\n(0 = chamou só as necessárias  |  7 = chamou 7 a mais do que precisava)",
            "Nº de ferramentas esperadas não chamadas\n(0 = chamou todas  |  3 = deixou de chamar 3 necessárias)",
        ],
    ):
        stats = _tool_avg_by_count(tc, col)
        xs = sorted(stats.keys())
        ys = [stats[x][0] for x in xs]
        ns = [stats[x][1] for x in xs]
        sizes = [max(60, n * 40) for n in ns]

        ax.plot(xs, ys, color=color, linewidth=1.8, zorder=2, alpha=0.6)
        sc = ax.scatter(xs, ys, s=sizes, color=color, zorder=3,
                        edgecolors="white", linewidths=1.5, alpha=0.9)

        for x, y, n in zip(xs, ys, ns):
            ax.text(x, y + 0.04, f"{y:.2f}\n(n={n})",
                    ha="center", va="bottom", fontsize=7.5, color=color)

        ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--",
                   linewidth=1.2, zorder=1, label=f"threshold {threshold:.1f}")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=9)
        ax.set_xlabel(legenda_x, fontsize=9, labelpad=8)
        ax.set_ylabel("Score médio (Tool Correctness)", fontsize=9)
        ax.set_title(titulo, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, framealpha=0)

    fig.suptitle(f"Tool Correctness — impacto de ferramentas extras e faltando  |  {modelo}", fontsize=11)
    safe = modelo.replace("/", "-").replace(" ", "_")
    saver.save(f"dispersao_{safe}.png")


def dispersao_agregado(df: pd.DataFrame, saver: Saver, threshold: float) -> None:
    tc = df[df["metric_name"].str.lower().str.contains("tool correctness", na=False)].copy()
    if tc.empty:
        return

    tc["extras"] = tc["tools_extra_list"].apply(len)
    tc["missing"] = tc["tools_missing_list"].apply(len)
    models = list(tc["modelo"].unique())

    for col, titulo, legenda_x, filename in [
        (
            "extras",
            "Score médio vs ferramentas a mais — todos os modelos",
            "Nº de ferramentas extras chamadas além das esperadas\n"
            "(0 = chamou só as necessárias  |  7 = chamou 7 a mais do que precisava naquele caso)",
            "dispersao_extras_agregado.png",
        ),
        (
            "missing",
            "Score médio vs ferramentas faltando — todos os modelos",
            "Nº de ferramentas esperadas que não foram chamadas\n"
            "(0 = chamou todas  |  3 = deixou de chamar 3 ferramentas necessárias naquele caso)",
            "dispersao_missing_agregado.png",
        ),
    ]:
        all_xs: set[int] = set()
        for model in models:
            sub = tc[tc["modelo"] == model]
            all_xs.update(int(v) for v in sub[col].unique())
        xs_sorted = sorted(all_xs)

        fig, ax = plt.subplots(figsize=(max(9, len(xs_sorted) * 1.2 + 2), 5))

        for model, color in zip(models, PALETTE):
            sub = tc[tc["modelo"] == model]
            stats = _tool_avg_by_count(sub, col)
            xs = sorted(stats.keys())
            ys = [stats[x][0] for x in xs]
            ns = [stats[x][1] for x in xs]
            sizes = [max(55, n * 38) for n in ns]

            ax.plot(xs, ys, color=color, linewidth=1.8, zorder=2, alpha=0.6, label=model)
            ax.scatter(xs, ys, s=sizes, color=color, zorder=3,
                       edgecolors="white", linewidths=1.5, alpha=0.9)

            for x, y, n in zip(xs, ys, ns):
                ax.annotate(
                    f"{y:.2f}",
                    xy=(x, y),
                    xytext=(0, 11),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold",
                )

        ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--",
                   linewidth=1.2, zorder=1, label=f"threshold {threshold:.1f}")
        ax.set_xticks(xs_sorted)
        ax.set_xticklabels([str(x) for x in xs_sorted], fontsize=10)
        ax.set_xlabel(legenda_x, fontsize=9, labelpad=10)
        ax.set_ylabel("Score médio (Tool Correctness)", fontsize=10)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=9, framealpha=0, loc="lower left")
        ax.text(0.99, 0.02,
                "Tamanho do ponto proporcional ao nº de casos naquele grupo.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#777777")
        saver.save(filename)


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