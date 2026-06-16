from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path.cwd()
DEFAULT_OUTPUT_DIR = ROOT_DIR / "analises" / "graphs"

PASTEL = {
    "Proposed": "#A8E6B0",
    "LLM Without Proposed": "#F7B9B9",
    "Expected Called": "#B7D7F0",
    "Expected Missing": "#F6C7A8",
    "Extra Called": "#D9C2F0",
    "Passed": "#A8E6B0",
    "Failed": "#F7B9B9",
}
FALLBACK_COLORS = ["#A8E6B0", "#F7B9B9", "#B7D7F0", "#F6C7A8", "#D9C2F0", "#F6E6A8"]
ERROR_COLOR = "#6E6E6E"
GRID_COLOR = "#EEEEEE"

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
        "savefig.dpi": 300,
        "ps.fonttype": 42,
    }
)

TARGET_METRICS = {
    "Tool Correctness": ["tool correctness", "tool_correctness"],
    "Task Completion": ["task completion", "task completation", "task_completion", "task_completation"],
    "G-Eval": ["g-eval", "g eval", "geval", "g_eval"],
}

MODEL_COLUMNS = ["model", "modelo", "model_name", "llm_model", "llm", "judge_model"]
APPROACH_COLUMNS = ["approach", "variant", "condition", "configuration", "setup", "dataset", "grupo", "source", "tipo"]


def safe_name(value: str) -> str:
    value = value.strip().lower().replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "chart"


def split_pipe(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def normalize_text(value: object) -> str:
    return str(value).strip()


def normalize_approach(value: object) -> str:
    raw = normalize_text(value)
    key = raw.lower().replace("_", "-").strip()

    if key in {"smart", "proposed", "com proposta", "with proposed", "with-proposed"}:
        return "Proposed"

    no_data_keys = {
        "no data",
        "no-data",
        "nodata",
        "without data",
        "without-data",
        "sem dados",
        "sem-dados",
        "llm without proposed",
        "llm-without-proposed",
        "without proposed",
        "without-proposed",
    }
    if key in no_data_keys or "no data" in key or "without proposed" in key:
        return "LLM Without Proposed"

    return raw or "Proposed"


def parse_label(label: str) -> tuple[str | None, str | None]:
    for separator in ("::", "|", ";"):
        if separator in label:
            left, right = [part.strip() for part in label.split(separator, 1)]
            left_app = normalize_approach(left)
            right_app = normalize_approach(right)
            if left_app in {"Proposed", "LLM Without Proposed"}:
                return right or None, left_app
            if right_app in {"Proposed", "LLM Without Proposed"}:
                return left or None, right_app
            return left or None, right or None

    app = normalize_approach(label)
    if app in {"Proposed", "LLM Without Proposed"}:
        return None, app
    return label, None


def get_score_column(df: pd.DataFrame) -> str:
    for column in ("score", "simple_tool_score", "selection_score"):
        if column in df.columns:
            return column
    raise ValueError("CSV without score column. Expected: score, simple_tool_score or selection_score.")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def load_csv(path: Path, label: str, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    score_col = get_score_column(df)
    df["score"] = pd.to_numeric(df[score_col], errors="coerce")

    if "metric_name" not in df.columns:
        df["metric_name"] = "Score"
    df["metric_name"] = df["metric_name"].astype(str).str.strip()

    if "passed" in df.columns:
        df["passed"] = df["passed"].astype(str).str.lower().isin(["true", "1", "yes", "passed"])
    else:
        df["passed"] = df["score"] >= threshold

    for column in ("tools_missing", "tools_extra", "tools_called", "expected_tools"):
        if column not in df.columns:
            df[column] = ""

    label_model, label_approach = parse_label(label)

    model_col = first_existing_column(df, MODEL_COLUMNS)
    approach_col = first_existing_column(df, APPROACH_COLUMNS)

    if model_col:
        df["model"] = df[model_col].astype(str).str.strip()
    else:
        df["model"] = label_model or "Model"

    if approach_col:
        df["approach"] = df[approach_col].apply(normalize_approach)
    else:
        df["approach"] = label_approach or "Proposed"

    if label_model and model_col is None:
        df["model"] = label_model
    if label_approach and approach_col is None:
        df["approach"] = label_approach

    df["tools_missing_list"] = df["tools_missing"].apply(split_pipe)
    df["tools_extra_list"] = df["tools_extra"].apply(split_pipe)
    df["tools_called_list"] = df["tools_called"].apply(split_pipe)
    df["expected_tools_list"] = df["expected_tools"].apply(split_pipe)

    return df


def load_all(csv_paths: list[Path], labels: list[str], threshold: float) -> pd.DataFrame:
    frames = [load_csv(path, label, threshold) for path, label in zip(csv_paths, labels)]
    return pd.concat(frames, ignore_index=True)


def resolve_output_dir(base_dir: Path, overwrite: bool, run_name: str | None) -> Path:
    if overwrite:
        return base_dir

    stem = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / stem
    suffix = 2
    while output_dir.exists():
        output_dir = base_dir / f"{stem}_{suffix}"
        suffix += 1
    return output_dir


class Saver:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, stem: str) -> None:
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        for ext in ("png", "eps"):
            path = self.output_dir / f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
            print(f"  {path.name}")
        plt.close(fig)


def ci95(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / np.sqrt(len(clean)))


def metric_mask(df: pd.DataFrame, display_name: str) -> pd.Series:
    aliases = TARGET_METRICS[display_name]
    metric = df["metric_name"].astype(str).str.lower()
    mask = pd.Series(False, index=df.index)
    for alias in aliases:
        mask = mask | metric.str.contains(alias, regex=False, na=False)
    return mask


def ordered_models(df: pd.DataFrame) -> list[str]:
    return list(pd.unique(df["model"].dropna()))


def ordered_approaches(df: pd.DataFrame) -> list[str]:
    values = list(pd.unique(df["approach"].dropna()))
    preferred = ["Proposed", "LLM Without Proposed"]
    return [item for item in preferred if item in values] + [item for item in values if item not in preferred]


def color_for(label: str, idx: int) -> str:
    return PASTEL.get(label, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def set_score_axis(ax, means: list[float], errors: list[float]) -> None:
    top = max([m + e for m, e in zip(means, errors) if not np.isnan(m)] + [1.0])
    ax.set_ylim(0, max(1.05, top * 1.15))
    ax.set_ylabel("Average score")


def plot_metric(df: pd.DataFrame, metric_name: str, saver: Saver) -> None:
    sub = df[metric_mask(df, metric_name)].copy()
    if sub.empty:
        print(f"  skipped: {metric_name}")
        return

    models = ordered_models(sub)
    approaches = ordered_approaches(sub)
    x = np.arange(len(models))
    width = 0.72 / max(len(approaches), 1)

    fig_width = max(8, len(models) * 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    fig.suptitle(f"{metric_name}", fontsize=16, y=0.985)

    all_means: list[float] = []
    all_errors: list[float] = []

    for i, approach in enumerate(approaches):
        means = []
        errors = []
        for model in models:
            values = sub[(sub["model"] == model) & (sub["approach"] == approach)]["score"]
            means.append(float(values.mean()) if not values.empty else np.nan)
            errors.append(ci95(values))

        all_means.extend(means)
        all_errors.extend(errors)
        offset = (i - (len(approaches) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            yerr=errors,
            capsize=6,
            ecolor=ERROR_COLOR,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
            color=color_for(approach, i),
            edgecolor="#777777",
            linewidth=0.8,
            width=width * 0.9,
            label=approach,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    set_score_axis(ax, all_means, all_errors)
    ax.legend(frameon=False, fontsize=10)
    saver.save(fig, f"metric_{safe_name(metric_name)}")


def plot_tools(df: pd.DataFrame, saver: Saver) -> None:
    tc = df[metric_mask(df, "Tool Correctness")].copy()
    if tc.empty:
        print("  skipped: tools")
        return

    tc["Expected Called"] = tc.apply(
        lambda row: max(len(row["expected_tools_list"]) - len(row["tools_missing_list"]), 0), axis=1
    )
    tc["Expected Missing"] = tc["tools_missing_list"].apply(len)
    tc["Extra Called"] = tc["tools_extra_list"].apply(len)

    groups = tc[["model", "approach"]].drop_duplicates().to_dict("records")
    labels = [f"{row['model']}\n{row['approach']}" for row in groups]
    x = np.arange(len(groups))
    series = ["Expected Called", "Expected Missing", "Extra Called"]
    width = 0.72 / len(series)

    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.8), 6.1))
    fig.suptitle("Tool Correctness Aggregate - Tool Usage", fontsize=16, y=0.985)

    all_tops = []
    for i, key in enumerate(series):
        means = []
        errors = []
        for row in groups:
            values = tc[(tc["model"] == row["model"]) & (tc["approach"] == row["approach"])][key]
            means.append(float(values.mean()) if not values.empty else np.nan)
            errors.append(ci95(values))

        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            yerr=errors,
            capsize=6,
            ecolor=ERROR_COLOR,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
            color=color_for(key, i),
            edgecolor="#777777",
            linewidth=0.8,
            width=width * 0.9,
            label=key,
            zorder=3,
        )
        all_tops.extend([m + e for m, e in zip(means, errors) if not np.isnan(m)])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Average number of tools")
    ax.set_ylim(0, max(all_tops + [1]) * 1.18)
    ax.legend(frameon=False, fontsize=10)
    saver.save(fig, "tools_aggregate")


def binary_ci95(successes: pd.Series) -> float:
    values = successes.astype(float).dropna()
    n = len(values)
    if n < 2:
        return 0.0
    p = float(values.mean())
    return float(1.96 * np.sqrt(p * (1 - p) / n))


def plot_passed_failed(df: pd.DataFrame, saver: Saver) -> None:
    groups = df[["model", "approach"]].drop_duplicates().to_dict("records")
    labels = [f"{row['model']}\n{row['approach']}" for row in groups]
    x = np.arange(len(groups))
    series = ["Passed", "Failed"]
    width = 0.34

    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.8), 6.1))
    fig.suptitle("Pass/Fail Aggregate", fontsize=16, y=0.985)

    for i, key in enumerate(series):
        means = []
        errors = []
        for row in groups:
            values = df[(df["model"] == row["model"]) & (df["approach"] == row["approach"])]["passed"]
            if key == "Passed":
                binary = values.astype(bool)
            else:
                binary = ~values.astype(bool)
            means.append(float(binary.mean()) if not binary.empty else np.nan)
            errors.append(binary_ci95(binary))

        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            means,
            yerr=errors,
            capsize=6,
            ecolor=ERROR_COLOR,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
            color=color_for(key, i),
            edgecolor="#777777",
            linewidth=0.8,
            width=width * 0.9,
            label=key,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=10)
    saver.save(fig, "passed_failed_aggregate")


def latest_csv() -> Path:
    files = sorted((ROOT_DIR / "outputs").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No CSV found in outputs/.")
    return files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PNG and EPS charts for DeepEval metrics.")
    parser.add_argument("--csv", type=Path, nargs="+", default=None, metavar="CSV", help="One or more CSV files.")
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        metavar="LABEL",
        help='Labels for each CSV. Use "Model::Proposed" or "Model::LLM Without Proposed" when needed.',
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--run-name", type=str, default=None, help="Subfolder name for this run.")
    parser.add_argument("--overwrite", action="store_true", help="Write directly into --out-dir.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Pass/fail threshold when the CSV has no passed column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        csv_paths = args.csv
        labels = args.labels if args.labels else [path.stem for path in csv_paths]
        if len(labels) != len(csv_paths):
            raise ValueError("--labels must have the same number of items as --csv.")
    else:
        csv_paths = [latest_csv()]
        labels = [csv_paths[0].stem]

    output_dir = resolve_output_dir(args.out_dir, args.overwrite, args.run_name)
    df = load_all(csv_paths, labels, args.threshold)
    saver = Saver(output_dir)

    print(f"Output: {output_dir}")

    for metric_name in TARGET_METRICS:
        plot_metric(df, metric_name, saver)

    plot_tools(df, saver)
    plot_passed_failed(df, saver)


if __name__ == "__main__":
    main()
