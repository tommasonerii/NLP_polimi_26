"""
Generate one accuracy-progress chart for each PoliMillionaire category.

The curated experiment list below follows the notebook progression in
project/notebooks and the notes in README.md.  Missing category/run pairs are
left out intentionally: News and Philosophy/Psychology were added later, and
some runs were focused on a single category.  Category/run points with fewer
than five tried answers are filtered out by default because their accuracy is
too noisy for a progress chart.

Run from the project root:

    python3 project/src/plot_accuracy_progress.py
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


CATEGORIES = [
    "Entertainment",
    "Ancient History and Politics",
    "Science and Nature",
    "Maths",
    "Philosophy and Psychology",
    "News",
]

MIN_BEST_TOTAL = 10
MIN_TOTAL = 5


@dataclass(frozen=True)
class Experiment:
    order: int
    label: str
    filename: str
    notebook: str
    note: str


# Chosen to tell the development story without plotting every near-duplicate
# exploratory run.  Category-specific files are fine: the plotting step simply
# skips categories that are not present in that CSV.
EXPERIMENTS: list[Experiment] = [
    Experiment(
        0,
        "First option",
        "first_option_comp_0.csv",
        "00_api_smoke_test.ipynb",
        "Baseline naive Entertainment",
    ),
    Experiment(
        1,
        "First option",
        "first_option_comp_1.csv",
        "00_api_smoke_test.ipynb",
        "Baseline naive Ancient History and Politics",
    ),
    Experiment(
        2,
        "First option",
        "first_option_comp_2.csv",
        "00_api_smoke_test.ipynb",
        "Baseline naive Science and Nature",
    ),
    Experiment(
        3,
        "First option",
        "first_option_comp_3.csv",
        "00_api_smoke_test.ipynb",
        "Baseline naive Maths",
    ),
    Experiment(
        10,
        "TF-IDF",
        "simplewiki_tfidf_no_llm_all_competitions.csv",
        "01_quiz_tfidf_no_llm.ipynb",
        "SimpleWiki TF-IDF, no LLM",
    ),
    Experiment(
        20,
        "BM25",
        "simplewiki_bm25_no_llm_all_competitions.csv",
        "02_quiz_bm25_no_llm.ipynb",
        "SimpleWiki BM25, no LLM",
    ),
    Experiment(
        30,
        "KELM BM25",
        "simplewiki_kelm_bm25_no_llm_all_competitions.csv",
        "03_quiz_bm25_multi_index_no_llm.ipynb",
        "SimpleWiki + KELM BM25, no LLM",
    ),
    Experiment(
        40,
        "BERT rerank",
        "simplewiki_kelm_bert_no_llm_all_competitions_colab.csv",
        "05_quiz_bm25_multi_index_bert_no_llm_colab.ipynb",
        "BM25 multi-index + BERT reranker, no LLM",
    ),
    Experiment(
        50,
        "BERT + LLM/tools",
        "simplewiki_kelm_bm25_bert_llm_agentic_tools_colab.csv",
        "06_quiz_bm25_bert_llm_agentic_tools_colab.ipynb",
        "RAG + small LLM + initial agentic tools",
    ),
    Experiment(
        80,
        "Qwen GGUF",
        "run_qwen35_gguf_all_competitions.csv",
        "08_hybrid_pipeline.ipynb",
        "Qwen3.5 GGUF hybrid RAG",
    ),
    Experiment(
        90,
        "Qwen + math tools",
        "run_qwen35_gguf_math_tools_v2_all_competitions.csv",
        "09_hybrid_pipeline_math_tools.ipynb",
        "Hybrid RAG plus early Maths tool pipeline",
    ),
    Experiment(
        100,
        "Agentic tools",
        "run_qwen35_gguf_agentic_tools_v3_all_competitions.csv",
        "10_agentic_math_tools_prof_style.ipynb / 11_agentic_math_router_hardened.ipynb",
        "Agentic Maths tool router and hardening baseline",
    ),
    Experiment(
        120,
        "Validated V1",
        "run_qwen35_gguf_validated_tools_option_retrieval.csv",
        "12_validated_tools_option_retrieval.ipynb",
        "Validated tools + option-wise retrieval V1",
    ),
    Experiment(
        122,
        "Validated V2",
        "run_qwen35_gguf_validated_tools_option_retrieval_v2.csv",
        "12_V2_validated_tools_option_retrieval.ipynb",
        "GBNF, adaptive option retrieval, Maths fixes",
    ),
    Experiment(
        124,
        "News V4",
        "run_qwen35_gguf_validated_tools_option_retrieval_v4.csv",
        "12_V4_validated_tools_option_retrieval.ipynb",
        "News-focused run with headline-aware retrieval",
    ),
    Experiment(
        125,
        "Knowledge V5",
        "run_v5.csv",
        "12_V5-kaggle.ipynb / 12_V5_complete.ipynb",
        "Unified retrieval + answer-first micro-reasoning for knowledge categories",
    ),
    Experiment(
        126,
        "External V6",
        "run_qwen35_q8_qwen3reranker06b_external_bm25s_v6.csv",
        "12_V6_validated_tools_option_retrieval.ipynb",
        "External BM25S, Qwen3 reranker, News/Philosophy support",
    ),
    Experiment(
        130,
        "Math 7B",
        "run_qwen35_math_7B.csv",
        "12_V3_math_1M.ipynb",
        "Math-focused Qwen Math 7B run",
    ),
    Experiment(
        132,
        "Math 1M",
        "run_12_V3_math_1M.csv",
        "12_V3_math_1M.ipynb",
        "Math-focused semantic router/tool-recall run",
    ),
    Experiment(
        140,
        "Math V8",
        "run_v8.csv",
        "12-v8-maths.ipynb",
        "Latest local Maths-focused run present in logs",
    ),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_log(path: Path) -> pd.DataFrame | None:
    """Read a CSV log and normalize the boolean correctness column."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[skip] {path.name}: cannot read CSV ({exc})")
        return None

    required = {"competition_name", "correct"}
    if not required.issubset(df.columns):
        print(f"[skip] {path.name}: missing columns {sorted(required - set(df.columns))}")
        return None

    correct = (
        df["correct"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
    )
    if correct.isna().any():
        numeric = pd.to_numeric(df["correct"], errors="coerce")
        correct = correct.fillna(numeric)

    df = df.copy()
    df["_correct"] = correct
    df = df.dropna(subset=["competition_name", "_correct"])
    return df


def build_summary(
    logs_dir: Path,
    experiments: Iterable[Experiment],
    min_total: int = MIN_TOTAL,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for exp in experiments:
        log_path = logs_dir / exp.filename
        if not log_path.exists():
            print(f"[skip] {exp.filename}: file not found")
            continue

        df = read_log(log_path)
        if df is None or df.empty:
            continue

        grouped = (
            df.groupby("competition_name", dropna=False)["_correct"]
            .agg(correct="sum", total="count")
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            category = str(row.competition_name)
            if category not in CATEGORIES:
                continue
            total = int(row.total)
            correct = int(row.correct)
            if total <= 0:
                continue
            if total < min_total:
                print(
                    f"[skip] {exp.filename} / {category}: "
                    f"only {total} tried answers (< {min_total})"
                )
                continue
            rows.append(
                {
                    "order": exp.order,
                    "experiment": exp.label,
                    "filename": exp.filename,
                    "notebook": exp.notebook,
                    "note": exp.note,
                    "category": category,
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("No usable log rows found. Check --logs-dir and CSV schemas.")

    return summary.sort_values(["category", "order", "experiment"]).reset_index(drop=True)


def save_summary(summary: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "accuracy_progress_summary.csv"
    summary.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return out_path


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def annotate_points(ax: plt.Axes, x_values: list[int], y_values: list[float], labels: list[str]) -> None:
    for x, y, label in zip(x_values, y_values, labels):
        offset = 10 if y < 0.92 else -16
        va = "bottom" if offset > 0 else "top"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color="#30343b",
        )


def plot_category(summary: pd.DataFrame, category: str, out_dir: Path) -> Path | None:
    data = summary[summary["category"] == category].copy()
    if data.empty:
        print(f"[skip] {category}: no data")
        return None

    data = data.sort_values("order").reset_index(drop=True)
    data["x"] = range(len(data))
    data["label"] = data.apply(lambda r: f"{int(r.correct)}/{int(r.total)}", axis=1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_width = max(9.5, min(18, 0.85 * len(data) + 4))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))

    ax.plot(
        data["x"],
        data["accuracy"],
        marker=None,
        linewidth=2.5,
        color="#2f6fbd",
    )
    high_sample = data["total"] >= MIN_BEST_TOTAL
    ax.scatter(
        data.loc[high_sample, "x"],
        data.loc[high_sample, "accuracy"],
        s=56,
        color="#2f6fbd",
        edgecolor="#2f6fbd",
        zorder=3,
    )
    ax.scatter(
        data.loc[~high_sample, "x"],
        data.loc[~high_sample, "accuracy"],
        s=56,
        color="white",
        edgecolor="#2f6fbd",
        linewidth=2,
        zorder=3,
    )
    ax.fill_between(data["x"], data["accuracy"], 0, color="#2f6fbd", alpha=0.08)
    annotate_points(
        ax,
        data["x"].astype(int).tolist(),
        data["accuracy"].astype(float).tolist(),
        data["label"].tolist(),
    )

    ax.set_title(f"Accuracy progress - {category}", fontsize=15, weight="bold", pad=14)
    ax.set_ylabel("Accuracy (correct / total questions)")
    ax.set_xlabel("Experiment / notebook stage")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([i / 10 for i in range(0, 11)])
    ax.set_yticklabels([f"{i * 10}%" for i in range(0, 11)])
    ax.set_xticks(data["x"])
    ax.set_xticklabels(data["experiment"], rotation=35, ha="right")
    ax.margins(x=0.03)

    if len(data) == 1:
        ax.text(
            0.5,
            0.08,
            "Only one log with this category is available in logs/.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#5f6673",
        )
    elif category in {"News", "Philosophy and Psychology"}:
        ax.text(
            0.01,
            0.05,
            "This category was introduced later, so early baseline logs are absent.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#5f6673",
        )

    best_pool = data[data["total"] >= MIN_BEST_TOTAL]
    best_label = f"Best (>={MIN_BEST_TOTAL}q)" if not best_pool.empty else "Best"
    if best_pool.empty:
        best_pool = data
    best = best_pool.loc[best_pool["accuracy"].idxmax()]
    last = data.iloc[-1]
    improvement = (float(last["accuracy"]) - float(data.iloc[0]["accuracy"])) * 100
    best_text = (
        f"{best_label}: {best['experiment']} "
        f"({float(best['accuracy']) * 100:.1f}%, {int(best['correct'])}/{int(best['total'])})"
    )
    last_text = (
        f"First-to-last change: {improvement:+.1f} pp"
        if len(data) > 1
        else "Single available run"
    )
    ax.text(
        0.99,
        0.05,
        f"{best_text}\n{last_text}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#30343b",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d9dee7"},
    )

    if (~high_sample).any():
        ax.text(
            0.01,
            0.99,
            f"Open markers: n < {MIN_BEST_TOTAL}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#5f6673",
        )

    fig.tight_layout()
    out_path = out_dir / f"accuracy_progress_{slugify(category)}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_all_categories(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for category in CATEGORIES:
        path = plot_category(summary, category, out_dir)
        if path is not None:
            paths.append(path)
    return paths


def plot_overview(summary: pd.DataFrame, out_dir: Path) -> Path:
    """One compact overview useful for reports/slides."""
    pivot = summary.copy()
    pivot["point_label"] = pivot["experiment"] + " (" + pivot["total"].astype(str) + "q)"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharey=True)
    axes = axes.flatten()

    for ax, category in zip(axes, CATEGORIES):
        data = pivot[pivot["category"] == category].sort_values("order").reset_index(drop=True)
        if data.empty:
            ax.axis("off")
            ax.set_title(category)
            continue
        data["x"] = range(len(data))
        high_sample = data["total"] >= MIN_BEST_TOTAL
        ax.plot(data["x"], data["accuracy"], linewidth=2, color="#2f6fbd")
        ax.scatter(
            data.loc[high_sample, "x"],
            data.loc[high_sample, "accuracy"],
            s=34,
            color="#2f6fbd",
            edgecolor="#2f6fbd",
            zorder=3,
        )
        ax.scatter(
            data.loc[~high_sample, "x"],
            data.loc[~high_sample, "accuracy"],
            s=34,
            color="white",
            edgecolor="#2f6fbd",
            linewidth=1.5,
            zorder=3,
        )
        ax.set_title(category, fontsize=12, weight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(data["x"])
        ax.set_xticklabels(data["experiment"], rotation=45, ha="right", fontsize=8)
        ax.set_yticks([i / 10 for i in range(0, 11, 2)])
        ax.set_yticklabels([f"{i * 10}%" for i in range(0, 11, 2)])
        ax.grid(axis="x", visible=False)

    fig.suptitle("Accuracy progress by category", fontsize=16, weight="bold", y=0.995)
    fig.tight_layout()
    out_path = out_dir / "accuracy_progress_overview.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def print_console_summary(summary: pd.DataFrame, output_paths: list[Path], summary_path: Path) -> None:
    print("\nAccuracy summary by selected experiment:")
    view = summary.copy()
    view["accuracy_pct"] = (view["accuracy"] * 100).round(1)
    for category in CATEGORIES:
        data = view[view["category"] == category].sort_values("order")
        if data.empty:
            continue
        print(f"\n{category}")
        for row in data.itertuples(index=False):
            print(
                f"  {row.experiment:<20} {row.accuracy_pct:>5.1f}% "
                f"({int(row.correct)}/{int(row.total)})  [{row.filename}]"
            )

    print("\nSaved:")
    print(f"  {summary_path}")
    for path in output_paths:
        print(f"  {path}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Generate accuracy-progress charts for PoliMillionaire logs."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=root / "logs",
        help="Directory containing CSV logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "reports" / "figures" / "accuracy_progress",
        help="Directory where charts and summary CSV are written.",
    )
    parser.add_argument(
        "--no-overview",
        action="store_true",
        help="Skip the combined overview figure.",
    )
    parser.add_argument(
        "--min-total",
        type=int,
        default=MIN_TOTAL,
        help="Minimum tried answers required for a category/run point.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_dir = args.logs_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not logs_dir.exists():
        raise SystemExit(f"Logs directory not found: {logs_dir}")

    summary = build_summary(logs_dir, EXPERIMENTS, min_total=args.min_total)
    summary_path = save_summary(summary, out_dir)
    output_paths = plot_all_categories(summary, out_dir)
    if not args.no_overview:
        output_paths.append(plot_overview(summary, out_dir))

    if any(math.isnan(value) for value in summary["accuracy"].astype(float)):
        raise SystemExit("Unexpected NaN accuracy values found.")

    print_console_summary(summary, output_paths, summary_path)


if __name__ == "__main__":
    main()
