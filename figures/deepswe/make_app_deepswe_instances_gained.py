#!/usr/bin/env python3
"""Figure: DeepSWE v1.1 rollouts passed per task at every effort level, one heatmap per model.

    python figures/deepswe/make_app_deepswe_instances_gained.py

The repeated-run counterpart of `make_instances_gained.py`. DeepSWE v1.1 publishes about four
rollouts per task per effort level, so the gains and losses between effort levels can be read
off several runs, not one. A row per effort level, a column per task, and the cell's colour is
how many of that task's rollouts passed (0-4); a task lost at a higher level shows as a dark
cell above a pale one in the same column. Columns are sorted per model by total passes
(`solve_counts.load`), so a column position is not the same task in two panels. A small orange
dot marks a cell whose task had fewer than four rollouts at that level.

GLM-5.2 was run at `high` and `max`; DeepSeek V4 Flash has only `max` and is left out.

Writes to output/figures/:
    app_deepswe_instances_gained.pdf
    deepswe/deepswe_tasks.csv          the per-task counts drawn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import solve_counts  # noqa: E402
import style  # noqa: E402
from common import paths  # noqa: E402

OUT = paths.output_dir("figures") / "app_deepswe_instances_gained.pdf"

# Display names, in the paper's order. DeepSeek V4 Flash has a single level, so no comparison.
MODELS = ["GPT-5.6-luna", "GPT-5.6-terra", "GPT-5.6-sol", "Claude Sonnet 5", "GLM-5.2"]

PAPER, INK, MUTED = "#FFFFFF", "#1F232E", "#7C8798"
BASE = 8.5            # pt; the figure is drawn at about \linewidth


def draw(ax, d: dict) -> None:
    p, n = d["passed"], d["rollouts"]
    ax.imshow(p, aspect="auto", cmap=solve_counts.CMAP, norm=solve_counts.NORM, interpolation="nearest")
    for i in range(1, len(d["levels"])):
        ax.axhline(i - 0.5, color=PAPER, lw=1.0)
    short = np.argwhere(n < solve_counts.MAX_ROLLOUTS)
    if len(short):
        ax.scatter(short[:, 1], short[:, 0] - 0.28, s=3, color=solve_counts.SHORT_DOT, zorder=3,
                   linewidths=0)
    ax.set_yticks(range(len(d["levels"])))
    ax.set_yticklabels(d["levels"], fontsize=BASE)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=3)
    for side in ax.spines.values():
        side.set_color(INK)
        side.set_linewidth(0.6)


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plt.rcParams.update({
        "font.family": style.use_font(), "font.size": BASE, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    tasks = solve_counts.build_rows()
    print(f"wrote {solve_counts.write_tasks_csv(tasks)}")
    data = solve_counts.load(tasks)
    rows = [m for m in MODELS if m in data]

    # Inches, so the gaps stay fixed whatever each model's ladder length.
    width, row_h, gap_h, top_h, bottom_h = 6.8, 0.22, 0.12, 0.46, 0.22
    height = (top_h + bottom_h + sum(row_h * len(data[m]["levels"]) for m in rows)
              + gap_h * (len(rows) - 1))
    left, right = 0.125, 0.995
    fig = plt.figure(figsize=(width, height))

    # Horizontal colour bar, centred above the panels.
    cb_w, cb_h = 0.34, 0.075
    cax = fig.add_axes(((1 - cb_w) / 2 + (left - (1 - right)) / 2, 1 - 0.26 / height, cb_w,
                        cb_h / height))
    sm = plt.cm.ScalarMappable(cmap=solve_counts.CMAP, norm=solve_counts.NORM)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=range(solve_counts.MAX_ROLLOUTS + 1))
    cb.ax.tick_params(labelsize=BASE, length=0, pad=2)
    cb.outline.set_linewidth(0.5)
    cax.set_title("Rollouts passed", fontsize=BASE, pad=3)

    y = height - top_h
    for model in rows:
        d = data[model]
        h = row_h * len(d["levels"])
        ax = fig.add_axes((left, (y - h) / height, right - left, h / height))
        draw(ax, d)
        key = model.lower().replace(" ", "-")
        # Model name, rotated, clear of the widest effort label ("medium").
        ax.annotate(style.label_of(key).replace("Claude ", "Claude\n"), xy=(0, 0.5),
                    xycoords="axes fraction", xytext=(-40, 0), textcoords="offset points",
                    rotation=90, ha="right", va="center", fontsize=BASE, fontweight="bold",
                    color=style.style_of(key).color, linespacing=1.0)
        y -= h + gap_h
    ax.set_xlabel("Tasks, sorted by total rollouts passed", fontsize=BASE, labelpad=4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
