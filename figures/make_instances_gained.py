#!/usr/bin/env python3
"""Figure: what one step up the effort ladder gains and what it loses, per model per benchmark.

    python figures/make_instances_gained.py

Rows are benchmarks, columns are models. For each pair of neighbouring effort levels the same
units are scored at both, so every unit the step moves lands in one of two bars: solved at the
higher level and not the lower (solid, right), or solved at the lower level and not the higher
(hollow, left). Nested levels would leave every hollow bar empty.

Colour is the model (`style.MODELS`); the legend uses a neutral grey pair. Per-unit verdicts
come from the run trees through `ladders.ladders` (reasoning levels only, the benchmark's own
unit count). On AIME and GPQA "solved" means correct on at least one of the level's attempts.
A model that skips a level (Kimi K3's low/high/max) takes a longer step into the same slot.

Writes to output/figures/:
    instances_gained.pdf                  Terminal-Bench and SWE-Bench Pro (main text)
    instances_gained_other_datasets.pdf   AIME, GPQA and corrected SWE-Bench Pro (appendix)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import style  # noqa: E402
from common import paths  # noqa: E402
from datasets import rung_of  # noqa: E402
from ladders import PRO_CORRECTED_KEY, RUNGS, Ladder, ladders  # noqa: E402

# Benchmark rows: the agent loops for the main text, everything else for the appendix. Both
# grids use the same model columns.
ROWS = [("aime_2026", "AIME"), ("gpqa_diamond", "GPQA"),
        ("terminal_bench", "TerminalBench v2.1"), ("swebench_pro_pilot", "SWEBench-Pro")]
AGENTIC = [ROWS[2], ROWS[3]]
OTHER_DATASETS = [ROWS[0], ROWS[1], (PRO_CORRECTED_KEY, "SWEBench-Pro Corrected")]

# Column titles too long for one line at this panel width.
TITLE_WRAP = {"deepseek-v4-flash": "DeepSeek\nV4 Flash",
              "claude-sonnet-5": "Claude\nSonnet 5"}
# Row labels short enough to stand on one rotated line beside a row.
ROW_LABEL = {"terminal_bench": "Terminal-Bench", "swebench_pro_pilot": "SWE-Bench Pro",
             PRO_CORRECTED_KEY: "SWE-Bench Pro\n(corrected)"}

PAPER = "#FFFFFF"
INK = "#1F232E"
MUTED = "#7C8798"
ZERO = "#B9BFC9"      # the colour a bare `0` is written in: nothing moved, not a sign

BASE = 7.0            # pt; the figure is drawn near \textwidth


def _fade(color: str, keep: float) -> tuple[float, float, float]:
    """`color` mixed toward the surface, keeping `keep` of it (the hollow-bar fill)."""
    base, paper = np.array(mcolors.to_rgb(color)), np.array(mcolors.to_rgb(PAPER))
    return tuple(paper + keep * (base - paper))


# --------------------------------------------------------------------------- drawing

SHORT = {"low": "low", "medium": "med", "high": "high", "xhigh": "xhigh"}


def steps_of(lad: Ladder) -> dict[int, str]:
    """slot -> "a→b" for each step this ladder takes; the slot is the HIGHER level's rung.

    A model missing a level takes a longer step into the same slot (Kimi K3's low→high lands
    where medium→high does), so the slot's label is written from the model's own ladder rather
    than from the full rung list.
    """
    return {RUNGS.index(rung_of(b)) - 1: f"{SHORT[rung_of(a)]}→{SHORT[rung_of(b)]}"
            for a, b in zip(lad.efforts, lad.efforts[1:])}


def _ticks(lim: int) -> list[int]:
    """Three ticks, -s, 0, s, at the largest round number inside the row's bar range."""
    s = max([t for t in (1, 2, 3, 5, 10, 15, 20, 30, 50) if t <= lim] or [1])
    return [-s, 0, s]


def draw_panel(ax, lad: Ladder | None, col: str, lim: int, span: tuple[int, int],
               shared: dict[int, str], show_labels: bool) -> None:
    """One model on one benchmark: a diverging bar per step and its two counts.

    `shared` is the row's step label per slot. A model that skips a level takes a longer step
    into the same slot (Kimi K3's low→high lands where medium→high does); that step is named
    in the panel itself, in grey, so the shared tick label is never read as the step taken.
    """
    if lad is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "no ladder", transform=ax.transAxes, fontsize=BASE, color=ZERO,
                ha="center", va="center")
        return
    steps = steps_of(lad)
    slots = sorted(steps)
    for a, b in zip(lad.efforts, lad.efforts[1:]):
        slot = RUNGS.index(rung_of(b)) - 1
        gain = len(lad.resolved[b] - lad.resolved[a])
        lost = len(lad.resolved[a] - lad.resolved[b])
        if steps[slot] != shared[slot]:
            # Tucked under the lost count, left of the zero line, where nothing else is drawn.
            ax.annotate(steps[slot], (-lost, slot), textcoords="offset points",
                        xytext=(-2, -5.5), ha="right", va="top", fontsize=BASE - 1.5,
                        color=MUTED, style="italic", zorder=5)
        ax.barh(slot, gain, height=0.62, color=col, zorder=3)
        ax.barh(slot, -lost, height=0.62, facecolor=_fade(col, 0.20), edgecolor=col,
                linewidth=0.8, zorder=3)
        # A step that moved nothing one way is a bare grey `0`: "+0" would read as a sign.
        ax.annotate(f"+{gain}" if gain else "0", (gain, slot), textcoords="offset points",
                    xytext=(2, 0), ha="left", va="center", fontsize=BASE,
                    color=col if gain else ZERO, zorder=5)
        ax.annotate(f"−{lost}" if lost else "0", (-lost, slot), textcoords="offset points",
                    xytext=(-2, 0), ha="right", va="center", fontsize=BASE,
                    color=col if lost else ZERO, zorder=5)
    # The zero line spans the steps this model HAS: on a shorter ladder it would otherwise hang
    # over slots the model never ran.
    ax.vlines(0, min(slots) - 0.45, max(slots) + 0.45, color=INK, lw=0.8, zorder=4)
    # Room on both sides for the ±N labels.
    ax.set_xlim(-lim * 1.35, lim * 1.45)
    ax.patch.set_alpha(0.0)   # a label that spills past the edge is not painted over
    ax.set_ylim(span[1] + 0.5, span[0] - 0.5)
    if show_labels:
        ax.set_yticks(sorted(shared))
        ax.set_yticklabels([shared[i] for i in sorted(shared)], fontsize=BASE, color=MUTED)
    else:
        ax.set_yticks([])
    ax.set_xticks(_ticks(lim))
    ax.tick_params(axis="x", length=2, width=0.6, pad=2, labelsize=BASE - 0.5)
    ax.tick_params(axis="y", length=0, pad=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_bounds(-lim, lim)


def render(rows: list[tuple[str, str]], lads: dict[str, list[Ladder]], models: list[str],
           out: Path) -> None:
    """One grid: `rows` benchmarks down, `models` across, legend underneath."""
    n_steps = len(RUNGS) - 1
    # One symmetric x scale per benchmark: losses and gains comparable within it, and benchmarks
    # on different denominators deliberately not comparable with each other.
    lim = {}
    for key, _ in rows:
        flows = [max(len(l.resolved[b] - l.resolved[a]), len(l.resolved[a] - l.resolved[b]))
                 for l in lads[key] for a, b in zip(l.efforts, l.efforts[1:])]
        lim[key] = max(max(flows), 3)
    labels = {i: f"{SHORT[RUNGS[i]]}→{SHORT[RUNGS[i + 1]]}" for i in range(n_steps)}

    # Fixed margins in inches, so a two-row grid is not padded like a four-row one: room for the
    # column titles above, the x ticks and the legend below, and tight gaps between rows.
    row_h, gap_h, top_h, bottom_h = 0.74, 0.20, 0.30, 0.33
    height = top_h + bottom_h + len(rows) * row_h + (len(rows) - 1) * gap_h
    fig, axes = plt.subplots(len(rows), len(models), figsize=(6.8, height), squeeze=False)
    fig.subplots_adjust(left=0.125, right=0.992, top=1 - top_h / height, bottom=bottom_h / height,
                        wspace=0.02, hspace=gap_h / row_h)
    for r, (key, title) in enumerate(rows):
        for c, model in enumerate(models):
            ax = axes[r, c]
            lad = next((l for l in lads[key] if l.model == model), None)
            col = style.style_of(model).color
            draw_panel(ax, lad, col, lim[key], (0, n_steps - 1), labels, c == 0)
            if r == 0:
                ax.set_title(TITLE_WRAP.get(model, style.label_of(model)), pad=4,
                             fontsize=BASE + 1, fontweight="bold", color=col, linespacing=1.1)
            if c == 0:
                ax.set_ylabel(ROW_LABEL.get(key, title), fontsize=BASE + 0.5, fontweight="bold",
                              color=INK, labelpad=6)
                if lad is None:   # axis("off") hides the label; restore it
                    ax.yaxis.set_visible(True)
                    ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(False)

    ex = "#8A93A2"  # a neutral stand-in in the legend: a model colour there would name a model
    fig.legend(handles=[Patch(facecolor=ex, label="solved at the higher level, not the lower"),
                        Patch(facecolor=_fade(ex, 0.20), edgecolor=ex, linewidth=0.8,
                              label="solved at the lower level, not the higher")],
               loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=2, frameon=False,
               fontsize=BASE + 0.5, handletextpad=0.5, columnspacing=2.0, borderpad=0,
               handlelength=1.4, handleheight=1.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plt.rcParams.update({
        "font.family": style.use_font(), "font.size": BASE, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    lads = {key: ladders(key) for key, _ in ROWS}
    # Columns from the four primary benchmarks, so the main-text and appendix grids line up.
    models = sorted({lad.model for ls in lads.values() for lad in ls}, key=style.model_sort_key)
    lads[PRO_CORRECTED_KEY] = ladders(PRO_CORRECTED_KEY)
    out_dir = paths.output_dir("figures")
    render(AGENTIC, lads, models, out_dir / "instances_gained.pdf")
    render(OTHER_DATASETS, lads, models, out_dir / "instances_gained_other_datasets.pdf")


if __name__ == "__main__":
    main()
