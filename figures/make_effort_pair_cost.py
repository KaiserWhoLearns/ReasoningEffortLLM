#!/usr/bin/env python3
"""Figure: two effort levels that solve as many tasks, naive laddering beside them, and what
each costs a user — cost on the left axis and tasks solved on the right.

    python figures/make_effort_pair_cost.py
    python figures/make_effort_pair_cost.py --model claude-sonnet-5 --lo none --hi xhigh

One model on one agentic benchmark, two of its effort levels with the same solve count, and the
naive-laddering policy (`routing/make_laddering.py`): start at `low`, and on a failed check rerun
the task at the top level. Each of the three gets a pair of bars: mean cost per task as the API
bill alone (`API`), then the API bill plus the user's waiting time (`full`; `common/latency.py`:
decode seconds at the model's table throughput, priced at the active wage). Each bar is stacked
like `make_solvable_cost.py`'s (input & context re-reads / hidden reasoning / visible output /
waiting) and labelled with its dollars and its ratio to the higher fixed level. Tasks solved is
on the right axis, one dot per policy with its 95% Wilson interval.

Costs are over all units of the run: the two levels solve the same number of units, so the mean
per task is the price of the same headline accuracy. Laddering's cost per task is the start
level's split on every unit plus the top level's on the units it escalated, read from
`output/laddering/laddering_choices.csv` (jump mode, `low` arm); its API total is checked
against `laddering_summary.csv`. The paired overlap of the two fixed levels (solved at both /
only at one) and McNemar's exact p are printed.

The default pair is GPT-5.6-terra `high` vs `xhigh` on Terminal-Bench.

Writes output/figures/effort_pair.pdf.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_solvable_cost as S  # noqa: E402  (the stacked-bar split and its styling)
import style  # noqa: E402
from common import latency, paths  # noqa: E402
from datasets import DATASETS  # noqa: E402

LADDER_DIR = paths.output_dir("laddering")
LADDER_COLOR = "#17836B"      # laddering's colour in the self-routing figure
TITLES = dict(S.ROWS)
PARTS = ("input", "reasoning", "output", "wait")
API = PARTS[:3]
BASE = S.BASE
BAR_W = 0.36                  # two bars per level, API and full, side by side
PAIR_GAP = 0.2               # half the distance between a level's two bar centres


@dataclass
class Bar:
    label: str                # tick text
    color: str
    solved: set[str]
    n_units: int
    parts: dict[str, float]   # mean $ per unit: input / reasoning / output / wait

    def total(self, keys: tuple[str, ...]) -> float:
        return sum(self.parts[k] for k in keys)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(pct, lo, hi) — the same 95% Wilson interval the Terminal-Bench loader reports."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (centre - half), 100 * (centre + half)


def unit_parts(model: str, key: str, run) -> dict[str, dict[str, float]]:
    """{unit: $ split} over every unit of `run`, as `make_solvable_cost.split_unit` splits it."""
    v = dict(run.values)
    if key == "terminal_bench":
        v["reasoning"] = S.tb_reasoning_per_task(model, run.effort)
    units = sorted(v["cost"])
    share = None
    if any(u not in v.get("reasoning", {}) for u in units):
        if run.effort == "none":
            share = 0.0
        else:
            have = [u for u in v.get("reasoning", {}) if u in v["output"]]
            tot = sum(v["output"][u] for u in have)
            share = sum(v["reasoning"][u] for u in have) / tot if tot else 0.0
    out = {u: S.split_unit(model, key, run.effort, v, u, share) for u in units}
    out = {u: p for u, p in out.items() if p}
    assert len(out) == run.n_units, (model, run.effort, len(out), run.n_units)
    return out


def mean_parts(per_unit: dict[str, dict[str, float]]) -> dict[str, float]:
    return {k: st.mean(p[k] for p in per_unit.values()) for k in PARTS}


def ladder_bar(key: str, model: str, runs: dict) -> Bar:
    """Naive laddering, `low` start, jump to the top on failure — its per-task split and solves."""
    def rows(name):
        with (LADDER_DIR / name).open(newline="") as f:
            return [r for r in csv.DictReader(f) if r["dataset"] == key and r["model"] == model
                    and r["mode"] == "jump" and r["arm"] == "low"]
    (summary,) = rows("laddering_summary.csv")
    picks = rows("laddering_choices.csv")
    start, top = summary["start_effort"], summary["top_effort"]
    per = {e: unit_parts(model, key, runs[e]) for e in (start, top)}
    assert len(picks) == int(summary["n_units"]) == len(per[start])
    total = {k: 0.0 for k in PARTS}
    for c in picks:
        escalated = c["escalated"] == "True"
        assert c["start_effort"] == start and c["ended_at"] == (top if escalated else start)
        for k in PARTS:
            total[k] += per[start][c["unit"]][k] + (per[top][c["unit"]][k] if escalated else 0.0)
    api = sum(total[k] for k in API)
    assert abs(api - float(summary["ladder_cost"])) < 0.01, (model, api, summary["ladder_cost"])
    solved = {c["unit"] for c in picks if c["resolved"] == "True"}
    assert len(solved) == int(summary["ladder_resolved"])
    return Bar("ladder", LADDER_COLOR, solved, len(picks),
               {k: v / len(picks) for k, v in total.items()})


def style_axes(ax) -> None:
    ax.tick_params(axis="y", length=0, pad=2, labelsize=BASE - 0.5)
    ax.tick_params(axis="x", length=0, pad=2, labelsize=BASE - 0.5)
    ax.grid(True, axis="y", color=S.GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(S.INK)


def draw_combined(bars: list[Bar], ref: Bar):
    """Cost bars on the left axis, tasks solved on the right, one group per level."""
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    fig.subplots_adjust(left=0.14, right=0.87, top=0.97, bottom=0.27)
    top = max(b.total(PARTS) for b in bars)
    halo = [patheffects.withStroke(linewidth=2.2, foreground=S.PAPER)]
    ticks, labels = [], []
    for i, b in enumerate(bars):
        for j, (name, keys) in enumerate((("API", API), ("full", PARTS))):
            x = i + (2 * j - 1) * PAIR_GAP
            bottom = 0.0
            for k in keys:
                ax.bar(x, b.parts[k], BAR_W, bottom=bottom, zorder=3,
                       **S.segment_style(b.color, k))
                bottom += b.parts[k]
            ax.text(x, bottom + 0.015 * top, f"${bottom:.2f}\n×{bottom / ref.total(keys):.2f}",
                    ha="center", va="bottom", fontsize=BASE - 1.5, color=S.INK,
                    linespacing=1.0, zorder=8, path_effects=halo)
            ticks.append(x)
            labels.append(name)
        ax.text(i, -0.1, b.label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=BASE, color=S.INK)
    ax.set_xticks(ticks, labels)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_ylim(0, math.ceil(top * 1.6))      # whole dollars; the default pair reads $0-$4
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(style.money_tick))
    ax.set_ylabel("Cost per task (bars)", fontsize=BASE, labelpad=3)
    style_axes(ax)
    ax.tick_params(axis="x", labelsize=BASE - 1.5)

    # Tasks solved on the right axis: one dot per level, joined, with its Wilson interval.
    # Drawn on the cost axes in cost units (a twin axes always paints over its host, whatever
    # the zorder), so the bar labels can sit above the line; the twin only carries the ticks.
    lo_pct, hi_pct = 40, 100
    ymax = ax.get_ylim()[1]
    to_cost = lambda pct: (pct - lo_pct) / (hi_pct - lo_pct) * ymax  # noqa: E731
    pts = [wilson(len(b.solved), b.n_units) for b in bars]
    xs = list(range(len(bars)))
    ax.plot(xs, [to_cost(p[0]) for p in pts], color=S.INK, lw=1.0, zorder=6)
    for x, b, (pct, lo, hi) in zip(xs, bars, pts):
        ax.errorbar(x, to_cost(pct), yerr=[[to_cost(pct) - to_cost(lo)], [to_cost(hi) - to_cost(pct)]],
                    fmt="none", ecolor=S.INK, elinewidth=0.7, capsize=2, capthick=0.7, zorder=6)
        ax.scatter(x, to_cost(pct), s=26, facecolor=b.color, edgecolor=S.INK, linewidth=0.8,
                   zorder=7)
        ax.annotate(f"{pct:.1f}%", (x, to_cost(hi)), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=BASE - 1, color=S.INK, zorder=8,
                    path_effects=halo)
    acc = ax.twinx()
    acc.set_ylim(lo_pct, hi_pct)
    acc.yaxis.set_major_locator(MultipleLocator(20))
    acc.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}%"))
    acc.set_ylabel("Tasks solved (dots)", fontsize=BASE, labelpad=3)
    acc.tick_params(axis="y", length=0, pad=2, labelsize=BASE - 0.5)
    for side in ("top", "right", "left", "bottom"):
        acc.spines[side].set_visible(False)
    acc.grid(False)

    names = {"input": "Input & context re-reads", "reasoning": "Reasoning", "output": "Output",
             "wait": "Waiting"}
    handles = [Patch(label=names[p], **S.segment_style(S.LEGEND_GREY, p)) for p in PARTS]
    handles.append(Line2D([], [], color=S.INK, lw=1.0, marker="o", markersize=4,
                          markerfacecolor=S.LEGEND_GREY, markeredgecolor=S.INK,
                          label="Tasks solved"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3,
               frameon=False, fontsize=BASE - 1, handletextpad=0.4, columnspacing=1.0,
               borderpad=0, handlelength=1.4, handleheight=0.9, labelspacing=0.35)
    return fig


def save(fig, stem: str) -> None:
    out = S.OUT_DIR / f"{stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"wrote {out}")


def make(key: str, model: str, lo: str, hi: str, stem: str) -> None:
    plt.rcParams.update({
        "font.family": style.use_font(), "font.size": BASE, "text.color": S.INK,
        "axes.labelcolor": S.INK, "axes.edgecolor": S.INK, "xtick.color": S.INK,
        "ytick.color": S.INK, "figure.facecolor": S.PAPER, "axes.facecolor": S.PAPER,
        "savefig.facecolor": S.PAPER, "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.9,
    })
    runs = {r.effort: r for r in DATASETS[key].loader()[0] if r.model == model}
    col = style.style_of(model).color
    fixed = [Bar(e, col, runs[e].resolved_units, runs[e].n_units,
                 mean_parts(unit_parts(model, key, runs[e]))) for e in (lo, hi)]
    for b, e in zip(fixed, (lo, hi)):
        assert abs(wilson(len(b.solved), b.n_units)[0] - runs[e].accuracy[0]) < 0.05
    ladder = ladder_bar(key, model, runs)
    bars = fixed + [ladder]

    save(draw_combined(bars, ref=fixed[1]), stem)

    a, b = fixed[0].solved, fixed[1].solved
    only_a, only_b = len(a - b), len(b - a)
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    print(f"\n{style.label_of(model)} on {TITLES[key]}: {lo} vs {hi} vs naive laddering, "
          f"wage ${latency.WAGE.usd_per_hour:.2f}/h")
    for bar in bars:
        pct, l_, h_ = wilson(len(bar.solved), bar.n_units)
        print(f"  {bar.label:7} solved {len(bar.solved)}/{bar.n_units} = {pct:.1f}% "
              f"[{l_:.1f}, {h_:.1f}]   " + "  ".join(f"{k} {style.fmt_money(v)}"
                                                  for k, v in bar.parts.items()))
    print(f"  {lo} vs {hi}: solved at both {len(a & b)}, only at {lo} {only_a}, only at {hi} "
          f"{only_b}; McNemar exact p = {p:.2f}")
    for name, keys in (("API bill", API), ("API + waiting", PARTS)):
        ref = fixed[1].total(keys)
        print(f"  {name:14} " + "   ".join(f"{bar.label} {style.fmt_money(bar.total(keys))} "
                                         f"(×{bar.total(keys) / ref:.2f})" for bar in bars))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="terminal_bench", choices=sorted(TITLES))
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--lo", default="high")
    ap.add_argument("--hi", default="xhigh")
    ap.add_argument("--stem", default="effort_pair")
    a = ap.parse_args()
    make(a.bench, a.model, a.lo, a.hi, a.stem)
