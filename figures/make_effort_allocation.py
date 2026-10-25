#!/usr/bin/env python3
"""Figures: self-routing and naive laddering against fixed effort levels on Terminal-Bench.

    python figures/make_effort_allocation.py
    python figures/make_effort_allocation.py --only self-routing        # API-cost figures only
    python figures/make_effort_allocation.py --only self-routing-full --wage developer

Reads the self-routing tables (`output/self_routing/self_routing_{summary,levels,choices}.csv`)
and the laddering tables (`output/laddering/laddering_{summary,levels,choices}.csv`, jump mode,
`low` arm); never calls an API. Each panel is one model: every fixed effort level, the
self-router, a random router, naive laddering and the oracle, with cost relative to the model's
most accurate fixed level.

    self_routing.pdf           GPT-5.6-sol, GPT-5.6-luna, Claude Sonnet 5, DeepSeek V4 Flash
    app_self_routing.pdf       GLM-5.2, Inkling, Kimi K3
    app_self_routing_full.pdf  all seven, re-priced as API bill plus waiting cost

The full-cost figure rebuilds each model's Terminal-Bench pool from the run trees
(`routing/vanilla.py`), checks that its API costs reproduce the saved tables, then adds each
instance's waiting cost (`common/latency.py`) at every level; the oracle is re-solved on full
cost and router calls keep their API cost only. It also prints laddering's saving over fixed top
effort per model and its median across models; `--wage` picks the hourly wage that prices the
waiting (`common.latency.WAGES`), and a non-default wage adds its name to the figure's file name.

Writes to output/figures/.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import style  # noqa: E402
from common import latency, paths  # noqa: E402

SELF_DIR = paths.output_dir("self_routing")
LADDER_DIR = paths.output_dir("laddering")
OUT_DIR = paths.output_dir("figures")

INK, MUTED, GRID = "#1F232E", "#7C8798", "#E6E9EE"
ORACLE = "#D62728"
LADDER = "#17836B"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def routing_data():
    summary = [r for r in read_csv(SELF_DIR / "self_routing_summary.csv")
               if r["dataset"] == "terminal_bench"]
    levels = [r for r in read_csv(SELF_DIR / "self_routing_levels.csv")
              if r["dataset"] == "terminal_bench"]
    ladders = {r["model"]: r for r in read_csv(LADDER_DIR / "laddering_summary.csv")
               if r["dataset"] == "terminal_bench" and r["arm"] == "low"
               and r["mode"] == "jump"}
    ladder_levels = {(r["model"], r["effort"]): r
                     for r in read_csv(LADDER_DIR / "laddering_levels.csv")
                     if r["dataset"] == "terminal_bench" and r["mode"] == "jump"}
    summary.sort(key=lambda r: style.model_sort_key(r["model"]))
    assert summary and len({r["model"] for r in summary}) == len(summary)
    for r in summary:
        assert r["variant"] == "plain"
        assert 0 < int(r["n_units"]) <= int(r["n_routable"])
        assert float(r["oracle_resolve_pct"]) >= float(r["best_resolve_pct"])
        fixed = [x for x in levels if x["model"] == r["model"]
                 and not x["effort"].startswith("*")]
        best = next(x for x in fixed if x["effort"] == r["best_effort"])
        assert abs(float(best["cost"]) - float(r["best_cost"])) < .001
        assert all(x["n_units"] == r["n_units"] for x in fixed)
        ladder = ladders[r["model"]]
        assert ladder["basis"] == "instance" and ladder["n_units"] == r["n_units"]
        for x in fixed:
            matched = ladder_levels[(r["model"], x["effort"])]
            assert matched["n_units"] == x["n_units"]
            assert abs(float(matched["cost"]) - float(x["cost"])) < .001
            assert float(matched["resolved"]) == float(x["resolved"])
        assert abs(float(ladder["self_low_cost"]) - float(r["self_total_cost"])) < .001
        assert abs(float(ladder["self_low_resolve_pct"]) - float(r["self_resolve_pct"])) < .11
        r["ladder_cost"] = ladder["ladder_cost"]
        r["ladder_resolve_pct"] = ladder["ladder_resolve_pct"]
    return summary, levels


def full_cost_data(rows, levels):
    """Re-price every policy with the waiting term of common/latency.py added to the API bill.

    Rebuilds each model's Terminal-Bench pool from the run trees, checks that its API costs
    reproduce the saved routing tables, then adds each instance's waiting cost at every level.
    The oracle is re-solved on full cost; router calls keep their API cost only.
    Returns rows and levels shaped like routing_data()'s, plus laddering for every model.
    """
    sys.path.insert(0, str(REPO / "routing"))
    import selection
    import vanilla
    from datasets import DATASETS
    pools, _, _ = vanilla.pools_for(DATASETS["terminal_bench"])
    choices = [c for c in read_csv(SELF_DIR / "self_routing_choices.csv")
               if c["dataset"] == "terminal_bench" and c["variant"] == "plain"]
    ladder_choices = [c for c in read_csv(LADDER_DIR / "laddering_choices.csv")
                      if c["dataset"] == "terminal_bench" and c["mode"] == "jump"
                      and c["arm"] == "low"]
    ladder_summary = {r["model"]: r for r in read_csv(LADDER_DIR / "laddering_summary.csv")
                      if r["dataset"] == "terminal_bench" and r["arm"] == "low"
                      and r["mode"] == "jump"}

    def close(a, b, tol=.01):
        return abs(a - b) <= tol

    def priced(model):
        pool = pools[model]
        wait = {r.effort: r.values["wait"] for r in pool.runs}
        full = {e: {u: pool.costs[e][u] + wait[e][u] for u in pool.units} for e in pool.pool}
        return pool, full

    ladders = {}
    for model, summary in ladder_summary.items():
        pool, full = priced(model)
        picks = [c for c in ladder_choices if c["model"] == model]
        assert len(picks) == int(summary["n_units"])
        api = total = 0.0
        for c in picks:
            u, start, top = c["unit"], c["start_effort"], summary["top_effort"]
            assert u in pool.units and start == summary["start_effort"]
            escalated = c["escalated"] == "True"
            assert c["ended_at"] == (top if escalated else start)
            api += pool.costs[start][u] + (pool.costs[top][u] if escalated else 0)
            total += full[start][u] + (full[top][u] if escalated else 0)
        assert close(api, float(summary["ladder_cost"])), (model, api)
        units = {c["unit"] for c in picks}
        top_api = sum(pool.costs[summary["top_effort"]][u] for u in units)
        assert close(top_api, float(summary["top_cost"])), (model, top_api)
        ladders[model] = {
            "ladder_api": api, "ladder_full": total, "top_api": top_api,
            "top_full": sum(full[summary["top_effort"]][u] for u in units),
            "ladder_resolve_pct": float(summary["ladder_resolve_pct"]),
            "top_resolve_pct": float(summary["top_resolve_pct"])}

    full_rows, full_levels = [], []
    for r in rows:
        model = r["model"]
        pool, full = priced(model)
        picks = [c for c in choices if c["model"] == model]
        units = {c["unit"] for c in picks}
        assert len(units) == len(picks) == int(r["n_units"]) and units <= pool.units
        menu, to_level = selection.build_menu(pool.pool)
        for x in levels:
            if x["model"] == model and not x["effort"].startswith("*") and x["effort"] in full:
                assert close(sum(pool.costs[x["effort"]][u] for u in units), float(x["cost"]))
        self_api = sum(pool.costs[to_level[c["self_effort"]]][u] + float(c["router_cost"])
                       for c in picks for u in [c["unit"]])
        assert close(self_api, float(r["self_total_cost"]), .02), (model, self_api)
        oracle_api, oracle_solved, _ = pool.oracle_over(units)
        assert close(oracle_api, float(r["oracle_cost"]))
        assert close(vanilla.random_over(pool, units).cost, float(r["random_cost"]))

        fixed = {e: sum(full[e][u] for u in units) for e in pool.pool}
        picks_full = [to_level[name] for name in menu]
        oracle_full = 0.0
        for u in units:
            solving = [e for e in pool.pool if u in pool.resolved[e]]
            oracle_full += min(full[e][u] for e in (solving or pool.pool))
        out = dict(r)
        out.update({
            "best_cost": fixed[r["best_effort"]],
            "self_total_cost": sum(full[to_level[c["self_effort"]]][c["unit"]]
                                   + float(c["router_cost"]) for c in picks),
            "oracle_cost": oracle_full,
            "random_cost": sum(sum(full[e][u] for e in picks_full) / len(picks_full)
                               for u in units),
            "ladder_cost": ladders[model]["ladder_full"]})
        assert units == {c["unit"] for c in ladder_choices if c["model"] == model}
        full_rows.append(out)
        for x in levels:
            if x["model"] == model and not x["effort"].startswith("*") and x["effort"] in full:
                full_levels.append({**x, "cost": fixed[x["effort"]]})
    return full_rows, full_levels, ladders


def save(fig, name, **kwargs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / f"{name}.pdf"
    fig.savefig(target, **kwargs)
    plt.close(fig)
    print(f"wrote {target}")


MAIN_SELF_MODELS = ("gpt-5.6-sol", "gpt-5.6-luna", "claude-sonnet-5", "deepseek-v4-flash")
APPENDIX_SELF_MODELS = ("glm-5.2", "inkling", "kimi-k3")


def draw_self(rows, levels, name, xlim,
              xlabel="Cost relative to the most accurate fixed effort (1.0)", nrows=1):
    # One row of panels with a shared cost axis; the benchmark is named in the LaTeX caption.
    # Width scales with the panel count so each panel keeps its three-panel size. `nrows` > 1
    # wraps the panels into a grid that keeps the one-row panel size, with the axis label and
    # legend under the last row.
    ncols = -(-len(rows) // nrows)
    fig_size = (2.4 * max(ncols, 3), 2.9 + 2.3 * (nrows - 1))
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, squeeze=False)
    # the one-row margins in inches, so the grid's legend band is as tall as the row's
    foot = .27 * 2.9 / fig_size[1]
    fig.subplots_adjust(left=.09 * 3 / max(ncols, 3), right=.99, top=1 - .09 * 2.9 / fig_size[1],
                        bottom=foot, wspace=.3, hspace=.35)
    for i, r in enumerate(rows):
        ax = axes.flat[i]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID)
        ax.tick_params(length=0, pad=4, labelsize=10)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, lw=.6)
        model = r["model"]
        color = style.style_of(model).color
        base = float(r["best_cost"])
        fixed = [x for x in levels if x["model"] == model
                 and not x["effort"].startswith("*")]
        fixed.sort(key=lambda x: float(x["cost"]))
        xs = [float(x["cost"]) / base for x in fixed]
        ys = [float(x["resolve_pct"]) for x in fixed]
        ax.scatter(xs, ys, s=24, facecolors="white", edgecolors=color, linewidths=.9, zorder=3)
        best = next(x for x in fixed if x["effort"] == r["best_effort"])
        # The best fixed point gets a small label; all fixed settings are plotted.
        ax.annotate(best["effort"], (1, float(best["resolve_pct"])),
                    xytext=(-4, 6), textcoords="offset points",
                    fontsize=9,
                    ha="right", color=MUTED)
        # The oracle is red in every panel so the upper bound is easy to find.
        for prefix, marker, size, fill, edge in [("oracle", "*", 150, ORACLE, ORACLE),
                                                ("random", "D", 36, "white", color),
                                                ("self", "s", 44, color, INK),
                                                ("ladder", "^", 65, LADDER, INK)]:
            cost_key = "self_total_cost" if prefix == "self" else f"{prefix}_cost"
            x, y = float(r[cost_key]) / base, float(r[f"{prefix}_resolve_pct"])
            ax.scatter(x, y, marker=marker, s=size, facecolors=fill, edgecolors=edge,
                       linewidths=.75, zorder=6 if prefix == "oracle" else 5)
            xs.append(x)
            ys.append(y)
        assert all(xlim[0] <= x <= xlim[1] for x in xs), (model, min(xs), max(xs))
        ax.axvline(1, color=INK, lw=1.0, ls=(0, (3, 2.5)), alpha=.75, zorder=1)
        ax.set_ylim(max(0, min(ys) - 6), min(100, max(ys) + 7))
        ax.xaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        if i % ncols == 0:
            ax.set_ylabel("Tasks solved (%)", fontsize=11)
        ax.set_title(style.label_of(model), color=color, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlim(*xlim)
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    fig.text(.54, .145 * 2.9 / fig_size[1], xlabel, ha="center", fontsize=11)
    fig.legend(handles=[
        Line2D([], [], marker="o", mfc="white", mec=MUTED, linestyle="none", label="Fixed effort"),
        Line2D([], [], marker="s", color=MUTED, linestyle="none", label="Self-router"),
        Line2D([], [], marker="D", mfc="white", mec=MUTED, linestyle="none", label="Random router"),
        Line2D([], [], marker="^", mfc=LADDER, mec=INK, markersize=8, linestyle="none", label="Naïve laddering"),
        Line2D([], [], marker="*", color=ORACLE, markersize=11, linestyle="none", label="Oracle")],
        loc="lower center", bbox_to_anchor=(.54, .03 * 2.9 / fig_size[1]), ncol=5, frameon=False,
        columnspacing=1.0, handletextpad=.35, fontsize=9.5)
    save(fig, name, bbox_inches="tight", pad_inches=.03)


def draw_self_figures(rows, levels):
    by_model = {r["model"]: r for r in rows}
    assert set(by_model) == set(MAIN_SELF_MODELS) | set(APPENDIX_SELF_MODELS)
    # Shared axes within each row include all policies without clipping.
    draw_self([by_model[m] for m in MAIN_SELF_MODELS], levels, "self_routing", (0, 2.05))
    # Kimi's max effort costs 2.09 times its most accurate fixed effort (high).
    draw_self([by_model[m] for m in APPENDIX_SELF_MODELS], levels, "app_self_routing", (0, 2.2))


def draw_self_full(rows, levels):
    full_rows, full_levels, ladders = full_cost_data(rows, levels)
    by_model = {r["model"]: r for r in full_rows}
    for model, l in ladders.items():
        print(f"{model:18s} ladder vs top: API {100 * (1 - l['ladder_api'] / l['top_api']):+6.1f}% "
              f"saved, full {100 * (1 - l['ladder_full'] / l['top_full']):+6.1f}% saved; "
              f"success {l['top_resolve_pct']:.1f} -> {l['ladder_resolve_pct']:.1f}")
    saving = [100 * (1 - l["ladder_full"] / l["top_full"]) for l in ladders.values()]
    print(f"median full-cost saving of laddering over fixed top effort, {len(saving)} models, "
          f"wage ${latency.WAGE.usd_per_hour:.2f}/h: {statistics.median(saving):.1f}%")
    stem = "app_self_routing_full" + ("" if latency.WAGE.key == latency.DEFAULT_WAGE
                                      else f"_{latency.WAGE.key}")
    # Every model with a self-router comparison, two rows of four.
    draw_self([by_model[m] for m in MAIN_SELF_MODELS + APPENDIX_SELF_MODELS], full_levels,
              stem, (0, 2.05),
              xlabel="Full cost relative to the most accurate fixed effort (1.0)", nrows=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["self-routing", "self-routing-full"],
                        help="build only the API-cost figures, or only the full-cost figure "
                             "(default: all three)")
    parser.add_argument("--wage", choices=sorted(latency.WAGES), default=latency.DEFAULT_WAGE,
                        help="hourly wage that prices the waiting time in the full-cost figure")
    args = parser.parse_args()
    latency.set_wage(args.wage)
    plt.rcParams.update({"font.family": style.use_font(), "font.size": 7,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    rows, levels = routing_data()
    if args.only in (None, "self-routing"):
        draw_self_figures(rows, levels)
    if args.only in (None, "self-routing-full"):
        draw_self_full(rows, levels)


if __name__ == "__main__":
    main()
