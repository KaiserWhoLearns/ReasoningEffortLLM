#!/usr/bin/env python3
"""Figures: GPT-5.6-sol's reference look-ups and solve rate on original vs. corrected SWE-Bench Pro.

    python figures/create_swebenchpro_reference_panels.py

Both conditions use the 44 pilot instances. Look-up behaviour is classified directly from the
stored trajectories with the leakage classifiers (`behavior/leakage/oracle_hunt.py` for reading
the gold fix commit from git history on the original environment, `net_fetch_intent.py` for
answer-seeking network fetches on the corrected one). Solve rates come from `datasets.load_pro`,
including its per-instance budget rule.

Writes to output/figures/:
    swebenchpro_looking_count.pdf    instances with a gold-commit or online look-up, per effort
    swebenchpro_corrected_perf.pdf   solve rate per effort, original vs. corrected
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "behavior" / "leakage"))

import net_fetch_intent as intent  # noqa: E402
import style  # noqa: E402
from common import paths  # noqa: E402
from datasets import _pilot_ids, load_pro  # noqa: E402
from oracle_hunt import classify, gold_sha  # noqa: E402

MODEL = "gpt-5.6-sol"
EFFORTS = ["low", "medium", "high", "xhigh"]

# These two panels are set in Inter (`fonts/`), falling back to the shared font if it is absent.
FONT_DIR = HERE / "fonts"
INK, MUTED, GRID, PAPER = "#1F232E", "#7C8798", "#E6E9EE", "#FFFFFF"


def measurements() -> dict:
    pilot = _pilot_ids()
    result = {}
    for condition, benchmark in (("official", "swebench-pro"),
                                 ("corrected", "swebench-pro-corrected")):
        runs = {r.effort: r for r in load_pro(benchmark, subset=pilot)[0] if r.model == MODEL}
        result[condition] = []
        for effort in EFFORTS:
            root = paths.run_dir(benchmark, MODEL, effort)
            verdicts = json.loads((root.parent / "eval" / effort / "eval_results.json").read_text())
            assert pilot <= verdicts.keys(), (condition, effort, "Incomplete evaluation")
            gold = fetch = 0
            for iid in sorted(pilot):
                trajectories = list((root / iid).glob("*.traj.json"))
                assert len(trajectories) == 1, (condition, effort, iid, trajectories)
                data = json.loads(trajectories[0].read_text())
                commands = [a["command"] for m in data["messages"]
                            if m.get("role") == "assistant"
                            for a in ((m.get("extra") or {}).get("actions") or [])
                            if a.get("command")]
                gold += classify(commands, gold_sha(iid))[2] > 0
                org, repo = intent.org_repo(iid)
                patched = intent.patch_files((data.get("info") or {}).get("submission", ""))
                fetch += any(intent.classify(c, org, repo, patched)["kind"] in intent.ANSWER_SEEKING
                             for c in commands if intent.NET.search(c))
            run = runs[effort]
            assert run.n_units == len(pilot), (condition, effort, run.n_units)
            result[condition].append({
                "effort": effort,
                "n": len(pilot),
                "gold_read_pct": 100 * gold / len(pilot),
                "answer_seeking_pct": 100 * fetch / len(pilot),
                "resolved_pct": run.accuracy[0],
            })
    return result


def use_font() -> str:
    """Register Inter from `fonts/`; fall back to `style.use_font()` if it is not there."""
    files = sorted(FONT_DIR.glob("Inter-*.ttf"))
    for path in files:
        fm.fontManager.addfont(str(path))
    return "Inter" if files else style.use_font()


def apply_rc(base: float) -> None:
    plt.rcParams.update({
        "font.family": use_font(), "font.size": base,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
        "xtick.color": INK, "ytick.color": MUTED,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.titlesize": base + 3, "axes.titleweight": "semibold",
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.7,
    })


def panel(ylabel: str):
    # Fixed, equal canvases and axes positions align the two subfigures.
    fig, ax = plt.subplots(figsize=(2.9, 2.2))
    fig.subplots_adjust(left=0.215, right=0.98, bottom=0.12, top=0.86)
    ax.set_xticks(range(len(EFFORTS)), EFFORTS)
    ax.set_xlim(-0.6, len(EFFORTS) - 0.4)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=4)
    return fig, ax


def legend(ax):
    # Anchored to the canvas edge so the two-column legend fits the narrow panel.
    ax.figure.legend(*ax.get_legend_handles_labels(), loc="upper left",
                     bbox_to_anchor=(0.01, 0.995), borderaxespad=0, ncol=2, fontsize=7.5,
                     handlelength=1.4, handletextpad=0.4, columnspacing=0.8)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    data = measurements()
    apply_rc(base=9)
    hue = style.style_of(MODEL).color
    xs = list(range(len(EFFORTS)))
    out = paths.output_dir("figures", create=True)

    fig, ax = panel("Instances (%)")
    gold = [r["gold_read_pct"] for r in data["official"]]
    fetch = [r["answer_seeking_pct"] for r in data["corrected"]]
    ax.bar([x - 0.20 for x in xs], gold, width=0.38, facecolor="white", edgecolor=hue,
           linewidth=1.1, hatch="////", label=r"Gold $\mathtt{git}$ commit look up")
    ax.bar([x + 0.20 for x in xs], fetch, width=0.38, color=hue,
           edgecolor="white", linewidth=0.5, label="Online look up")
    for x, (g, f) in enumerate(zip(gold, fetch)):
        # Labels aligned away from the inter-bar gap so the shorter bar's text cannot extend
        # into its taller neighbour.
        for dx, value, align in ((-0.025, g, "right"), (0.025, f, "left")):
            ax.annotate(f"{value:.1f}%", (x + dx, value), xytext=(0, 3),
                        textcoords="offset points", ha=align, va="bottom", fontsize=6.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    legend(ax)
    fig.savefig(out / "swebenchpro_looking_count.pdf")
    plt.close(fig)

    fig, ax = panel("Score (Solve Rate)")
    for condition, label, linestyle, fill in (
        ("official", "Official", "--", "white"),
        ("corrected", "Corrected", "-", hue),
    ):
        ax.plot(xs, [r["resolved_pct"] for r in data[condition]], color=hue,
                linestyle=linestyle, linewidth=1.6, marker="o", markersize=5,
                markerfacecolor=fill, markeredgecolor=hue, markeredgewidth=1.1, label=label)
    ax.set_ylim(52, 82)
    ax.set_yticks([55, 60, 65, 70, 75, 80])
    legend(ax)
    fig.savefig(out / "swebenchpro_corrected_perf.pdf")
    plt.close(fig)
    print(json.dumps(data, indent=2))
    print(f"wrote {out / 'swebenchpro_looking_count.pdf'}\nwrote {out / 'swebenchpro_corrected_perf.pdf'}")


if __name__ == "__main__":
    main()
