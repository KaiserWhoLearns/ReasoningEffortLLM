#!/usr/bin/env python3
"""Figure: one SWE-Bench Pro task at two efforts, same verdict, and at xhigh the agent reads the
gold fix.

    python figures/create_swebenchpro_example.py
    python figures/create_swebenchpro_example.py --dump xhigh   # print the per-step trajectory

A side-by-side step timeline of GPT-5.6-sol on
`instance_internetarchive__openlibrary-00bec1e7c8f3...` at `none` (left) and `xhigh` (right).
Both runs resolve. The `none` run solves the task from the code and the issue text in 15 calls.
The `xhigh` run does the same work, then lists all branches (`git log --all`), finds the fix
commit still present in the container's clone, reads it, overwrites its own validator with
`git show <gold>:<file>`, runs the tests from that commit, and confirms a byte-for-byte match
before submitting. This is the gold-commit leakage that `behavior/leakage/oracle_hunt.py` counts
and that the corrected environment removes.

The phases below were read off the two trajectories (`--dump` prints each step's thought and
command); the header numbers come from `output/leakage/pro_features.jsonl`
(`behavior/leakage/extract_traj_features.py`). Bar length is the number of steps in the phase;
colour is the phase's kind (both panels show the same model):

    grey    routine work (reading, editing, submitting)
    blue    a genuine test run (pytest on the repository's own tests)
    orange  looks into git history short of the answer / validates against the copied gold
    red     reads or copies the gold fix commit itself

Writes output/figures/app_swebenchpro_example.pdf.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import style  # noqa: E402
from common import paths  # noqa: E402
from style import GRID, INK, MUTED  # noqa: E402

FEATURES = paths.output_dir("leakage") / "pro_features.jsonl"
PDF_OUT = paths.output_dir("figures") / "app_swebenchpro_example.pdf"

MODEL = "gpt-5.6-sol"
INSTANCE_SUB = "openlibrary-00bec1e7"

MONO = "Menlo"          # filenames / commands inside a label; falls back if missing
C_ROUTINE, C_TEST, C_HIST, C_GOLD = "#c4c7ce", "#2f5fd0", "#c77414", "#c0392b"
# The card behind the whole figure, a shade darker than style.SURFACE; the canvas itself is saved
# transparent.
CARD = "#f6f6f5"
# pytest invocations per run, counted off the trajectories (none: step 12; xhigh: 12, 20, 21).
N_TEST_RUNS = {"none": 1, "xhigh": 3}


@dataclass(frozen=True)
class Phase:
    steps: tuple[int, int]          # inclusive 1-based call range
    label: str                      # `code` spans use backticks; "\n" breaks the line
    kind: str = "routine"           # routine | test | hist | gold
    tag: str = ""                   # small boxed annotation right of the label


KIND_COLOR = {"routine": C_ROUTINE, "test": C_TEST, "hist": C_HIST, "gold": C_GOLD}
KIND_TEXT = {"routine": INK, "test": C_TEST, "hist": C_HIST, "gold": C_GOLD}

# Phase boundaries as read off the trajectories (`--dump`).
NONE_PHASES = [
    Phase((1, 4), "read `import_validator.py`, `code.py`, the tests"),
    Phase((5, 6), "write repro script, fix `PYTHONPATH`, confirm the failure"),
    Phase((7, 9), "rewrite the validator (two models + fallback);\nad-hoc edge-case checks"),
    Phase((10, 11), "`ruff`, tidy imports and a stale comment"),
    Phase((12, 12), "`pytest` importapi suite — passes", "test"),
    Phase((13, 15), "diff, submit"),
]
XHIGH_PHASES = [
    Phase((1, 4), "read `import_validator.py`, `code.py`, builder, the tests"),
    Phase((5, 6), "write repro script, fix `PYTHONPATH`, confirm the failure"),
    Phase((7, 8), "`git show` of the commit that introduced the validator", "hist"),
    Phase((9, 11), "rewrite the validator (two models + fallback);\nrepro passes"),
    Phase((12, 14), "`pytest` + `ruff` — genuine pass, then an edge-case matrix", "test"),
    Phase((15, 17), "git archaeology → `git log --all` finds the\ngold fix commit; reads its diff", "gold",
          tag="gold read"),
    Phase((18, 18), "overwrites its validator with `git show <gold>:<file>`", "gold",
          tag="gold copy"),
    Phase((19, 21), "reruns repro; runs the gold commit's own tests", "hist"),
    Phase((22, 25), "`ruff`/`black` format checks; `cmp` — byte-for-byte\nmatch with the gold file", "hist"),
    Phase((26, 27), "diff, submit"),
]


def load_header(effort: str) -> dict:
    rows = [json.loads(l) for l in open(FEATURES)]
    hits = [r for r in rows if r["model"] == MODEL and r["effort"] == effort
            and INSTANCE_SUB in r["instance_id"]]
    assert len(hits) == 1, (effort, len(hits))
    return hits[0]


def dump(effort: str) -> None:
    r = load_header(effort)
    j = json.load(open(paths.DATA_ROOT / r["traj_path"]))
    i = 0
    for m in j["messages"]:
        if m.get("role") != "assistant":
            continue
        i += 1
        thought = (m.get("content") or "").split("```")[0].strip().replace("\n", " ")[:150]
        print(f"[{i}] {thought}")
        for a in ((m.get("extra") or {}).get("actions") or []):
            print("     $", a.get("command", "").replace("\n", "⏎")[:200])


# ── rich text: a label is plain sans with backticked spans in monospace ───────────────────────
def rich_text(fig, ax, x, y, label: str, color: str, size: float, weight="normal", va="center"):
    """Place `label` starting at data (x, y); returns the x where the text ended (data units)."""
    renderer = fig.canvas.get_renderer()
    lines = label.split("\n")
    dy_pt = size * 1.45                                     # line pitch in points
    # convert a point offset to data units on this axes
    y0_px = ax.transData.transform((0, y))[1]
    line_px = dy_pt * fig.dpi / 72
    n = len(lines)
    x_end = x
    for li, line in enumerate(lines):
        y_px = y0_px + (n - 1) / 2 * line_px - li * line_px
        y_line = ax.transData.inverted().transform((0, y_px))[1]
        cx = x
        parts = line.split("`")
        for pi, part in enumerate(parts):
            if not part:
                continue
            mono = pi % 2 == 1
            t = ax.text(cx, y_line, part, ha="left", va="center", color=color, fontsize=size * (0.93 if mono else 1),
                        family=MONO if mono else None, weight=weight, zorder=5)
            bb = t.get_window_extent(renderer=renderer)
            cx = ax.transData.inverted().transform((bb.x1, 0))[0]
        x_end = max(x_end, cx)
    return x_end


def row_pitch(p: Phase) -> float:
    return 1.0 if "\n" not in p.label else 1.45


def panel_depth(phases: list[Phase]) -> float:
    return 0.5 + sum(row_pitch(p) for p in phases)


def draw_panel(fig, ax, x0, width, effort: str, phases: list[Phase], depth: float, hdr: dict):
    """One panel in data units: x0..x0+width, rows from y=0 (top) downward, one row per phase."""
    ROW = 1.0
    bar_x1 = x0 + width
    bar_max = 6.4                        # data units of the longest bar
    step_scale = bar_max / 9             # 9 steps == longest phase across both panels
    # header
    y = 0.0
    ax.text(x0, y, effort, ha="left", va="center", fontsize=12.5, weight="bold", family=MONO, color=INK)
    n_tests = N_TEST_RUNS[effort]
    ax.text(bar_x1, y, f"{hdr['api_calls']} calls · ${hdr['cost']:.3f} · "
                       f"{'resolved' if hdr['resolved'] else 'not resolved'}",
            ha="right", va="center", fontsize=10.5, family=MONO, color=MUTED)
    ax.plot([x0, bar_x1], [y - 0.5 * ROW, y - 0.5 * ROW], color=INK, lw=1.0, zorder=3)
    # label column: a fixed gap after the widest step range ("10–11"), measured on this canvas
    probe = ax.text(x0, 0, "10–11", fontsize=8.5, family=MONO, alpha=0)
    bb = probe.get_window_extent(renderer=fig.canvas.get_renderer())
    label_x = ax.transData.inverted().transform((bb.x1, 0))[0] - x0 + 0.45
    probe.remove()
    # rows: a two-line label gets a taller row so the text never touches the rules
    y_top = y - 0.5 * ROW
    for i, p in enumerate(phases):
        pitch = row_pitch(p)
        yc = y_top - pitch / 2
        y_top -= pitch
        a, b = p.steps
        ax.text(x0, yc, f"{a}–{b}" if a != b else f"{a}", ha="left", va="center",
                fontsize=8.5, family=MONO, color=MUTED)
        col = KIND_TEXT[p.kind]
        weight = "bold" if p.kind == "gold" else "normal"
        rich_text(fig, ax, x0 + label_x, yc, p.label, col, 9.6, weight=weight)
        # bar
        n_steps = b - a + 1
        L = max(n_steps * step_scale, 0.28)
        h = 0.20
        if n_steps == 1:
            ax.add_patch(plt.Circle((bar_x1 - 0.14, yc), 0.13, color=KIND_COLOR[p.kind], zorder=4))
        else:
            ax.add_patch(FancyBboxPatch((bar_x1 - L, yc - h / 2), L, h,
                                        boxstyle="round,pad=0,rounding_size=0.1",
                                        fc=KIND_COLOR[p.kind], ec="none", zorder=4))
        if p.tag:
            tx = bar_x1 - L - 0.35
            ax.text(tx, yc, p.tag, ha="right", va="center", fontsize=8, family=MONO,
                    color=KIND_COLOR[p.kind], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc=CARD,
                              ec=KIND_COLOR[p.kind], lw=0.8))
        ax.plot([x0, bar_x1], [y_top, y_top], color=GRID, lw=0.8, zorder=2)
    # footer at the common bottom (both panels share `depth`, so the cards are the same height)
    yf = -depth - 0.4
    ax.text(bar_x1, yf, f"{hdr['api_calls']} steps · {n_tests} test run{'s' if n_tests != 1 else ''}",
            ha="right", va="center", fontsize=9, family=MONO, color=MUTED)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", choices=["none", "xhigh"], help="print the per-step trajectory and exit")
    args = ap.parse_args()
    if args.dump:
        dump(args.dump)
        return

    style.apply_rc()
    h_none, h_xh = load_header("none"), load_header("xhigh")
    assert h_none["resolved"] and h_xh["resolved"]
    assert h_none["api_calls"] == NONE_PHASES[-1].steps[1] and h_xh["api_calls"] == XHIGH_PHASES[-1].steps[1]
    print(f"cost ratio xhigh/none = {h_xh['cost'] / h_none['cost']:.2f}")

    depth = max(panel_depth(NONE_PHASES), panel_depth(XHIGH_PHASES))
    PW, GAP = 17.0, 1.2
    fig = plt.figure(figsize=(11.0, 0.36 * (depth + 2)), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.6, 2 * PW + GAP + 0.6)
    ax.set_ylim(-(depth + 1.1), 0.6)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.add_patch(FancyBboxPatch((xl[0], yl[0]), xl[1] - xl[0], yl[1] - yl[0],
                                boxstyle="round,pad=0,rounding_size=0.6", fc=CARD, ec="none",
                                zorder=0, clip_on=False))
    fig.canvas.draw()

    # No title block: the instance, model and bar-length encoding are stated in the LaTeX caption.
    draw_panel(fig, ax, 0, PW, "none", NONE_PHASES, depth, h_none)
    draw_panel(fig, ax, PW + GAP, PW, "xhigh", XHIGH_PHASES, depth, h_xh)
    # divider between panels, full panel height so both read as equal-height cards
    xm = PW + GAP / 2
    ax.plot([xm, xm], [0.5, -depth - 0.8], color=GRID, lw=1.0)

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT, bbox_inches="tight", pad_inches=0.02, transparent=True)
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
