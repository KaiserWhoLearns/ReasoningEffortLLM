#!/usr/bin/env python3
"""Figure: what a unit the model ALWAYS solves costs at each effort level, `none` to the top.

    python figures/make_solvable_cost.py               # solvable_cost.pdf (API dollars)
    python figures/make_solvable_cost.py --full        # solvable_cost_full.pdf (API + waiting)
    python figures/make_solvable_cost.py --both        # both
    python figures/make_solvable_cost.py --corrected   # solvable_cost_corrected.pdf + .csv
    python figures/make_solvable_cost.py --both --deepswe   # solvable_cost[_full]_deepswe_v1_1.pdf

Rows are benchmarks, groups are models, and the bars inside a group are the effort levels in
ladder order — `none`, `low`, `medium`, `high`, `xhigh`/`max` — each in its own slot, so a model
that skips a level leaves that slot empty.

The quantity is the mean cost of one unit the model solves at EVERY level it ran, `none`
included, divided by that cost at `low` (or the bottom reasoning level, named in the footer, for
a model with no `low`). Fixing the unit set makes this a price comparison rather than a
difficulty comparison. Bars are comparable within a group; the footer gives the dollar value of
1x.

Each bar is stacked by what the dollars bought:
    input      the prompt, read back on every call (uncached and cache-read tokens together):
               the API bill minus the output part
    reasoning  output tokens the provider bills but the caller never sees
    output     output tokens the caller does see
    waiting    (--full only) decode seconds at the model's table throughput priced at the
               active wage (`common/latency.py`)
Reasoning and visible output are both billed at the output rate, so the split between them is
a token split. Per-unit reasoning tokens come from the trajectories on AIME, GPQA and
SWE-Bench Pro. harbor's Terminal-Bench trajectories record no reasoning count, so each agent
step is joined to its call in the run's LiteLLM capture (`litellm_calls.jsonl`) on the
(prompt, completion) token pair (`tb_reasoning_per_task`). Colour is the model; the legend uses
a neutral grey. Unit counts and always-solved set sizes are printed to stdout.

`--corrected` draws corrected SWE-Bench Pro on the 44 pilot instances and writes the plotted
values to `solvable_cost_corrected.csv` beside the PDF.

`--deepswe` draws the DeepSWE v1.1 row alone. It is the public leaderboard's own rollouts
(https://deepswe.datacurve.ai/data/v1.1; 113 tasks, about 4 rollouts per model and effort level,
mini-swe-agent), read from the snapshots in `paths.DEEPSWE_ROOT` (`deepswe/snapshot_*.py`):
    levels     no `none` was run; the GPT-5.6 tiers and Claude Sonnet 5 are drawn low..xhigh
               (their `max` left out) and GLM-5.2 at high and max (max in the XH slot,
               anchored on high). Models without two drawn levels are left off.
    unit       a task, solved at a level if at least one of its rollouts passed; a task's cost
               at a level is the mean over its passing rollouts there.
    dollars    each rollout priced from its tokens at `common/pricing.py`'s rates, with cache
               reads at the cached rate and recorded cache writes. The price table has no
               long-context tier.
    reasoning  from the trajectories' `total_reasoning_tokens`; Claude Sonnet 5's trajectories
               record none, so its bars are input + output only.
The numbers behind the row are written to `deepswe_cost/` beside the PDF.

Writes to output/figures/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import style  # noqa: E402
from common import latency, models, paths, pricing  # noqa: E402
from datasets import rung_of  # noqa: E402
from ladders import BENCH_N, PRO_CORRECTED_KEY, load_runs  # noqa: E402

OUT_DIR = paths.output_dir("figures")

# Benchmark rows, top to bottom: the two single-call benchmarks, then the agent loops.
ROWS = [("aime_2026", "AIME 2026"), ("gpqa_diamond", "GPQA Diamond"),
        ("terminal_bench", "Terminal-Bench 2.1"), ("swebench_pro_pilot", "SWE-Bench Pro")]
PRO_CORRECTED_ROW = (PRO_CORRECTED_KEY, "SWE-Bench Pro\n(corrected)")

DEEPSWE_KEY = "deepswe_v1_1"
DEEPSWE_ROW = (DEEPSWE_KEY, "DeepSWE v1.1")
DEEPSWE_TRIALS = paths.DEEPSWE_ROOT / "deepswe_v1_1_trials.csv"
DEEPSWE_TOKENS = paths.DEEPSWE_ROOT / "deepswe_v1_1_rollout_tokens.csv"
DEEPSWE_CSV_DIR = OUT_DIR / "deepswe_cost"
# The leaderboard's model slugs -> the study's.
DEEPSWE_MODELS = {"gpt-5-6-luna": "gpt-5.6-luna", "gpt-5-6-terra": "gpt-5.6-terra",
                  "gpt-5-6-sol": "gpt-5.6-sol", "claude-sonnet-5": "claude-sonnet-5",
                  "glm-5-2": "glm-5.2", "inkling": "inkling",
                  "deepseek-v4-flash": "deepseek-v4-flash"}
DEEPSWE_LEVELS = ["low", "medium", "high", "xhigh", "max"]

SLOTS = ["none", "low", "medium", "high", "xhigh"]          # `max` lands in the xhigh slot
SLOT_TICK = {"none": "N", "low": "L", "medium": "M", "high": "H", "xhigh": "XH"}
ANCHOR = "low"

PAPER = "#FFFFFF"
INK = "#1F232E"
MUTED = "#7C8798"
GRID = "#E6E9EE"
LEGEND_GREY = "#8A93A2"   # a neutral stand-in in the legend: a model colour there names a model

BASE = 7.0                # pt; drawn near \textwidth
BAR_W = 0.82
GROUP_GAP = 1.4           # empty slots between two models' groups


# ------------------------------------------------------------------------------------------ data
@dataclass
class Level:
    effort: str
    slot: str
    parts: dict[str, float]      # mean $ per always-solved unit: input / reasoning / output / wait

    def total(self, keys: tuple[str, ...]) -> float:
        return sum(self.parts[k] for k in keys)


@dataclass
class Group:
    model: str
    n_always: int
    n_units: int
    levels: dict[str, Level]     # slot -> Level
    anchor: str                  # the slot 1× is taken against


def _fade(color: str, keep: float) -> tuple[float, float, float]:
    """`color` mixed toward the surface, keeping `keep` of it."""
    base, paper = np.array(mcolors.to_rgb(color)), np.array(mcolors.to_rgb(PAPER))
    return tuple(paper + keep * (base - paper))


_TB_REASONING: dict[tuple[str, str], dict[str, float]] = {}


def tb_reasoning_per_task(model: str, effort: str) -> dict[str, float]:
    """{task_name: reasoning tokens} for one Terminal-Bench run, joined from two files.

    harbor's per-step trajectory records prompt and completion tokens but no reasoning count;
    the LiteLLM capture beside it (`litellm_calls.jsonl`) records every call's full usage,
    reasoning tokens included, but names no trial. The two are joined on the (prompt,
    completion) token pair, which is unique per call in practice; a duplicate pair falls back to
    the call whose end time is nearest the step's timestamp.
    """
    key = (model, effort)
    if key in _TB_REASONING:
        return _TB_REASONING[key]
    run_dir = paths.data_root("terminal-bench-2-1") / models.litellm_slug(model) / effort
    calls: dict[tuple, list[tuple[float, dt.datetime | None]]] = defaultdict(list)
    seen: set[str] = set()
    cap = run_dir / "litellm_calls.jsonl"
    if cap.exists():
        for line in cap.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("response_id")
            if rid and rid in seen:
                continue
            if rid:
                seen.add(rid)
            u = r.get("usage") or {}
            rz = float((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            end = None
            try:
                end = dt.datetime.fromisoformat(r["end"]).replace(tzinfo=dt.timezone.utc)
            except (KeyError, TypeError, ValueError):
                pass
            calls[(u.get("prompt_tokens"), u.get("completion_tokens"))].append((rz, end))
    out: dict[str, float] = {}
    unmatched = 0
    for traj in run_dir.glob("*/*/agent/trajectory.json"):
        trial_dir = traj.parent.parent
        try:
            task = json.loads((trial_dir / "result.json").read_text())["task_name"]
            steps = json.loads(traj.read_text())["steps"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        total = 0.0
        for st_ in steps:
            if st_.get("source") != "agent":
                continue
            m = st_.get("metrics") or {}
            cands = calls.get((m.get("prompt_tokens"), m.get("completion_tokens")), [])
            if not cands:
                unmatched += 1
                continue
            if len(cands) > 1 and st_.get("timestamp"):
                ts = dt.datetime.fromisoformat(st_["timestamp"])
                cands = sorted(cands, key=lambda c: abs((c[1] - ts).total_seconds())
                               if c[1] else float("inf"))
            total += cands[0][0]
        out[task] = total
    if unmatched:
        print(f"  note: {style.label_of(model)} {effort} on terminal_bench: {unmatched} agent "
              f"steps found no capture call — their reasoning counts as 0")
    _TB_REASONING[key] = out
    return out


def split_unit(model: str, key: str, effort: str, run_values: dict, unit: str,
               run_share: float | None) -> dict[str, float] | None:
    """One unit's $ split into input / reasoning / output / wait, or None if it has no cost."""
    v = run_values
    if unit not in v.get("cost", {}) or unit not in v.get("output", {}):
        return None
    cost, out_tok = v["cost"][unit], v["output"][unit]
    out_rate = pricing.rate_triplet(model)[2]
    out_cost = min(out_tok * out_rate, cost)          # never let the residual go negative
    reas_tok = v.get("reasoning", {}).get(unit)
    if reas_tok is None:                              # no per-unit count: use the run's share
        reas_tok = out_tok * run_share if run_share is not None else 0.0
    reas_tok = min(max(reas_tok, 0.0), out_tok)
    reas_cost = out_cost * (reas_tok / out_tok if out_tok else 0.0)
    return {"input": cost - out_cost, "reasoning": reas_cost, "output": out_cost - reas_cost,
            "wait": v.get("wait", {}).get(unit, 0.0)}


def groups(key: str) -> list[Group]:
    """Every model's ladder on one benchmark, `none` included, over its always-solved units.

    Keeps runs over the benchmark's own unit count, and models with at least two levels.
    """
    if key == DEEPSWE_KEY:
        return deepswe_groups()
    runs, _ = load_runs(key)
    n = BENCH_N.get(key) or Counter(r.n_units for r in runs).most_common(1)[0][0]
    by_model = defaultdict(list)
    for r in runs:
        if r.n_units != n:
            continue
        by_model[r.model].append(r)
    out = []
    for model, rs in sorted(by_model.items(), key=lambda kv: style.model_sort_key(kv[0])):
        rs.sort(key=lambda r: style.effort_key(r.effort))
        if len(rs) < 2:
            continue
        always = set.intersection(*[r.resolved_units for r in rs])
        if not always:
            print(f"  note: {style.label_of(model)} on {key}: no unit solved at every level")
            continue
        levels = {}
        for r in rs:
            slot = rung_of(r.effort)
            if slot not in SLOTS:
                continue
            values = r.values
            if key == "terminal_bench":
                values = dict(values, reasoning=tb_reasoning_per_task(model, r.effort))
            share = None
            if any(u not in values.get("reasoning", {}) for u in always):
                if r.effort == "none":
                    share = 0.0
                else:   # a few units without a count where the run has them: its own share
                    have = [u for u in values.get("reasoning", {}) if u in values["output"]]
                    tot_o = sum(values["output"][u] for u in have)
                    share = (sum(values["reasoning"][u] for u in have) / tot_o) if tot_o else 0.0
            parts = [split_unit(model, key, r.effort, values, u, share) for u in sorted(always)]
            parts = [p for p in parts if p is not None]
            if not parts:
                continue
            levels[slot] = Level(effort=r.effort, slot=slot,
                                 parts={k: st.mean(p[k] for p in parts) for k in parts[0]})
        anchor = ANCHOR if ANCHOR in levels else next(s for s in SLOTS if s in levels and s != "none")
        out.append(Group(model=model, n_always=len(always), n_units=n, levels=levels,
                         anchor=anchor))
    return out


# ------------------------------------------------------------------------------- DeepSWE v1.1
def deepswe_plotted(levels: set[str]) -> list[str]:
    """The levels a model's DeepSWE group draws: low..xhigh where run, and `max` only for a model
    with no `xhigh` (GLM-5.2), where it takes the XH slot (`datasets.RUNG_ALIAS`)."""
    out = [e for e in SLOTS if e in levels]
    if "max" in levels and "xhigh" not in levels:
        out.append("max")
    return out


def deepswe_price(model: str, t: dict) -> dict[str, float]:
    """One rollout's $ at the repository's rates, split input / reasoning / output / wait."""
    prompt, comp = int(t["prompt_tokens"]), int(t["completion_tokens"])
    cost = pricing.cost_from_usage(model, prompt_tokens=prompt, output_tokens=comp,
                                   cache_read=int(t["cache_read_tokens"]),
                                   cache_creation=int(t["cache_creation_tokens"]))
    out_cost = min(comp * pricing.rate_triplet(model)[2], cost)
    reas = t["reasoning_tokens"]
    share = min(max(int(reas), 0), comp) / comp if reas != "" and comp else 0.0
    return {"input": cost - out_cost, "reasoning": out_cost * share,
            "output": out_cost * (1 - share),
            "wait": latency.wait_cost_usd(latency.decode_seconds(model, comp)) or 0.0}


_DEEPSWE: dict[str, list] = {}


def deepswe_groups() -> list[Group]:
    """Every model's DeepSWE ladder over its always-solved tasks, priced from the snapshots.

    Verdicts come from every rollout in the trials snapshot, so a rollout without token counts
    still counts toward "solved"; costs are averaged over the passing rollouts that have token
    counts. Also keeps the priced rollouts and per-task parts for `write_deepswe_csvs`.
    """
    verdicts: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for t in csv.DictReader(DEEPSWE_TRIALS.open()):
        if t["model"] in DEEPSWE_MODELS:
            verdicts[DEEPSWE_MODELS[t["model"]]][(t["reasoning_effort"], t["task_name"])].append(t)
    tokens = {t["trial_name"]: t for t in csv.DictReader(DEEPSWE_TOKENS.open())}
    n = len({task for cells in verdicts.values() for (_, task) in cells})
    out, rollouts, task_rows = [], [], []
    for model in sorted(verdicts, key=style.model_sort_key):
        cells = verdicts[model]
        efforts = deepswe_plotted({e for (e, _) in cells})
        if len(efforts) < 2:
            continue
        solved = {e: {task for (e2, task), ts in cells.items() if e2 == e
                      and any(t["passed"] == "True" for t in ts)} for e in efforts}
        always = set.intersection(*solved.values())
        for (e, task), ts in cells.items():
            for t in ts:
                tok = tokens.get(t["trial_name"])
                rollouts.append(dict(model=model, effort=e, task_name=task,
                                     trial_name=t["trial_name"], passed=t["passed"],
                                     drawn=e in efforts, always_solved=task in always,
                                     site_cost_usd=t["cost_usd"],
                                     **({k: tok[k] for k in ("prompt_tokens", "cache_read_tokens",
                                                             "cache_creation_tokens",
                                                             "completion_tokens",
                                                             "reasoning_tokens",
                                                             "peak_prompt_tokens")} if tok else {}),
                                     **(deepswe_price(model, tok) if tok else {})))
        levels = {}
        for e in efforts:
            per_task = []
            for task in sorted(always):
                priced = [deepswe_price(model, tokens[t["trial_name"]])
                          for t in cells[(e, task)]
                          if t["passed"] == "True" and t["trial_name"] in tokens]
                if not priced:
                    raise SystemExit(f"DeepSWE {model} {e} {task}: solved, but no passing rollout "
                                     f"has token counts to price")
                parts = {k: st.mean(p[k] for p in priced) for k in priced[0]}
                per_task.append(parts)
                task_rows.append(dict(model=model, effort=e, task_name=task,
                                      passing_rollouts_priced=len(priced),
                                      **{f"{k}_usd": v for k, v in parts.items()}))
            levels[rung_of(e)] = Level(effort=e, slot=rung_of(e),
                                       parts={k: st.mean(p[k] for p in per_task)
                                              for k in per_task[0]})
        anchor = ANCHOR if ANCHOR in levels else next(s for s in SLOTS if s in levels)
        out.append(Group(model=model, n_always=len(always), n_units=n, levels=levels,
                         anchor=anchor))
    _DEEPSWE.update(rollouts=rollouts, tasks=task_rows)
    return out


def write_deepswe_csvs(gs: list[Group]) -> list[Path]:
    """The numbers behind the DeepSWE row: every rollout priced, every always-solved task's
    parts, and each drawn level's bar (mean $ and × the anchor, API and with waiting)."""
    DEEPSWE_CSV_DIR.mkdir(parents=True, exist_ok=True)
    for old in DEEPSWE_CSV_DIR.glob("*.csv"):
        old.unlink()
    written = []
    for name, rows in (("rollouts", _DEEPSWE["rollouts"]), ("always_solved_tasks",
                                                             _DEEPSWE["tasks"])):
        rows = sorted(rows, key=lambda r: (style.model_sort_key(r["model"]),
                                           DEEPSWE_LEVELS.index(r["effort"]), r["task_name"],
                                           r.get("trial_name", "")))
        fields = list(dict.fromkeys(k for r in rows for k in r))
        path = DEEPSWE_CSV_DIR / f"{name}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        written.append(path)
    bars = []
    for g in gs:
        for slot in SLOTS:
            lv = g.levels.get(slot)
            if lv is None:
                continue
            api = lv.total(("input", "reasoning", "output"))
            full = lv.total(("input", "reasoning", "output", "wait"))
            base = g.levels[g.anchor]
            bars.append(dict(model=g.model, effort=lv.effort, slot=slot, anchor=g.anchor,
                             always_solved=g.n_always, tasks=g.n_units,
                             **{f"{k}_usd": v for k, v in lv.parts.items()},
                             api_usd=api, api_x_anchor=api / base.total(("input", "reasoning",
                                                                         "output")),
                             full_usd=full, full_x_anchor=full / base.total(
                                 ("input", "reasoning", "output", "wait"))))
    path = DEEPSWE_CSV_DIR / "bars.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bars[0]))
        w.writeheader()
        w.writerows(bars)
    return written + [path]


# --------------------------------------------------------------------------------------- drawing
def fmt_anchor(v: float) -> str:
    """'1.0¢' below ten cents, '$0.13' above — the footer under a group."""
    return f"{100 * v:.1f}¢" if v < 0.10 else f"${v:.2f}"


def segment_style(col: str, part: str) -> dict:
    """How one stacked part of a bar is painted, in the model's colour."""
    if part == "input":
        return dict(facecolor=_fade(col, 0.32), edgecolor="none")
    if part == "reasoning":
        # light tint of the model colour, densely hatched in the full colour: reads as
        # "same kind as Output" but is clearly separable from the solid Output block
        return dict(facecolor=_fade(col, 0.55), edgecolor=col, hatch="//////", linewidth=0)
    if part == "output":
        return dict(facecolor=col, edgecolor="none")
    if part == "wait":
        return dict(facecolor=PAPER, edgecolor=col, hatch="....", linewidth=0.6)
    raise KeyError(part)


def row_cap(gs: list[Group], parts: tuple[str, ...]) -> float:
    """Top of a row's axis: the tallest bar, unless one model towers over every other.

    When the tallest group's peak is more than `OVERFLOW_RATIO` times the next group's, the axis
    stops just above that next group and the overflowing bars are clipped and labelled.
    """
    peaks = sorted((max(lv.total(parts) / g.levels[g.anchor].total(parts)
                        for lv in g.levels.values()) for g in gs), reverse=True)
    if len(peaks) > 1 and peaks[0] > OVERFLOW_RATIO * peaks[1]:
        return peaks[1] * 1.15
    return peaks[0] * 1.06


OVERFLOW_RATIO = 2.0


def draw_row(ax, gs: list[Group], parts: tuple[str, ...], first_row: bool,
             last_row: bool, models: list[str]) -> None:
    """One benchmark's groups, each model at a FIXED position (`models`), so a model with no
    group on this row leaves its place empty — with a note — instead of shifting its neighbours."""
    cap = row_cap(gs, parts)
    x0 = 0.0
    by_model = {g.model: g for g in gs}
    for model in models:
        col = style.style_of(model).color
        g = by_model.get(model)
        if g is None:
            mid = x0 + (len(SLOTS) - 1) / 2
            ax.text(mid, 0.5, "no ladder", transform=ax.get_xaxis_transform(), ha="center",
                    va="center", fontsize=BASE - 0.5, color=MUTED)
            if last_row:
                ax.text(mid, -0.21, style.label_of(model), transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=BASE + 1, fontweight="bold", color=col)
            x0 += len(SLOTS) + GROUP_GAP
            continue
        base = g.levels[g.anchor].total(parts)
        n_over = 0
        for i, slot in enumerate(SLOTS):
            lv = g.levels.get(slot)
            if lv is None:
                continue
            bottom = 0.0
            for part in parts:
                h = lv.parts[part] / base
                ax.bar(x0 + i, h, BAR_W, bottom=bottom, zorder=3, **segment_style(col, part))
                bottom += h
            if bottom > cap:
                # A clipped bar: a white break across it near the top, and its height above.
                # Neighbouring overflows alternate between two heights so the labels never touch.
                ax.bar(x0 + i, cap * 0.030, BAR_W + 0.1, bottom=cap * 0.905, color=PAPER,
                       zorder=4, clip_on=False)
                ax.annotate(f"×{bottom:.0f}", (x0 + i, cap), textcoords="offset points",
                            xytext=(0, 2 + 7 * (n_over % 2)), ha="center", va="bottom",
                            fontsize=BASE - 1, color=col, fontweight="bold", zorder=5,
                            annotation_clip=False)
                n_over += 1
        mid = x0 + (len(SLOTS) - 1) / 2
        which = "" if g.anchor == ANCHOR else f" ({g.anchor})"
        ax.text(mid, -0.06, f"1×{which} = {fmt_anchor(base)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=BASE - 0.5, color=MUTED)
        if last_row:
            ax.text(mid, -0.21, style.label_of(g.model), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=BASE + 1, fontweight="bold", color=col)
        if first_row and model == models[0]:
            for i, slot in enumerate(SLOTS):
                ax.text(x0 + i, 1.0, SLOT_TICK[slot], transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=BASE - 0.5, color=MUTED)
        x0 += len(SLOTS) + GROUP_GAP
    ax.axhline(1.0, color=INK, lw=0.7, ls=(0, (2.2, 2.2)), zorder=2)
    ax.set_xlim(-0.7, x0 - GROUP_GAP - 0.3)
    ax.set_ylim(0, cap)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}×"))
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=3, labelsize=BASE)
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(INK)


# Vertical layout in inches, so a figure of any row count keeps the four-row figure's
# proportions: 6.8 x 6.1 in with top=0.962, bottom=0.095, hspace=0.38 gives these exactly.
TOP_IN = 6.1 * (1 - 0.962)
BOTTOM_IN = 6.1 * 0.095
ROW_AXES_IN = 6.1 * (0.962 - 0.095) / (4 + 3 * 0.38)
HSPACE = 0.38


def fig_height(n_rows: int) -> float:
    return TOP_IN + BOTTOM_IN + ROW_AXES_IN * (n_rows + HSPACE * (n_rows - 1))


def make(full: bool, rows: list[tuple[str, str]]) -> None:
    parts = ("input", "reasoning", "output") + (("wait",) if full else ())
    stem = "solvable_cost_full" if full else "solvable_cost"
    corrected = rows == [PRO_CORRECTED_ROW]
    deepswe = rows == [DEEPSWE_ROW]
    if corrected:
        stem += "_corrected"
    if deepswe:
        stem += f"_{DEEPSWE_KEY}"
    save_options = {"bbox_inches": "tight", "pad_inches": 0.04} if corrected else {}
    wage = latency.WAGE

    plt.rcParams.update({
        "font.family": style.use_font(), "font.size": BASE, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.9,
    })

    data = {key: groups(key) for key, _ in rows}
    # Every row places the same models at the same x, in the study's order. The DeepSWE figure
    # leaves out the models with no group on it.
    models = [m for m in style.MODELS
              if not deepswe or any(g.model == m for gs in data.values() for g in gs)]
    height = fig_height(len(rows))
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.8, height), squeeze=False)
    fig.subplots_adjust(left=0.062, right=0.972, top=1 - TOP_IN / height,
                        bottom=BOTTOM_IN / height, hspace=HSPACE)
    for r, (key, title) in enumerate(rows):
        ax = axes[r, 0]
        draw_row(ax, data[key], parts, first_row=(r == 0), last_row=(r == len(rows) - 1),
                 models=models)
        ax.set_ylabel(title, fontsize=BASE + 1.5, fontweight="bold", color=INK, labelpad=6)
        if r == 0:
            ax.text(1.0, 1.0, f"bars: none → {style.rung_tick('xhigh')}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=BASE - 0.5, color=MUTED)

    names = {"input": "Input & Context Re-reads", "reasoning": "Reasoning (hidden for proprietary LLMs)",
             "output": "Output",
             "wait": f"Waiting (decode time × \\${wage.usd_per_hour:.2f}/h)"}
    handles = [Patch(label=names[p], **segment_style(LEGEND_GREY, p)) for p in parts]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 6.1 * 0.008 / height), ncol=len(parts),
               frameon=False, fontsize=BASE + 0.5, handletextpad=0.5, columnspacing=1.8,
               borderpad=0, handlelength=1.6, handleheight=1.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if deepswe:
        for path in write_deepswe_csvs(data[DEEPSWE_KEY]):
            print(f"wrote {path}")
    if corrected:
        summary = OUT_DIR / f"{stem}.csv"
        with summary.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "effort", "n_always", "n_units", "anchor",
                             *parts, "total_usd", "relative_cost"])
            for g in data[PRO_CORRECTED_KEY]:
                base = g.levels[g.anchor].total(parts)
                for lv in g.levels.values():
                    writer.writerow([g.model, lv.effort, g.n_always, g.n_units, g.anchor,
                                     *(lv.parts[p] for p in parts), lv.total(parts),
                                     lv.total(parts) / base])
        print(f"wrote {summary}")

    out = OUT_DIR / f"{stem}.pdf"
    fig.savefig(out, format="pdf", **save_options)
    plt.close(fig)
    print(f"wrote {out}")

    print(f"\n{stem}: mean $ per always-solved unit, and × the anchor ({'+'.join(parts)})")
    for key, title in rows:
        print(f"\n{title}")
        for g in data[key]:
            base = g.levels[g.anchor].total(parts)
            print(f"  {style.label_of(g.model):18} always-solved {g.n_always:3}/{g.n_units}"
                  f"  1× ({g.anchor}) = {style.fmt_money(base)}")
            for slot in SLOTS:
                lv = g.levels.get(slot)
                if lv is None:
                    continue
                shares = "  ".join(f"{p[:4]} {lv.parts[p] / lv.total(parts):4.0%}" for p in parts)
                print(f"      {lv.effort:7} {style.fmt_money(lv.total(parts)):>9}"
                      f"  ×{lv.total(parts) / base:5.2f}   {shares}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="API $ + waiting-time $ (solvable_cost_full)")
    ap.add_argument("--both", action="store_true", help="write both the API and the full figure")
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--corrected", action="store_true",
                       help="draw corrected SWE-Bench Pro instead of the four primary benchmarks")
    which.add_argument("--deepswe", action="store_true",
                       help="draw the DeepSWE v1.1 row instead of the four primary benchmarks")
    a = ap.parse_args()
    chosen = [PRO_CORRECTED_ROW] if a.corrected else [DEEPSWE_ROW] if a.deepswe else ROWS
    for full in ([False, True] if a.both else [a.full]):
        make(full, chosen)
