#!/usr/bin/env python3
"""Naive laddering: run at the lowest effort, and retry a failure once at the highest effort.

    python routing/make_laddering.py                              # every benchmark
    python routing/make_laddering.py --dataset swebench_pro_corrected

No API calls: every (model, effort, unit) verdict and cost is on disk from the sweeps, so the
policy is scored from them. A unit the start level resolved is charged that level; a unit it
failed is charged the start level and the top level, and is resolved if the top level resolved
it. The pool is the self-router's and the oracle's (`vanilla.pools_for`).

The unit (`basis` column):

    instance   SWE-Bench Pro (corrected environment) and Terminal-Bench, k=1.
    attempt    AIME 2026 and GPQA Diamond, which draw several attempts per problem. Attempt i at
               the start level is paired with attempt i at the top level, and every bar is scored
               per attempt (pass@1). The self-router's per-problem choice is applied to each
               attempt and its selection call is paid once per problem.

The self-router's choices are read from OUTPUT_ROOT/self_routing/self_routing_choices.csv
(make_self_routing.py) where they exist, and re-priced on this table's basis.

Writes to OUTPUT_ROOT/laddering/:

    laddering_summary.csv   one row per (benchmark, model)
    laddering_levels.csv    every policy: fixed levels, `*random*`, `*self[low]*`,
                            `*ladder[low]*`, `*oracle*`
    laddering_choices.csv   one row per unit
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (REPO_ROOT, REPO_ROOT / "figures", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import style  # noqa: E402
import vanilla  # noqa: E402
from common import models, paths  # noqa: E402
from datasets import (DATASETS, MIN_UNITS, UNPLOTTED_EFFORTS, DatasetSpec, Run,  # noqa: E402
                      _pilot_ids, load_pro)
from style import effort_key  # noqa: E402
from vanilla import ModelPool  # noqa: E402

IN_DIR = paths.output_dir("self_routing")
OUT_DIR = paths.output_dir("laddering")

# The ladder starts at the lowest pooled level and jumps to the highest.
ARM = "low"
MODE = "jump"
# The self-router arm put beside the ladder (its judge effort).
SELF_JUDGE = "low"

# Benchmarks scored per attempt rather than per problem.
ATTEMPT_LEVEL = {"aime_2026", "gpqa_diamond"}

# SWE-Bench Pro in the corrected environment: the 44-instance pilot set, read from the
# swebench_pro_corrected tree.
SPECS: dict[str, DatasetSpec] = dict(DATASETS)
SPECS["swebench_pro_corrected"] = dataclasses.replace(
    DATASETS["swebench_pro_pilot"], key="swebench_pro_corrected",
    title="SWE-Bench-Pro (corrected)",
    scaffold="mini-swe-agent scaffold, k=1, 44-instance pilot subset, corrected environment",
    loader=lambda: load_pro("swebench-pro-corrected", subset=_pilot_ids()))
ORDER = ["swebench_pro_corrected", "terminal_bench", "aime_2026", "gpqa_diamond"]


# --------------------------------------------------------------------------- attempt-level pools

def attempt_pools_for(key: str) -> tuple[dict[str, ModelPool], list[str]]:
    """AIME / GPQA pools with one unit per (problem, attempt), read from the samples.

    Same models, effort order and exclusions as `datasets.load_aime` / `load_gpqa` (`none` out of
    the pool, a level dropped if it has no positive cost), but every attempt is its own unit.
    """
    if key == "aime_2026":
        sys.path.insert(0, str(REPO_ROOT / "aime_2026"))
        import aime as mod  # noqa: PLC0415
        idx, root = "problem_idx", paths.data_root("aime-2026")
    else:
        sys.path.insert(0, str(REPO_ROOT / "gpqa_diamond"))
        import gpqa as mod  # noqa: PLC0415
        idx, root = "question_idx", paths.data_root("gpqa-diamond")
    notes: list[str] = []
    out: dict[str, ModelPool] = {}
    for model in sorted(style.MODELS, key=style.model_sort_key):
        runs: list[Run] = []
        costs: dict[str, dict[str, float]] = {}
        resolved: dict[str, set[str]] = {}
        for effort in sorted(models.EFFORT_ORDER, key=effort_key):
            if effort in UNPLOTTED_EFFORTS:
                continue
            samples = mod.read_samples(mod.run_dir(root, model, effort) / mod.SAMPLES_FILE)
            if not samples:
                continue
            c = {f"{s[idx]}#{s['run_idx']}": (mod.cost_of(model, s["usage"]) or 0.0)
                 for s in samples}
            r = {f"{s[idx]}#{s['run_idx']}" for s in samples if s["correct"]}
            if len({s[idx] for s in samples}) < MIN_UNITS or sum(c.values()) <= 0:
                continue
            runs.append(Run(model=model, effort=effort, n_units=len(c), n_resolved=len(r),
                            accuracy=(0.0, 0.0, 0.0), total_cost=sum(c.values()),
                            resolved_units=r))
            costs[effort], resolved[effort] = c, r
        if len(runs) < 2:
            continue
        units = set.intersection(*[{u for u, v in costs[r.effort].items() if v > 0}
                                   for r in runs])
        n_full = max(r.n_units for r in runs)
        if n_full - len(units):
            notes.append(f"{key}/{style.label_of(model)}: {n_full - len(units)} of {n_full} "
                         f"attempts dropped — not present with positive cost at every level")
        out[model] = ModelPool(dataset=key, model=model, runs=runs, costs=costs,
                               resolved=resolved, units=units, n_full=n_full, attempts=1)
    return out, notes


def pools_for(key: str) -> tuple[dict[str, ModelPool], list[str]]:
    if key in ATTEMPT_LEVEL:
        return attempt_pools_for(key)
    pools, notes, _ = vanilla.pools_for(SPECS[key])
    return pools, notes


def problem_of(unit: str) -> str:
    """The self-router's unit for one of ours: the problem an attempt belongs to."""
    return unit.split("#", 1)[0]


# --------------------------------------------------------------------------- the policies

@dataclass
class Policy:
    """One bar: a way of paying for the units, and what it resolved."""
    tick: str
    cost: float
    resolved: float
    n_units: int
    router_cost: float = 0.0        # of `cost`, the selection calls (self-router only)
    first_pass: float = 0.0         # ladder only: the start level on every unit
    escalation: float = 0.0         # ladder only: the retries
    start_effort: str = ""
    top_effort: str = ""

    @property
    def resolve_pct(self) -> float:
        return 100.0 * self.resolved / self.n_units if self.n_units else 0.0


@dataclass
class LadderResult:
    start_effort: str
    top_effort: str
    path: list[str]
    units: list[str]
    first_pass_cost: float = 0.0
    escalation_cost: float = 0.0
    resolved_units: set[str] = field(default_factory=set)
    resolved_at_start: set[str] = field(default_factory=set)
    escalated: set[str] = field(default_factory=set)
    rung_of: dict[str, str] = field(default_factory=dict)
    cost_of: dict[str, float] = field(default_factory=dict)
    attempts_of: dict[str, int] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return self.first_pass_cost + self.escalation_cost

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def resolved(self) -> int:
        return len(self.resolved_units)

    @property
    def resolve_pct(self) -> float:
        return 100.0 * self.resolved / self.n_units if self.n_units else 0.0

    def policy(self) -> Policy:
        return Policy(tick=f"*ladder[{ARM}]*", cost=self.cost, resolved=self.resolved,
                      n_units=self.n_units, first_pass=self.first_pass_cost,
                      escalation=self.escalation_cost, start_effort=self.start_effort,
                      top_effort=self.top_effort)


def ladder(pool: ModelPool, units: list[str]) -> LadderResult:
    """Start at the lowest pooled level; on failure, retry once at the highest."""
    rungs = pool.pool
    first, top = rungs[0], rungs[-1]
    path = [first] + ([top] if top != first else [])
    r = LadderResult(start_effort=first, top_effort=top, path=path, units=units)
    for u in units:
        bill, n = 0.0, 0
        for i, e in enumerate(path):
            c = pool.costs[e][u]
            bill += c
            n += 1
            if i == 0:
                r.first_pass_cost += c
            else:
                r.escalation_cost += c
            if u in pool.resolved[e]:
                r.resolved_units.add(u)
                if i == 0:
                    r.resolved_at_start.add(u)
                break
        r.rung_of[u], r.cost_of[u], r.attempts_of[u] = e, bill, n
        if n > 1:
            r.escalated.add(u)
    return r


@dataclass
class Choice:
    effort: str
    router_cost: float


def read_choices() -> dict[tuple[str, str], dict[str, Choice]]:
    """{(benchmark, model): {unit: the self-router's level and the selection call's cost}}."""
    path = IN_DIR / "self_routing_choices.csv"
    out: dict[tuple[str, str], dict[str, Choice]] = defaultdict(dict)
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["dataset"], r["model"])][r["unit"]] = Choice(r["self_effort"],
                                                                float(r["router_cost"]))
    return dict(out)


def self_policy(pool: ModelPool, units: list[str], choices: dict[str, Choice]) -> Policy | None:
    """The self-router's choices re-priced on this table's basis.

    On the attempt basis a problem's choice applies to all its attempts, and the selection call
    is paid once per problem.
    """
    if not units or any(problem_of(u) not in choices for u in units):
        return None
    cost = resolved = 0.0
    for u in units:
        pick = choices[problem_of(u)].effort
        cost += pool.costs[pick][u]
        resolved += 1 if u in pool.resolved[pick] else 0
    router = sum(c.router_cost for p, c in choices.items()
                 if p in {problem_of(u) for u in units})
    return Policy(tick=f"*self[{SELF_JUDGE}]*", cost=cost + router, resolved=resolved,
                  n_units=len(units), router_cost=router)


# --------------------------------------------------------------------------- the cross-check

def read_self_levels() -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    path = IN_DIR / "self_routing_levels.csv"
    out: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["dataset"], r["model"])][r["effort"]] = (float(r["cost"]),
                                                            float(r["resolved"]))
    return dict(out)


def crosscheck(key: str, model: str, levels: list[Policy],
               theirs: dict[str, tuple[float, float]], attempt_level: bool) -> None:
    """These bars against the self-routing tables: bills to the cent, and resolve counts too
    except on the attempt basis, where they count a different unit."""
    for p in levels:
        if p.tick not in theirs:
            continue
        cost, resolved = theirs[p.tick]
        if abs(p.cost - cost) > 5e-4 or (not attempt_level and abs(p.resolved - resolved) > 6e-3):
            raise SystemExit(
                f"{key}/{model}: {p.tick} here is ${p.cost:.4f} @ {p.resolved} vs the "
                f"self-routing tables' ${cost:.4f} @ {resolved}")


# --------------------------------------------------------------------------- tables

SUMMARY_FIELDS = [
    "dataset", "model", "label", "basis", "arm", "mode", "n_units", "n_problems",
    "attempts_per_problem", "pool", "start_effort", "top_effort", "path",
    "first_pass_cost", "escalation_cost", "ladder_cost", "n_escalated", "escalated_pct",
    "resolved_at_start", "resolved_after_escalation", "ladder_resolved", "ladder_resolve_pct",
    "unresolved_at_top",
    "start_cost", "start_resolve_pct", "top_cost", "top_resolve_pct",
    "best_effort", "best_cost", "best_resolve_pct",
    "oracle_cost", "oracle_resolve_pct", "random_cost", "random_resolve_pct",
    "self_low_cost", "self_low_resolve_pct",
    "cost_vs_oracle_x", "cost_vs_top_x", "cost_vs_best_x", "cost_vs_start_x",
    "cost_vs_self_low_x",
    "saving_vs_top_pct", "saving_vs_best_pct", "extra_over_oracle_pct",
    "resolve_gap_vs_oracle_pts", "resolve_gap_vs_top_pts", "resolve_gap_vs_best_pts",
    "resolve_gap_vs_self_low_pts", "cost_per_resolved", "mean_attempts",
]

LEVEL_FIELDS = ["dataset", "model", "label", "basis", "mode", "effort", "n_units", "cost",
                "resolved", "resolve_pct", "cost_per_resolved", "router_cost",
                "first_pass_cost", "escalation_cost", "start_effort", "top_effort",
                "best_effort"]

CHOICE_FIELDS = ["dataset", "model", "basis", "mode", "arm", "unit", "start_effort", "ended_at",
                 "attempts", "escalated", "resolved", "resolved_at_start", "ladder_cost",
                 "start_cost", "top_cost", "oracle_effort", "oracle_cost",
                 "solvable_at_some_level"]


@dataclass
class Scored:
    key: str
    model: str
    basis: str
    units: list[str]
    pool: ModelPool
    fixed: list[Policy]
    random: Policy
    self_low: Policy | None
    ladder: LadderResult
    oracle: Policy
    best: Policy

    @property
    def n_problems(self) -> int:
        return len({problem_of(u) for u in self.units})


def score(key: str, model: str, pool: ModelPool, units: list[str],
          choices: dict[tuple[str, str], dict[str, Choice]]) -> Scored:
    n = len(units)
    fixed = [Policy(tick=l.effort, cost=l.cost, resolved=l.resolved, n_units=n)
             for l in pool.levels_over(set(units))]
    rnd = vanilla.random_over(pool, set(units))
    o_cost, o_resolved, _ = pool.oracle_over(set(units))
    # Most resolved, cheapest among ties: the self-routing tables' `best_effort` rule.
    best = min(fixed, key=lambda p: (-p.resolved, p.cost))
    return Scored(key=key, model=model, basis="attempt" if key in ATTEMPT_LEVEL else "instance",
                  units=units, pool=pool, fixed=fixed,
                  random=Policy(tick="*random*", cost=rnd.cost, resolved=rnd.resolved,
                                n_units=n),
                  self_low=self_policy(pool, units, choices.get((key, model), {})),
                  ladder=ladder(pool, units),
                  oracle=Policy(tick="*oracle*", cost=o_cost, resolved=o_resolved, n_units=n),
                  best=best)


def summary_row(s: Scored) -> dict:
    lv = {p.tick: p for p in s.fixed}
    n = len(s.units)
    x = lambda a, b: round(a / b, 3) if b else ""  # noqa: E731
    r, self_low = s.ladder, s.self_low
    start, top = lv[r.start_effort], lv[r.top_effort]
    return {
        "dataset": s.key, "model": s.model, "label": style.label_of(s.model),
        "basis": s.basis, "arm": ARM, "mode": MODE, "n_units": n,
        "n_problems": s.n_problems, "attempts_per_problem": round(n / s.n_problems, 2),
        "pool": "/".join(s.pool.pool), "start_effort": r.start_effort,
        "top_effort": r.top_effort, "path": "->".join(r.path),
        "first_pass_cost": round(r.first_pass_cost, 4),
        "escalation_cost": round(r.escalation_cost, 4), "ladder_cost": round(r.cost, 4),
        "n_escalated": len(r.escalated),
        "escalated_pct": round(100 * len(r.escalated) / n, 1),
        "resolved_at_start": len(r.resolved_at_start),
        "resolved_after_escalation": r.resolved - len(r.resolved_at_start),
        "ladder_resolved": r.resolved, "ladder_resolve_pct": round(r.resolve_pct, 1),
        "unresolved_at_top": n - r.resolved,
        "start_cost": round(start.cost, 4), "start_resolve_pct": round(start.resolve_pct, 1),
        "top_cost": round(top.cost, 4), "top_resolve_pct": round(top.resolve_pct, 1),
        "best_effort": s.best.tick, "best_cost": round(s.best.cost, 4),
        "best_resolve_pct": round(s.best.resolve_pct, 1),
        "oracle_cost": round(s.oracle.cost, 4),
        "oracle_resolve_pct": round(s.oracle.resolve_pct, 1),
        "random_cost": round(s.random.cost, 4),
        "random_resolve_pct": round(s.random.resolve_pct, 1),
        "self_low_cost": round(self_low.cost, 4) if self_low else "",
        "self_low_resolve_pct": round(self_low.resolve_pct, 1) if self_low else "",
        "cost_vs_oracle_x": x(r.cost, s.oracle.cost), "cost_vs_top_x": x(r.cost, top.cost),
        "cost_vs_best_x": x(r.cost, s.best.cost), "cost_vs_start_x": x(r.cost, start.cost),
        "cost_vs_self_low_x": x(r.cost, self_low.cost) if self_low else "",
        "saving_vs_top_pct": round(100 * (1 - r.cost / top.cost), 1) if top.cost else "",
        "saving_vs_best_pct": (round(100 * (1 - r.cost / s.best.cost), 1)
                               if s.best.cost else ""),
        "extra_over_oracle_pct": (round(100 * (r.cost / s.oracle.cost - 1), 1)
                                  if s.oracle.cost else ""),
        "resolve_gap_vs_oracle_pts": round(r.resolve_pct - s.oracle.resolve_pct, 1),
        "resolve_gap_vs_top_pts": round(r.resolve_pct - top.resolve_pct, 1),
        "resolve_gap_vs_best_pts": round(r.resolve_pct - s.best.resolve_pct, 1),
        "resolve_gap_vs_self_low_pts": (round(r.resolve_pct - self_low.resolve_pct, 1)
                                        if self_low else ""),
        "cost_per_resolved": round(r.cost / r.resolved, 4) if r.resolved else "",
        "mean_attempts": round(sum(r.attempts_of.values()) / n, 3),
    }


def level_rows(s: Scored) -> list[dict]:
    policies = [*s.fixed, s.random, *([s.self_low] if s.self_low else []),
                s.ladder.policy(), s.oracle]
    rows = []
    for p in policies:
        rows.append({
            "dataset": s.key, "model": s.model, "label": style.label_of(s.model),
            "basis": s.basis, "mode": MODE, "effort": p.tick, "n_units": p.n_units,
            "cost": round(p.cost, 4),
            "resolved": (int(p.resolved) if float(p.resolved).is_integer()
                         else round(p.resolved, 2)),
            "resolve_pct": round(p.resolve_pct, 1),
            "cost_per_resolved": round(p.cost / p.resolved, 4) if p.resolved else "",
            "router_cost": round(p.router_cost, 4) if p.router_cost else "",
            "first_pass_cost": round(p.first_pass, 4) if p.first_pass else "",
            "escalation_cost": round(p.escalation, 4) if p.first_pass else "",
            "start_effort": p.start_effort, "top_effort": p.top_effort,
            "best_effort": s.best.tick,
        })
    return rows


def choice_rows(s: Scored) -> list[dict]:
    pool, r = s.pool, s.ladder
    rows = []
    for u in sorted(r.units):
        solving = [e for e in pool.pool if u in pool.resolved[e]]
        pick = min(solving or pool.pool, key=lambda e: (pool.costs[e][u], effort_key(e)))
        rows.append({
            "dataset": s.key, "model": s.model, "basis": s.basis, "mode": MODE,
            "arm": ARM, "unit": u, "start_effort": r.start_effort,
            "ended_at": r.rung_of[u], "attempts": r.attempts_of[u],
            "escalated": u in r.escalated, "resolved": u in r.resolved_units,
            "resolved_at_start": u in r.resolved_at_start,
            "ladder_cost": round(r.cost_of[u], 5),
            "start_cost": round(pool.costs[r.start_effort][u], 5),
            "top_cost": round(pool.costs[r.top_effort][u], 5),
            "oracle_effort": pick, "oracle_cost": round(pool.costs[pick][u], 5),
            "solvable_at_some_level": bool(solving),
        })
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=ORDER,
                    help="benchmark to score (repeatable; default: all)")
    args = ap.parse_args()

    choices = read_choices()
    theirs = read_self_levels()

    summary, levels, choice_out, notes = [], [], [], []
    for key in args.dataset or ORDER:
        pools, pool_notes = pools_for(key)
        notes += pool_notes
        for model, pool in pools.items():
            units = sorted(pool.units)
            # Where the self-router was asked, score only the units it was asked about, so all
            # policies share one denominator.
            asked = choices.get((key, model))
            if asked:
                units = [u for u in units if problem_of(u) in asked]
                if len(units) != len(pool.units):
                    notes.append(f"{key}/{style.label_of(model)}: {len(pool.units) - len(units)} "
                                 f"routable units without a self-routing selection left out")
            if not units:
                continue
            s = score(key, model, pool, units, choices)
            if (key, model) in theirs:
                crosscheck(key, model, [*s.fixed, s.oracle, s.random,
                                        *([s.self_low] if s.self_low else [])]
                           if s.basis == "instance" else s.fixed,
                           theirs[(key, model)], s.basis == "attempt")
            summary.append(summary_row(s))
            levels += level_rows(s)
            choice_out += choice_rows(s)
            r = s.ladder
            print(f"{key:>22} {style.label_of(model):<18} [{s.basis}, n={len(units)}] "
                  f"{r.start_effort}->{r.top_effort} ${r.cost:.2f} @ {r.resolve_pct:.1f}% "
                  f"({len(r.escalated)} retried)" + ("" if s.self_low else "  [no self-router]"))

    write_csv(OUT_DIR / "laddering_summary.csv", SUMMARY_FIELDS, summary)
    write_csv(OUT_DIR / "laddering_levels.csv", LEVEL_FIELDS, levels)
    write_csv(OUT_DIR / "laddering_choices.csv", CHOICE_FIELDS, choice_out)
    print(f"wrote laddering_{{summary,levels,choices}}.csv ({len(summary)} rows) -> {OUT_DIR}")
    for note in notes:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
