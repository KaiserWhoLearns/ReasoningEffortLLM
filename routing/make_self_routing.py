#!/usr/bin/env python3
"""Score the self-router against the fixed effort levels, random allocation and the oracle.

    python routing/make_self_routing.py                          # every benchmark
    python routing/make_self_routing.py --dataset terminal_bench

Reads the selections recorded by `run_selection.py` (DATA_ROOT/self_routing/) and the sweeps on
disk; makes no API calls. Writes to OUTPUT_ROOT/self_routing/:

    self_routing_summary.csv    one row per (benchmark, model)
    self_routing_levels.csv     one row per (benchmark, model, effort), plus `*self*`,
                                `*random*` and `*oracle*` rows
    self_routing_choices.csv    one row per instance: the level chosen, the oracle's level, and
                                what each cost and resolved

The figures are drawn by figures/make_effort_allocation.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE.parent / "figures", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import selection  # noqa: E402
import style  # noqa: E402
import tasks  # noqa: E402
import vanilla  # noqa: E402
from common import paths  # noqa: E402
from datasets import DATASETS  # noqa: E402
from vanilla import SelfRouting  # noqa: E402

OUT_DIR = paths.output_dir("self_routing")
DATASET_KEYS = [k for k in DATASETS if k in tasks.SOURCE_OF]

SUMMARY_FIELDS = [
    "dataset", "model", "label", "variant", "judge_effort", "n_units", "n_routable", "pool",
    "self_cost", "router_cost", "self_total_cost", "self_resolved", "self_resolve_pct",
    "oracle_cost", "oracle_resolved", "oracle_resolve_pct",
    "best_effort", "best_cost", "best_resolve_pct",
    "cheapest_effort", "cheapest_cost", "cheapest_resolve_pct", "mean_level_cost",
    "random_menu", "random_cost", "random_cost_sd", "random_resolved", "random_resolved_sd",
    "random_resolve_pct", "random_captured_pct", "saving_vs_random_pct",
    "resolve_gap_vs_random_pts",
    "saving_vs_best_pct", "saving_vs_cheapest_pct", "saving_vs_mean_pct",
    "oracle_saving_vs_best_pct", "captured_pct", "cost_vs_oracle_x", "router_overhead_pct",
    "agree_with_oracle_pct", "cheaper_than_oracle", "dearer_than_oracle",
    "resolve_gap_vs_oracle_pts", "resolve_gap_vs_best_pts", "unresolved_but_solvable",
    "dominated_by", "draws_per_unit", "router_calls", "unanimous_pct",
    "n_answered", "format_ok_pct", "off_format_pct", "bad_reply_pct", "n_bad_reply",
    "n_unanswered", "unanswered_pct", "self_cost_per_resolved",
]

LEVEL_FIELDS = ["dataset", "model", "label", "variant", "effort", "n_units", "cost", "resolved",
                "resolve_pct", "cost_per_resolved", "self_units", "self_share_pct",
                "oracle_units", "oracle_share_pct"]

CHOICE_FIELDS = ["dataset", "model", "variant", "unit", "self_effort", "oracle_effort",
                 "self_cost", "oracle_cost", "cheapest_cost", "self_resolved",
                 "solvable_at_some_level", "router_cost", "draws", "parsed_by"]


def summary_rows(results: list[SelfRouting]) -> list[dict]:
    rows = []
    for r in results:
        best, cheap = r.best, r.cheapest
        rnd = r.random_router
        mean_cost = sum(l.cost for l in r.levels) / len(r.levels)
        oracle_saving = 1.0 - r.oracle_cost / best.cost if best.cost else 0.0
        rows.append({
            "dataset": r.dataset, "model": r.model, "label": style.label_of(r.model),
            "variant": r.variant, "judge_effort": r.judge_effort,
            "n_units": r.n_units, "n_routable": r.n_routable,
            "pool": "/".join(l.effort for l in r.levels),
            "self_cost": round(r.self_cost, 4), "router_cost": round(r.router_cost, 4),
            "self_total_cost": round(r.total_cost, 4),
            "self_resolved": r.self_resolved, "self_resolve_pct": round(r.resolve_pct, 1),
            "oracle_cost": round(r.oracle_cost, 4), "oracle_resolved": r.oracle_resolved,
            "oracle_resolve_pct": round(r.oracle_resolve_pct, 1),
            "best_effort": best.effort, "best_cost": round(best.cost, 4),
            "best_resolve_pct": round(best.resolve_pct(r.n_units), 1),
            "cheapest_effort": cheap.effort, "cheapest_cost": round(cheap.cost, 4),
            "cheapest_resolve_pct": round(cheap.resolve_pct(r.n_units), 1),
            "mean_level_cost": round(mean_cost, 4),
            "random_menu": rnd.menu_note if rnd else "",
            "random_cost": round(rnd.cost, 4) if rnd else "",
            "random_cost_sd": round(rnd.cost_sd, 4) if rnd else "",
            "random_resolved": round(rnd.resolved, 2) if rnd else "",
            "random_resolved_sd": round(rnd.resolved_sd, 2) if rnd else "",
            "random_resolve_pct": round(rnd.resolve_pct, 1) if rnd else "",
            "random_captured_pct": ("" if r.random_captured is None
                                    else round(100 * r.random_captured, 1)),
            "saving_vs_random_pct": ("" if r.saving_vs_random is None
                                     else round(100 * r.saving_vs_random, 1)),
            "resolve_gap_vs_random_pts": ("" if r.resolve_gap_vs_random_pts is None
                                          else round(r.resolve_gap_vs_random_pts, 1)),
            "saving_vs_best_pct": round(100 * r.saving_vs(best), 1),
            "saving_vs_cheapest_pct": round(100 * r.saving_vs(cheap), 1),
            "saving_vs_mean_pct": round(100 * r.saving_vs_mean, 1),
            "oracle_saving_vs_best_pct": round(100 * oracle_saving, 1),
            "captured_pct": ("" if r.captured is None else round(100 * r.captured, 1)),
            "cost_vs_oracle_x": (round(r.total_cost / r.oracle_cost, 3) if r.oracle_cost else ""),
            "router_overhead_pct": round(r.overhead_pct, 2),
            "agree_with_oracle_pct": round(r.agree_pct, 1),
            "cheaper_than_oracle": r.n_cheaper, "dearer_than_oracle": r.n_dearer,
            "resolve_gap_vs_oracle_pts": round(r.resolve_pct - r.oracle_resolve_pct, 1),
            "resolve_gap_vs_best_pts": round(r.resolve_pct - best.resolve_pct(r.n_units), 1),
            "unresolved_but_solvable": r.lost,
            "dominated_by": "/".join(l.effort for l in r.dominated_by),
            "draws_per_unit": round(r.draws_per_unit, 2), "router_calls": r.router_calls,
            # Agreement between repeated draws; empty with one selection per instance.
            "unanimous_pct": "",
            "n_answered": r.n_answered,
            "format_ok_pct": round(r.format_ok_pct, 1),
            "off_format_pct": round(r.off_format_pct, 1),
            "bad_reply_pct": round(r.bad_reply_pct, 1),
            "n_bad_reply": r.n_fallback,
            "n_unanswered": r.n_unanswered,
            "unanswered_pct": round(r.unanswered_pct, 1),
            "self_cost_per_resolved": (round(r.total_cost / r.self_resolved, 4)
                                       if r.self_resolved else ""),
        })
    return rows


def level_rows(results: list[SelfRouting]) -> list[dict]:
    rows = []
    for r in results:
        for lv in r.levels:
            rows.append({
                "dataset": r.dataset, "model": r.model, "label": style.label_of(r.model),
                "variant": r.variant, "effort": lv.effort, "n_units": r.n_units,
                "cost": round(lv.cost, 4), "resolved": lv.resolved,
                "resolve_pct": round(lv.resolve_pct(r.n_units), 1),
                "cost_per_resolved": round(lv.cost / lv.resolved, 4) if lv.resolved else "",
                "self_units": r.mix.get(lv.effort, 0),
                "self_share_pct": round(100 * r.mix.get(lv.effort, 0) / r.n_units, 1),
                "oracle_units": r.oracle_mix.get(lv.effort, 0),
                "oracle_share_pct": round(100 * r.oracle_mix.get(lv.effort, 0) / r.n_units, 1),
            })
        policies = [("self", r.total_cost, float(r.self_resolved))]
        if r.random_router is not None:
            policies.append(("random", r.random_router.cost, r.random_router.resolved))
        policies.append(("oracle", r.oracle_cost, float(r.oracle_resolved)))
        for tick, cost, resolved in policies:
            rows.append({
                "dataset": r.dataset, "model": r.model, "label": style.label_of(r.model),
                "variant": r.variant, "effort": f"*{tick}*", "n_units": r.n_units,
                "cost": round(cost, 4),
                # The random router's count is an expectation; the others are whole numbers.
                "resolved": (int(resolved) if float(resolved).is_integer()
                             else round(resolved, 2)),
                "resolve_pct": round(100 * resolved / r.n_units, 1) if r.n_units else 0.0,
                "cost_per_resolved": round(cost / resolved, 4) if resolved else "",
                "self_units": r.n_units, "self_share_pct": 100.0,
                "oracle_units": r.n_units, "oracle_share_pct": 100.0,
            })
    return rows


def choice_rows(spec_key: str) -> list[dict]:
    """One row per scored instance."""
    spec = DATASETS[spec_key]
    source = tasks.SOURCE_OF[spec_key]
    pools, _, _ = vanilla.pools_for(spec)
    rows = []
    for model, pool in pools.items():
        votes, _ = vanilla.choices_for(pool, source, selection.judge_effort_for(model))
        choices, costs = votes.choices, votes.costs
        units = pool.units & set(choices)
        if not units:
            continue
        levels = pool.levels_over(units)
        fallback = min(levels, key=lambda l: (-l.resolved, l.cost)).effort
        for unit in sorted(units):
            pick = choices[unit] or fallback
            solving = [e for e in pool.pool if unit in pool.resolved[e]]
            oracle_pick = min(solving or pool.pool,
                              key=lambda e: (pool.costs[e][unit], style.effort_key(e)))
            rows.append({
                "dataset": spec_key, "model": model, "variant": selection.VARIANT,
                "unit": unit, "self_effort": pick, "oracle_effort": oracle_pick,
                "self_cost": round(pool.costs[pick][unit], 5),
                "oracle_cost": round(pool.costs[oracle_pick][unit], 5),
                "cheapest_cost": round(min(pool.costs[e][unit] for e in pool.pool), 5),
                "self_resolved": unit in pool.resolved[pick],
                "solvable_at_some_level": bool(solving),
                "router_cost": round(costs.get(unit, 0.0), 6),
                # The level the reply named (empty when it named none), and how it was parsed.
                "draws": choices[unit] or "",
                "parsed_by": votes.parsed_by.get(unit, ""),
            })
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=DATASET_KEYS,
                    help="benchmark to score (repeatable; default: all)")
    args = ap.parse_args()

    summary, levels, choices, notes = [], [], [], []
    for key in args.dataset or DATASET_KEYS:
        results, _bounds, dataset_notes, _omitted = vanilla.route_dataset_self(DATASETS[key])
        notes.extend(dataset_notes)
        if not results:
            print(f"{key}: no selections on disk")
            continue
        summary += summary_rows(results)
        levels += level_rows(results)
        choices += choice_rows(key)

    write_csv(OUT_DIR / "self_routing_summary.csv", SUMMARY_FIELDS, summary)
    write_csv(OUT_DIR / "self_routing_levels.csv", LEVEL_FIELDS, levels)
    write_csv(OUT_DIR / "self_routing_choices.csv", CHOICE_FIELDS, choices)
    print(f"wrote self_routing_{{summary,levels,choices}}.csv ({len(summary)} model-benchmark "
          f"rows) -> {OUT_DIR}")
    for note in notes:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
