"""Per-unit effort ladders: which units each effort level of a model resolves.

A `Ladder` is one model on one benchmark: its reasoning effort levels in order and the set of
units each level resolves. `ladders(key)` builds them from the run trees for a
`datasets.DATASETS` key or `PRO_CORRECTED_KEY` (corrected SWE-Bench Pro on the pilot instances).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import style
from datasets import DATASETS, UNPLOTTED_EFFORTS, _pilot_ids, load_pro

PRO_CORRECTED_KEY = "swebench_pro_corrected"

# A benchmark's own size. A run covering a different number of units is left out of a ladder;
# benchmarks without an entry use the most common unit count on disk.
BENCH_N = {"swebench_pro_pilot": 44, PRO_CORRECTED_KEY: 44, "terminal_bench": 89}

LADDER = ["none", "low", "medium", "high", "xhigh"]
RUNGS = [e for e in LADDER if e not in UNPLOTTED_EFFORTS]


@dataclass
class Ladder:
    """One model's reasoning ladder on one benchmark, with per-unit verdicts."""
    model: str
    benchmark: str
    n_units: int
    efforts: list[str]                    # in ladder order
    resolved: dict[str, set[str]]         # effort -> units it resolved


def load_runs(key: str):
    """(runs, title) for a `DATASETS` key or `PRO_CORRECTED_KEY`."""
    if key == PRO_CORRECTED_KEY:
        return load_pro("swebench-pro-corrected", subset=_pilot_ids())[0], \
            "SWE-Bench-Pro (corrected)"
    return DATASETS[key].loader()[0], DATASETS[key].title


def ladders(key: str) -> list[Ladder]:
    """Per-unit ladders for one benchmark, in the study's model order.

    Keeps runs over the benchmark's own unit count and reasoning levels only (`none` excluded),
    and drops any model left with fewer than two levels.
    """
    runs, title = load_runs(key)
    n = BENCH_N.get(key) or Counter(r.n_units for r in runs).most_common(1)[0][0]
    by_model: dict[str, list] = defaultdict(list)
    for r in runs:
        if r.effort in UNPLOTTED_EFFORTS or r.n_units != n:
            continue
        by_model[r.model].append(r)
    out = []
    for model, rs in by_model.items():
        rs.sort(key=lambda r: style.effort_key(r.effort))
        if len(rs) < 2:
            continue
        out.append(Ladder(model=model, benchmark=title, n_units=n,
                          efforts=[r.effort for r in rs],
                          resolved={r.effort: set(r.resolved_units) for r in rs}))
    out.sort(key=lambda lad: style.model_sort_key(lad.model))
    return out
