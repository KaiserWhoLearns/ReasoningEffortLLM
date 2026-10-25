"""Oracle effort router: the cheapest recorded successful effort per instance.

For each instance the oracle picks the cheapest effort level that resolved it, and if no level
did, the cheapest level overall. It chooses with the verdict in hand, so it is a bound on what
per-instance effort routing can save, not a deployable policy.

The routing pool and instance set follow three rules:

* `none` is not in the pool (`datasets.UNPLOTTED_EFFORTS`); the pool is the reasoning ladder.
* A level that billed $0 over the whole run is a dead run and leaves the pool.
* An instance is routable only where every pooled level has a positive recorded cost for it
  (a $0 instance is a crashed or empty trajectory, not a free one). Every fixed level is
  re-scored over the same routable set.

Used by `vanilla.py` (self-routing) and `make_laddering.py`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "figures"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import style  # noqa: E402
from datasets import MIN_UNITS, UNPLOTTED_EFFORTS, DatasetSpec, Run, rows_for  # noqa: E402
from style import effort_key  # noqa: E402

# A model whose routable set is below this fraction of the benchmark is omitted: its bound would
# be computed on a different instance set from every other model's.
MIN_ROUTABLE_FRACTION = 0.6


@dataclass
class Level:
    """One fixed effort level, re-scored over the routed instance set."""
    effort: str
    cost: float
    resolved: int

    def resolve_pct(self, n: int) -> float:
        return 100.0 * self.resolved / n if n else 0.0


@dataclass
class ModelRouting:
    """One model's oracle bound on one benchmark."""
    dataset: str
    model: str
    n_units: int                      # instances the bound is computed over
    n_full: int                       # instances the benchmark has for this model
    n_dropped: int                    # dropped for want of a positive cost at every level
    levels: list[Level]
    oracle_cost: float
    oracle_resolved: int              # instances solved by at least one pooled level
    mix: dict[str, int] = field(default_factory=dict)         # effort -> instances routed to it
    mix_solved: dict[str, int] = field(default_factory=dict)  # ... of which resolved
    unsolvable_cost: float = 0.0      # spent on instances no level solves
    attempts: int = 1                 # attempts per unit; >1 only on AIME and GPQA Diamond


def unit_costs(run: Run) -> tuple[dict[str, float], float]:
    """Per-unit cost in the currency of `run.total_cost`, and the relative gap that was closed.

    AIME and GPQA store a per-attempt mean, so a unit's cost is that mean times the attempts per
    unit. A unit that drew fewer attempts is over-priced by this reconstruction, so the per-unit
    costs are rescaled to sum to the run's own total.
    """
    raw = {u: v * run.attempts_per_unit for u, v in run.values.get("cost", {}).items()}
    total = sum(raw.values())
    if total <= 0 or run.total_cost <= 0:
        return raw, 0.0
    scale = run.total_cost / total
    return {u: v * scale for u, v in raw.items()}, total / run.total_cost - 1


def route_model(dataset: str, model: str, runs: list[Run]) -> tuple[ModelRouting | None, list[str]]:
    """The bound for one model, or None with the reason it could not be computed."""
    notes: list[str] = []
    pool = [r for r in runs if r.effort not in UNPLOTTED_EFFORTS]
    label = style.label_of(model)

    for r in pool:
        if r.total_cost <= 0:
            notes.append(f"{dataset}/{label}: dropped level `{r.effort}` from the pool — it "
                         f"billed $0 over all {r.n_units} units")
    pool = [r for r in pool if r.total_cost > 0]
    if len(pool) < 2:
        return None, notes + [f"{dataset}/{label}: only {len(pool)} live effort level(s), "
                              f"nothing to route between"]
    pool.sort(key=lambda r: effort_key(r.effort))

    costs = {}
    for r in pool:
        costs[r.effort], gap = unit_costs(r)
        if abs(gap) > 0.005:
            notes.append(f"{dataset}/{label} {r.effort}: per-unit costs rescaled by "
                         f"{100 * gap:+.1f}% to the run's ${r.total_cost:.2f} (some units drew "
                         f"fewer than {r.attempts_per_unit} attempts)")
    n_full = max(r.n_units for r in pool)
    units = set.intersection(*[{u for u, c in costs[r.effort].items() if c > 0} for r in pool])
    dropped = n_full - len(units)
    if dropped:
        notes.append(f"{dataset}/{label}: {dropped} of {n_full} units dropped — no "
                     f"positive recorded cost at every pooled level (crashed or empty "
                     f"trajectories); the bound and every level beside it are scored over the "
                     f"remaining {len(units)}")
    if len(units) < MIN_UNITS or len(units) < MIN_ROUTABLE_FRACTION * n_full:
        return None, notes + [f"{dataset}/{label}: only {len(units)} of {n_full} units are "
                              f"routable, omitted"]

    levels = [Level(effort=r.effort,
                    cost=sum(costs[r.effort][u] for u in units),
                    resolved=sum(1 for u in units if u in r.resolved_units))
              for r in pool]

    solved_by = {r.effort: r.resolved_units for r in pool}
    choice: dict[str, str] = {}
    oracle_cost = unsolvable_cost = 0.0
    for u in units:
        solving = [r.effort for r in pool if u in solved_by[r.effort]]
        candidates = solving or [r.effort for r in pool]
        pick = min(candidates, key=lambda e: (costs[e][u], effort_key(e)))
        choice[u] = pick
        oracle_cost += costs[pick][u]
        if not solving:
            unsolvable_cost += costs[pick][u]

    mix = {r.effort: 0 for r in pool}
    mix_solved = {r.effort: 0 for r in pool}
    for u, e in choice.items():
        mix[e] += 1
        if u in solved_by[e]:
            mix_solved[e] += 1

    return ModelRouting(
        dataset=dataset, model=model, n_units=len(units), n_full=n_full, n_dropped=dropped,
        levels=levels, oracle_cost=oracle_cost,
        oracle_resolved=sum(mix_solved.values()), mix=mix, mix_solved=mix_solved,
        unsolvable_cost=unsolvable_cost,
        attempts=max(r.attempts_per_unit for r in pool)), notes


def route_dataset(spec: DatasetSpec) -> tuple[list[ModelRouting], list[str], list[str]]:
    """Every model's bound on one benchmark, in display order.

    Returns the bounds, the load notes, and the models that have runs but no computable bound.
    Runs come through `datasets.rows_for`, so the same partial-run drops as the figures apply.
    """
    _, notes, runs = rows_for(spec)
    by_model: dict[str, list[Run]] = {}
    for r in runs:
        by_model.setdefault(r.model, []).append(r)

    out, omitted = [], []
    for model in sorted(by_model, key=style.model_sort_key):
        result, model_notes = route_model(spec.key, model, by_model[model])
        notes.extend(model_notes)
        if result is None:
            omitted.append(model)
        else:
            out.append(result)
    return out, notes, omitted
