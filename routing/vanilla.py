"""Scoring the self-router: what a model's own effort choices cost and resolve.

No API calls. The choices are on disk (`run_selection.py`) and so is what every level cost and
resolved on every instance (the sweeps), so scoring is a join: each instance is charged the run
its chosen level made on it, plus the selection call.

The routing pool is rebuilt here by `routing.py`'s rules and checked against
`routing.route_model` before anything is scored (`crosscheck`).

The self-router is charged for its selection calls (`router_cost`, included in `total_cost`).
A level that fails an instance leaves it unresolved; there is no retry. A reply naming no level
on the menu runs at the model's most accurate fixed level and is counted in `n_fallback`.

The control is a random router (`random_over`) that draws a level uniformly from the same menu
per instance; its cost and resolve count are exact expectations over that draw.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (REPO_ROOT, REPO_ROOT / "figures", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import routing  # noqa: E402
import selection  # noqa: E402
import style  # noqa: E402
import tasks  # noqa: E402
from datasets import MIN_UNITS, UNPLOTTED_EFFORTS, DatasetSpec, Run, rows_for  # noqa: E402
from routing import Level, ModelRouting  # noqa: E402
from style import effort_key  # noqa: E402

MIN_ROUTABLE_FRACTION = routing.MIN_ROUTABLE_FRACTION


# --------------------------------------------------------------------------- the shared pool

@dataclass
class ModelPool:
    """One model's routable instances on one benchmark, and what each level did on each."""
    dataset: str
    model: str
    runs: list[Run]                                  # live pooled levels, ladder order
    costs: dict[str, dict[str, float]]               # effort -> unit -> cost
    resolved: dict[str, set[str]]                    # effort -> units it resolved
    units: set[str]
    n_full: int
    attempts: int

    @property
    def pool(self) -> list[str]:
        return [r.effort for r in self.runs]

    def levels_over(self, units: set[str]) -> list[Level]:
        """Every fixed level, re-scored over `units`."""
        return [Level(effort=e,
                      cost=sum(self.costs[e][u] for u in units),
                      resolved=sum(1 for u in units if u in self.resolved[e]))
                for e in self.pool]

    def oracle_over(self, units: set[str]) -> tuple[float, int, dict[str, int]]:
        """(cost, resolved, mix) of the oracle bound, restricted to `units`."""
        cost, solved, mix = 0.0, 0, {e: 0 for e in self.pool}
        for u in units:
            solving = [e for e in self.pool if u in self.resolved[e]]
            pick = min(solving or self.pool, key=lambda e: (self.costs[e][u], effort_key(e)))
            cost += self.costs[pick][u]
            mix[pick] += 1
            solved += 1 if solving else 0
        return cost, solved, mix


def pool_of(dataset: str, model: str, runs: list[Run]) -> tuple[ModelPool | None, list[str]]:
    """One model's pool under routing.py's rules, or None with the reason there is none."""
    notes: list[str] = []
    label = style.label_of(model)
    pool = [r for r in runs if r.effort not in UNPLOTTED_EFFORTS and r.total_cost > 0]
    if len(pool) < 2:
        return None, [f"{dataset}/{label}: only {len(pool)} live effort level(s), nothing to "
                      f"route between"]
    pool.sort(key=lambda r: effort_key(r.effort))

    costs = {r.effort: routing.unit_costs(r)[0] for r in pool}
    n_full = max(r.n_units for r in pool)
    units = set.intersection(*[{u for u, c in costs[r.effort].items() if c > 0} for r in pool])
    if n_full - len(units):
        # Worded as routing.py words it, so the two notes collapse to one.
        notes.append(f"{dataset}/{label}: {n_full - len(units)} of {n_full} units dropped — no "
                     f"positive recorded cost at every pooled level (crashed or empty "
                     f"trajectories); the bound and every level beside it are scored over the "
                     f"remaining {len(units)}")
    if len(units) < MIN_UNITS or len(units) < MIN_ROUTABLE_FRACTION * n_full:
        return None, notes + [f"{dataset}/{label}: only {len(units)} of {n_full} units are "
                              f"routable, omitted"]
    return ModelPool(dataset=dataset, model=model, runs=pool, costs=costs,
                     resolved={r.effort: r.resolved_units for r in pool}, units=units,
                     n_full=n_full, attempts=max(r.attempts_per_unit for r in pool)), notes


def pools_for(spec: DatasetSpec) -> tuple[dict[str, ModelPool], list[str], list[str]]:
    """Every model's pool on one benchmark, in display order, plus notes and omitted models."""
    _, notes, runs = rows_for(spec)
    by_model: dict[str, list[Run]] = {}
    for r in runs:
        by_model.setdefault(r.model, []).append(r)

    out: dict[str, ModelPool] = {}
    omitted: list[str] = []
    for model in sorted(by_model, key=style.model_sort_key):
        pool, pool_notes = pool_of(spec.key, model, by_model[model])
        notes.extend(pool_notes)
        if pool is None:
            omitted.append(model)
        else:
            out[model] = pool
    return out, notes, omitted


def crosscheck(pool: ModelPool, oracle: ModelRouting) -> None:
    """This module's pool against `routing.route_model`'s bound, to the cent. Raises on mismatch."""
    cost, solved, _ = pool.oracle_over(pool.units)
    mine = {l.effort: (round(l.cost, 6), l.resolved) for l in pool.levels_over(pool.units)}
    theirs = {l.effort: (round(l.cost, 6), l.resolved) for l in oracle.levels}
    if len(pool.units) != oracle.n_units or mine != theirs \
            or abs(cost - oracle.oracle_cost) > 1e-6 or solved != oracle.oracle_resolved:
        raise SystemExit(
            f"{pool.dataset}/{pool.model}: pool disagrees with routing.py — n "
            f"{len(pool.units)} vs {oracle.n_units}, oracle ${cost:.4f} vs "
            f"${oracle.oracle_cost:.4f}")


# --------------------------------------------------------------------------- the random control

@dataclass
class RandomRouting:
    """A router that picks uniformly from the self-router's menu, per instance.

    `cost` and `resolved` are exact expectations over the draw:
        E[bill]     = sum over instances of the mean cost of the menu options
        E[resolved] = sum over instances of the share of menu options that resolve it
    `cost_sd` / `resolved_sd` are the standard deviations of a single draw. It pays no router cost.
    """
    menu: list[str]                       # the option names it drew between
    to_level: dict[str, str]              # option name -> the pooled level it is charged at
    n_units: int
    cost: float
    resolved: float
    cost_sd: float
    resolved_sd: float
    mix: dict[str, float]                 # expected instances per pooled level (n / |menu|)

    @property
    def resolve_pct(self) -> float:
        return 100.0 * self.resolved / self.n_units if self.n_units else 0.0

    @property
    def menu_note(self) -> str:
        """The draw as `option->level` pairs."""
        return "/".join(f"{name}->{self.to_level[name]}" for name in self.menu)


def random_over(pool: ModelPool, units: set[str]) -> RandomRouting:
    """The uniform-random router over `pool`'s menu, in closed form."""
    menu, to_level = selection.build_menu(pool.pool)
    picks = [to_level[name] for name in menu]
    k = len(picks)
    cost = resolved = var_cost = var_resolved = 0.0
    for u in units:
        costs = [pool.costs[e][u] for e in picks]
        mean = sum(costs) / k
        cost += mean
        # Independent across instances, so the per-instance variances add.
        var_cost += sum((c - mean) ** 2 for c in costs) / k
        p = sum(1 for e in picks if u in pool.resolved[e]) / k
        resolved += p
        var_resolved += p * (1.0 - p)
    n = len(units)
    return RandomRouting(
        menu=menu, to_level=to_level, n_units=n, cost=cost, resolved=resolved,
        cost_sd=var_cost ** 0.5, resolved_sd=var_resolved ** 0.5,
        mix={e: n / k for e in picks})


# --------------------------------------------------------------------------- the self-router

@dataclass
class SelfRouting:
    """One model's own effort choices on one benchmark, priced against its recorded runs."""
    dataset: str
    model: str
    variant: str
    judge_effort: str
    n_units: int                      # instances scored (pool ∩ instances with a selection)
    n_routable: int                   # instances in the pool
    levels: list[Level]               # every fixed level, re-scored over the scored instances
    oracle_cost: float
    oracle_resolved: int
    oracle_mix: dict[str, int]
    self_cost: float                  # the routed runs' bill, selection calls not included
    self_resolved: int
    router_cost: float                # the selection calls
    mix: dict[str, int] = field(default_factory=dict)
    mix_solved: dict[str, int] = field(default_factory=dict)
    n_fallback: int = 0               # replies that named no level on the menu
    n_unanswered: int = 0             # instances whose selection call never came back
    n_agree: int = 0                  # routed to the oracle's level
    n_cheaper: int = 0                # routed to a level cheaper than the oracle's on that unit
    n_dearer: int = 0                 # ... dearer
    lost: int = 0                     # resolved by the oracle, not by the self-router
    won: int = 0                      # resolved by the self-router and by no level (always 0)
    attempts: int = 1
    router_calls: int = 0
    parsed_by: dict[str, int] = field(default_factory=dict)
    random_router: RandomRouting | None = None

    @property
    def best(self) -> Level:
        """The most accurate fixed level, ties broken by price."""
        return min(self.levels, key=lambda l: (-l.resolved, l.cost))

    @property
    def cheapest(self) -> Level:
        return min(self.levels, key=lambda l: l.cost)

    @property
    def total_cost(self) -> float:
        """The routed runs plus the selection calls."""
        return self.self_cost + self.router_cost

    @property
    def resolve_pct(self) -> float:
        return 100.0 * self.self_resolved / self.n_units if self.n_units else 0.0

    @property
    def oracle_resolve_pct(self) -> float:
        return 100.0 * self.oracle_resolved / self.n_units if self.n_units else 0.0

    def saving_vs(self, level: Level) -> float:
        """Fraction of `level`'s bill this policy does not pay. Negative = it costs more."""
        return 1.0 - self.total_cost / level.cost if level.cost else 0.0

    @property
    def saving_vs_mean(self) -> float:
        mean = sum(l.cost for l in self.levels) / len(self.levels)
        return 1.0 - self.total_cost / mean if mean else 0.0

    @property
    def dominated_by(self) -> list[Level]:
        """Fixed levels both no dearer than this policy and at least as accurate."""
        return [l for l in self.levels
                if l.cost <= self.total_cost and l.resolved >= self.self_resolved]

    def captured_by(self, cost: float) -> float | None:
        """Share of the oracle's saving over the best fixed level that a bill of `cost` realises.

        1.0 is the bound, 0.0 the best fixed level. None when the bound saves nothing.
        """
        headroom = self.best.cost - self.oracle_cost
        if headroom <= 0:
            return None
        return (self.best.cost - cost) / headroom

    @property
    def captured(self) -> float | None:
        return self.captured_by(self.total_cost)

    @property
    def random_captured(self) -> float | None:
        return None if self.random_router is None \
            else self.captured_by(self.random_router.cost)

    @property
    def saving_vs_random(self) -> float | None:
        """Fraction of the random router's expected bill this policy does not pay."""
        if self.random_router is None or not self.random_router.cost:
            return None
        return 1.0 - self.total_cost / self.random_router.cost

    @property
    def resolve_gap_vs_random_pts(self) -> float | None:
        return None if self.random_router is None \
            else self.resolve_pct - self.random_router.resolve_pct

    @property
    def overhead_pct(self) -> float:
        """The selection calls as a share of the policy's bill."""
        return 100.0 * self.router_cost / self.total_cost if self.total_cost else 0.0

    @property
    def n_answered(self) -> int:
        """Instances whose reply named a level on the menu."""
        return self.n_units - self.n_fallback

    @property
    def draws_per_unit(self) -> float:
        """Usable answers per scored instance (one selection call each)."""
        return self.n_answered / self.n_units if self.n_units else 0.0

    @property
    def format_ok_pct(self) -> float:
        """Replies with the requested `EFFORT: <level>` line."""
        return 100.0 * self.parsed_by.get("line", 0) / self.router_calls \
            if self.router_calls else 0.0

    @property
    def off_format_pct(self) -> float:
        """Replies that named a level outside the requested line."""
        return 100.0 * self.parsed_by.get("scan", 0) / self.router_calls \
            if self.router_calls else 0.0

    @property
    def bad_reply_pct(self) -> float:
        """Replies naming no level at all."""
        return 100.0 * self.n_fallback / self.n_units if self.n_units else 0.0

    @property
    def unanswered_pct(self) -> float:
        """Instances whose call never came back, as a share of all instances asked."""
        asked = self.n_units + self.n_unanswered
        return 100.0 * self.n_unanswered / asked if asked else 0.0

    @property
    def agree_pct(self) -> float:
        return 100.0 * self.n_agree / self.n_units if self.n_units else 0.0


# --------------------------------------------------------------------------- reading choices

@dataclass
class Votes:
    """One model's scored selections on one benchmark."""
    choices: dict[str, str | None]        # unit -> the pooled level chosen (None: no level named)
    costs: dict[str, float]               # unit -> selection call cost
    parsed_by: dict[str, str]             # unit -> parse rule
    unanswered: int                       # units whose calls failed and were never re-asked


def choices_for(pool: ModelPool, source: str, judge_effort: str) -> tuple[Votes, list[str]]:
    """The recorded selection for every unit of this pool (`selection.is_scored` records only).

    A unit asked more than once keeps its latest answer.
    """
    menu, to_level = selection.build_menu(pool.pool)
    notes: list[str] = []
    latest: dict[str, dict] = {}
    for record in selection.read_records(selection.record_path(source, pool.model)):
        if selection.is_scored(record, menu, judge_effort) and record["unit"] in pool.units:
            latest[record["unit"]] = record

    choices: dict[str, str | None] = {}
    costs: dict[str, float] = {}
    parsed: dict[str, str] = {}
    for unit, r in latest.items():
        name = r.get("choice")
        choices[unit] = to_level.get(name) if name in menu else None
        costs[unit] = r.get("cost_usd") or 0.0
        parsed[unit] = r.get("parsed_by", "unparsed")

    label = f"{pool.dataset}/{style.label_of(pool.model)}"
    unanswered = len({r["unit"] for r in
                      selection.read_records(selection.error_path(source, pool.model))
                      if selection.is_scored(r, menu, judge_effort)
                      and r["unit"] in pool.units and r["unit"] not in choices})
    if unanswered:
        notes.append(f"{label}: {unanswered} instance(s) never answered (the selection call "
                     f"failed after every retry); reported, not scored")
    missing = len(pool.units) - len(choices)
    if missing and choices:
        notes.append(f"{label}: {missing} of {len(pool.units)} routable units have no "
                     f"selection; every policy is scored over the {len(choices)} that do")
    return Votes(choices=choices, costs=costs, parsed_by=parsed, unanswered=unanswered), notes


def evaluate(pool: ModelPool, votes: Votes, *, judge_effort: str) -> SelfRouting:
    """Score one model's choices against its recorded runs.

    Only units in both the pool and `votes.choices` are scored, and the oracle and every fixed
    level are re-scored over exactly those units.
    """
    choices, router_costs = votes.choices, votes.costs
    units = pool.units & set(choices)
    levels = pool.levels_over(units)
    oracle_cost, oracle_resolved, oracle_mix = pool.oracle_over(units)
    best = min(levels, key=lambda l: (-l.resolved, l.cost))

    mix = {e: 0 for e in pool.pool}
    mix_solved = {e: 0 for e in pool.pool}
    self_cost = router_cost = 0.0
    self_resolved = n_fallback = n_agree = n_cheaper = n_dearer = lost = won = 0
    for u in sorted(units):
        pick = choices[u]
        if pick is None:
            # No usable answer: run the most accurate fixed level.
            pick = best.effort
            n_fallback += 1
        self_cost += pool.costs[pick][u]
        router_cost += router_costs.get(u) or 0.0
        mix[pick] += 1
        hit = u in pool.resolved[pick]
        self_resolved += hit
        mix_solved[pick] += hit

        solving = [e for e in pool.pool if u in pool.resolved[e]]
        oracle_pick = min(solving or pool.pool,
                          key=lambda e: (pool.costs[e][u], effort_key(e)))
        if pick == oracle_pick:
            n_agree += 1
        elif pool.costs[pick][u] < pool.costs[oracle_pick][u]:
            n_cheaper += 1
        else:
            n_dearer += 1
        lost += bool(solving) and not hit
        won += hit and not solving

    counts: dict[str, int] = {}
    for unit in units:
        how = votes.parsed_by[unit]
        counts[how] = counts.get(how, 0) + 1

    return SelfRouting(
        dataset=pool.dataset, model=pool.model, variant=selection.VARIANT,
        judge_effort=judge_effort, n_units=len(units), n_routable=len(pool.units), levels=levels,
        oracle_cost=oracle_cost, oracle_resolved=oracle_resolved, oracle_mix=oracle_mix,
        self_cost=self_cost, self_resolved=self_resolved, router_cost=router_cost,
        mix=mix, mix_solved=mix_solved, n_fallback=n_fallback, n_agree=n_agree,
        n_cheaper=n_cheaper, n_dearer=n_dearer, lost=lost, won=won, attempts=pool.attempts,
        n_unanswered=votes.unanswered, router_calls=len(units), parsed_by=counts,
        random_router=random_over(pool, units))


def route_dataset_self(spec: DatasetSpec
                       ) -> tuple[list[SelfRouting], list[ModelRouting], list[str], list[str]]:
    """Every model's self-routing and its oracle bound on one benchmark.

    Returns the self-routing results, the matching bounds, notes, and omitted models.
    """
    source = tasks.SOURCE_OF[spec.key]
    pools, notes, omitted = pools_for(spec)
    bounds, bound_notes, _ = routing.route_dataset(spec)
    by_model = {b.model: b for b in bounds}
    for note in bound_notes:
        if note not in notes:
            notes.append(note)

    out: list[SelfRouting] = []
    kept: list[ModelRouting] = []
    for model, pool in pools.items():
        if model not in by_model:
            notes.append(f"{spec.key}/{style.label_of(model)}: no oracle bound, skipped")
            omitted.append(model)
            continue
        crosscheck(pool, by_model[model])
        judged_at = selection.judge_effort_for(model)
        votes, choice_notes = choices_for(pool, source, judged_at)
        notes.extend(choice_notes)
        if not votes.choices:
            notes.append(f"{spec.key}/{style.label_of(model)}: no selections recorded at judge "
                         f"effort {judged_at}; run run_selection.py")
            omitted.append(model)
            continue
        out.append(evaluate(pool, votes, judge_effort=judged_at))
        kept.append(by_model[model])
    return out, kept, notes, omitted
