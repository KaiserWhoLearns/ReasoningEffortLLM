"""Load every benchmark's runs into one shape, so the figures do not depend on which they got.

The benchmarks store results differently: mini-swe-agent trajectories plus the official
evaluator's verdicts (SWE-Bench Pro), harbor trial metrics plus a LiteLLM capture
(Terminal-Bench), and a flat samples.jsonl of single API calls (AIME, GPQA). Each loader reduces
one (model, effort level) to a `Run`: how many units resolved, what the run cost, and what each
unit cost, took and consumed.

Two rules are applied identically to every benchmark:

**One rate basis.** Every dollar is tokens x `common/pricing.py`. Pro reads `cost_from_tokens`,
Terminal-Bench reads `our_cost_usd_no_cachewrite`, AIME and GPQA recompute from their samples.
On Terminal-Bench, harbor's trial metrics record cache reads but not cache writes, so Anthropic
rows there are a slight lower bound.

**One easy set.** Per-unit means (`rows_for`) are computed over the units the model resolves at
every reasoning effort level it ran, a within-model difficulty control; `n_easy` and a bootstrap
interval are reported beside every mean.

Runs that do not cover the benchmark's full unit set are dropped by `rows_for` and named in the
returned notes.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import style
from stats import boot_ci, wilson
from style import effort_key

from common import latency, models, paths

# An effort directory below this many units is a smoke test, not a measurement.
MIN_UNITS = 30

# Effort levels kept in the tables but left off the figures and out of the easy-set
# intersection. `none` switches reasoning off rather than being a cheaper rung, and a level that
# solves little would shrink the difficulty-controlled set to what it can do.
UNPLOTTED_EFFORTS = {"none"}

# Effort names that share another name's slot on an axis shared across models: `max` and `xhigh`
# are each vendor's top level (see `style.MERGED_RUNG_TICK`).
RUNG_ALIAS = {"max": "xhigh"}


def rung_of(effort: str) -> str:
    """The slot an effort level occupies on an axis shared across models."""
    return RUNG_ALIAS.get(effort, effort)


# --------------------------------------------------------------------------- the run shape

@dataclass
class Run:
    """One (model, effort level) run, in benchmark-independent terms."""
    model: str                       # canonical short slug (a style.MODELS key)
    effort: str
    n_units: int                     # denominator of the headline rate
    n_resolved: int
    accuracy: tuple[float, float, float]   # (pct, lo, hi): Wilson, or SE across units on AIME/GPQA
    total_cost: float                # the whole run, at table rates
    resolved_units: set[str]
    values: dict[str, dict[str, float]] = field(default_factory=dict)  # metric -> unit -> value
    attempts_per_unit: int = 1        # >1 on AIME and GPQA, which draw k attempts per unit
    # Seconds a caller waits for the output tokens at the model's table throughput
    # (common/latency.py), and the API bill plus that wait priced at the active wage. None when
    # the model has no throughput entry.
    total_time_s: float | None = None
    total_full_cost: float | None = None


# Time-derived per-unit metrics, computed for every benchmark (see `with_time`).
TIME_METRICS = ("output", "time", "wait", "full_cost", "latency")


def with_time(model: str, values: dict[str, dict[str, float]],
              output: dict[str, float]) -> dict[str, dict[str, float]]:
    """Add per-unit output tokens and the three time-derived metrics to a run's `values`.

    `output` is output tokens per unit on the same footing as `values["cost"]`, so `full_cost`
    adds like with like. Units missing a cost keep their time but get no full cost.
    """
    values["output"] = dict(output)
    times = {u: latency.decode_seconds(model, o) for u, o in output.items()}
    values["time"] = {u: s for u, s in times.items() if s is not None}
    values["wait"] = {u: latency.wait_cost_usd(s) for u, s in values["time"].items()}
    values["full_cost"] = {u: values["cost"][u] + w for u, w in values["wait"].items()
                           if u in values.get("cost", {})}
    return values


def run_time_totals(model: str, total_cost: float,
                    total_output_tokens: float) -> tuple[float | None, float | None]:
    """(total seconds, total full cost) for a run from its whole-run output-token count."""
    seconds = latency.decode_seconds(model, total_output_tokens)
    return seconds, latency.full_cost_usd(total_cost, seconds)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str                       # display name
    unit: str                        # "instance" | "task" | "problem" | "question"
    units: str
    scaffold: str                    # one clause for a caption
    metrics: tuple[str, ...]
    value_unit: str                  # what one per-unit VALUE is measured over
    panel_title: str                 # "{name}" is the metric noun
    accuracy_label: str
    accuracy_kind: str               # "wilson" | "cluster-se": what the interval means
    cost_axis: Callable[[list[dict]], str]
    """x-axis label for a cost-vs-accuracy plane, given the row dicts being plotted."""
    loader: Callable[[], tuple[list[Run], list[str]]]
    zoom_floor: float | None = None
    """Accuracy above which a combined figure adds a second, clipped panel."""


# --------------------------------------------------------------------------- derived rows

def rows_for(spec: DatasetSpec) -> tuple[list[dict], list[str], list[Run]]:
    """Load a dataset and reduce it to one flat row per (model, effort level).

    Rows carry the headline rate (full unit set) and the per-unit means with intervals (easy
    set). The raw per-unit samples ride along under `_samples`.
    """
    runs, notes = spec.loader()
    if not runs:
        return [], notes, []

    full_n = max(r.n_units for r in runs)
    for r in [r for r in runs if r.n_units != full_n]:
        notes.append(f"dropped {style.label_of(r.model)} {r.effort}: covered {r.n_units} of "
                     f"{full_n} {spec.units}, so neither its rate nor its spend is comparable")
    runs = [r for r in runs if r.n_units == full_n]

    by_model: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_model[r.model].append(r)

    rows: list[dict] = []
    for model, model_runs in by_model.items():
        model_runs.sort(key=lambda r: effort_key(r.effort))
        # The easy set is the intersection over the plotted levels (see UNPLOTTED_EFFORTS).
        ladder = [r for r in model_runs if r.effort not in UNPLOTTED_EFFORTS]
        easy = set.intersection(*[r.resolved_units for r in ladder]) if ladder else set()
        if ladder and not easy:
            notes.append(f"{style.label_of(model)}: no {spec.unit} is resolved at all "
                         f"{len(ladder)} plotted effort levels — per-{spec.unit} means omitted")
        if not ladder:
            notes.append(f"{style.label_of(model)}: every level it ran is unplotted "
                         f"({', '.join(sorted(UNPLOTTED_EFFORTS))}) — absent from the figures")
        for run in model_runs:
            pct, lo, hi = run.accuracy
            row = {
                "dataset": spec.key, "model": model, "label": style.label_of(model),
                "effort": run.effort,
                "plotted": run.effort not in UNPLOTTED_EFFORTS,
                "n": run.n_units, "resolved": run.n_resolved,
                "attempts": run.attempts_per_unit,
                "resolve_pct": round(pct, 1), "ci_lo": round(lo, 1), "ci_hi": round(hi, 1),
                "total_cost": round(run.total_cost, 3),
                "cost_per_unit": round(run.total_cost / run.n_units, 4),
                "total_time_s": (None if run.total_time_s is None
                                 else round(run.total_time_s, 1)),
                "total_full_cost": (None if run.total_full_cost is None
                                    else round(run.total_full_cost, 3)),
                "full_cost_per_unit": (None if run.total_full_cost is None
                                       else round(run.total_full_cost / run.n_units, 4)),
                # Per value unit: per attempt on AIME/GPQA, per unit elsewhere.
                "full_cost_per_value_unit": (
                    None if run.total_full_cost is None
                    else run.total_full_cost / run.n_units / run.attempts_per_unit),
                "n_easy": len(easy),
                "wage": latency.WAGE.key,
                "wage_usd_per_hour": latency.WAGE.usd_per_hour,
                "_samples": {},
            }
            # Every metric with per-unit values gets an easy-set mean.
            for metric in list(spec.metrics) + [m for m in TIME_METRICS
                                                 if run.values.get(m)]:
                vals = [run.values.get(metric, {}).get(u) for u in sorted(easy)]
                mean, mlo, mhi = boot_ci(vals)
                row[f"{metric}_easy"] = None if mean is None else round(mean, 6)
                row[f"{metric}_easy_lo"] = None if mlo is None else round(mlo, 6)
                row[f"{metric}_easy_hi"] = None if mhi is None else round(mhi, 6)
                row["_samples"][metric] = [v for v in vals if v is not None]
            # Reasoning tokens are a subset of the token column, carried for context only: they
            # are not reported for every provider.
            r_mean, _, _ = boot_ci([run.values.get("reasoning", {}).get(u) for u in sorted(easy)])
            row["reasoning_easy"] = None if r_mean is None else round(r_mean, 1)
            rows.append(row)

    rows.sort(key=lambda r: (style.model_sort_key(r["model"]), effort_key(r["effort"])))
    return rows, notes, runs


# --------------------------------------------------------------------------- SWE-Bench Pro

# Per-instance budget, matching `swebench_pro/report.py --cost-cap`: an instance the evaluator
# passed but which spent more than this does not count as solved.
PRO_COST_CAP = 5.0


def _pro_efforts(model_dir: Path) -> list[str]:
    """Effort dirs with both a cost summary and a scored eval, in ladder order."""
    out = []
    for d in model_dir.iterdir():
        if not d.is_dir() or d.name not in models.EFFORT_ORDER:
            continue
        if not (d / "cost_summary.json").exists():
            continue
        if not (model_dir / "eval" / d.name / "eval_results.json").exists():
            continue
        out.append(d.name)
    return sorted(out, key=effort_key)


def _pro_cost_rows(effort_dir: Path) -> dict[str, dict]:
    """{instance_id: per-instance cost row} from a level's cost summary."""
    rows: dict[str, dict] = {}
    for row in json.loads((effort_dir / "cost_summary.json").read_text()).get("per_instance", []):
        rows.setdefault(row["instance_id"], row)
    return rows


def _pilot_ids() -> set[str]:
    """The 44 pilot instance ids, from the subset file the runs were launched with."""
    path = paths.REPO_ROOT / "swebench_pro" / "subsets" / "pilot.txt"
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_pro(benchmark: str = "swebench-pro", subset: set[str] | None = None,
             cost_cap: float = PRO_COST_CAP) -> tuple[list[Run], list[str]]:
    """Load a SWE-Bench Pro result tree (`swebench-pro` or `swebench-pro-corrected`).

    `subset` restricts every run to a fixed set of instance ids (the paper uses the 44-instance
    pilot, `_pilot_ids()`). `cost_cap` applies the per-instance budget rule; pass 0 to score
    straight off the evaluator's verdicts.
    """
    root = paths.data_root(benchmark)
    runs, notes = [], []
    over_budget_total = 0
    for model_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        model = model_dir.name
        efforts = _pro_efforts(model_dir)
        if not efforts:
            notes.append(f"skipped {model}: no effort dir has both cost_summary.json and a "
                         f"scored eval/<effort>/eval_results.json")
            continue
        for effort in efforts:
            verdicts = json.loads(
                (model_dir / "eval" / effort / "eval_results.json").read_text())
            per = _pro_cost_rows(model_dir / effort)
            if subset is not None:
                verdicts = {i: v for i, v in verdicts.items() if i in subset}
                per = {i: r for i, r in per.items() if i in subset}
                if not verdicts:
                    continue
            # Budget rule: an instance that spent more than the cap is not counted as solved.
            over = set()
            if cost_cap:
                over = {i for i, v in verdicts.items() if v
                        and (per.get(i, {}).get("cost_from_tokens") or 0) > cost_cap}
                over_budget_total += len(over)
            n = len(verdicts)
            k = sum(1 for i, v in verdicts.items() if v and i not in over)
            if n < MIN_UNITS:
                notes.append(f"skipped {model} {effort}: only {n} instances scored")
                continue
            values = {
                "cost": {i: r["cost_from_tokens"] for i, r in per.items()
                         if r.get("cost_from_tokens") is not None},
                "turns": {i: r["api_calls"] for i, r in per.items()
                          if r.get("api_calls") is not None},
                "tokens": {i: r["total_tokens"] for i, r in per.items()
                           if r.get("total_tokens") is not None},
                "reasoning": {i: r["reasoning_tokens"] for i, r in per.items()
                              if r.get("reasoning_available")},
            }
            with_time(model, values, {i: r["output_tokens"] for i, r in per.items()
                                      if r.get("output_tokens") is not None})
            # Summed over the instances actually scored, not the file's own aggregate.
            total_cost = sum(values["cost"].get(i, 0.0) for i in verdicts)
            total_time_s, total_full_cost = run_time_totals(
                model, total_cost, sum(values["output"].get(i, 0.0) for i in verdicts))
            runs.append(Run(
                model=model, effort=effort, n_units=n, n_resolved=k,
                accuracy=wilson(k, n),
                total_cost=total_cost,
                resolved_units={i for i, v in verdicts.items() if v and i not in over},
                values=values, total_time_s=total_time_s, total_full_cost=total_full_cost))
    if over_budget_total:
        notes.append(f"budget rule: {over_budget_total} evaluator-passed instance(s) cost more "
                     f"than ${cost_cap:.0f} and are not counted as solved")
    return runs, notes


# --------------------------------------------------------------------------- Terminal-Bench

def _tb_call_counts(run_dir: Path) -> dict[str, int]:
    """{trial_name: LM calls} from each trial's ATIF trajectory (steps sourced by the agent).

    Trials sit at `<effort>/<job>/<trial>/agent/trajectory.json`.
    """
    counts = {}
    for traj in run_dir.glob("*/*/agent/trajectory.json"):
        trial = traj.parent.parent.name
        try:
            steps = json.loads(traj.read_text())["steps"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        counts[trial] = sum(1 for s in steps if s.get("source") == "agent")
    return counts


def _tb_one_trial_per_task(per: list[dict]) -> tuple[list[dict], str | None]:
    """One trial per task where a run directory holds re-launched trials beside the original.

    A trial killed by a provider outage has no verdict (`resolved` is None) and is re-launched
    into the same effort directory, so `cost_summary.json` lists that task more than once. Per
    task the trial with a verdict wins; if none (or several) has one, the one that generated the
    most output. Directories with one trial per task are returned unchanged.
    """
    by_task: dict[str, list[dict]] = {}
    for r in per:
        by_task.setdefault(r["task_name"], []).append(r)
    if all(len(v) == 1 for v in by_task.values()):
        return per, None
    kept = []
    for rows in by_task.values():
        rows = sorted(rows, key=lambda r: (r["resolved"] is not None,
                                           r.get("n_output_tokens") or 0), reverse=True)
        kept.append(rows[0])
    return kept, (f"collapsed {len(per)} trials to {len(kept)} tasks, keeping per task the "
                  f"trial with a verdict, else the one with the most output")


def load_tb() -> tuple[list[Run], list[str]]:
    root = paths.data_root("terminal-bench-2-1")
    # Run dirs are named by the provider-prefixed litellm id; map back to the short slug.
    to_slug = {models.litellm_slug(s): s for s in models.MODELS}
    runs, notes = [], []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = to_slug.get(model_dir.name, model_dir.name)
        efforts = sorted((d.name for d in model_dir.iterdir()
                          if d.is_dir() and d.name in models.EFFORT_ORDER
                          and (d / "cost_summary.json").exists()), key=effort_key)
        missing = sorted(d.name for d in model_dir.iterdir()
                         if d.is_dir() and d.name in models.EFFORT_ORDER
                         and not (d / "cost_summary.json").exists())
        if missing:
            notes.append(f"skipped {style.label_of(model)} {missing}: no cost_summary.json — "
                         f"run terminal_bench/collect_cost.py")
        for effort in efforts:
            run_dir = model_dir / effort
            summary = json.loads((run_dir / "cost_summary.json").read_text())
            per = summary["per_instance"]
            total_cost = summary["aggregate"]["total_cost_from_table_usd"]
            per, collapse_note = _tb_one_trial_per_task(per)
            if collapse_note:
                notes.append(f"{style.label_of(model)} {effort}: {collapse_note}")
                total_cost = sum(r.get("our_cost_usd_no_cachewrite") or 0.0 for r in per)
            if len(per) < MIN_UNITS:
                notes.append(f"skipped {style.label_of(model)} {effort}: only {len(per)} trials")
                continue
            calls = _tb_call_counts(run_dir)
            n = len(per)
            k = sum(1 for r in per if r["resolved"] is True)
            values: dict[str, dict[str, float]] = {"cost": {}, "turns": {}, "tokens": {},
                                                   "reasoning": {}}
            for r in per:
                task = r["task_name"]
                if r.get("our_cost_usd_no_cachewrite") is not None:
                    values["cost"][task] = r["our_cost_usd_no_cachewrite"]
                if r["trial_name"] in calls:
                    values["turns"][task] = calls[r["trial_name"]]
                values["tokens"][task] = ((r.get("n_input_tokens") or 0)
                                         + (r.get("n_output_tokens") or 0))
                if r.get("reasoning_tokens") is not None:
                    values["reasoning"][task] = r["reasoning_tokens"]
            with_time(model, values, {r["task_name"]: r["n_output_tokens"] for r in per
                                      if r.get("n_output_tokens") is not None})
            total_time_s, total_full_cost = run_time_totals(
                model, total_cost, sum(values["output"].values()))
            runs.append(Run(
                model=model, effort=effort, n_units=n, n_resolved=k,
                accuracy=wilson(k, n),
                total_cost=total_cost,
                # A trial with no verifier verdict counts as unresolved.
                resolved_units={r["task_name"] for r in per if r["resolved"] is True},
                values=values, total_time_s=total_time_s, total_full_cost=total_full_cost))
    return runs, notes


# --------------------------------------------------------------------------- AIME and GPQA

def _import_benchmark(directory: str, module: str):
    """Import a benchmark module, e.g. `aime_2026/aime.py`."""
    path = str(paths.REPO_ROOT / directory)
    if path not in sys.path:
        sys.path.insert(0, path)
    return __import__(module)


def _load_single_call(benchmark: str, mod, idx_key: str, unit_noun: str,
                      refusals: Callable[[dict], bool] | None = None
                      ) -> tuple[list[Run], list[str]]:
    """AIME / GPQA: the unit is a problem, per-unit values are means over its attempts.

    "Resolved" is a unit the model gets right on at least one attempt. The accuracy is the mean
    per-attempt correctness with a normal interval from the standard error across units:
    attempts on the same unit are not independent, so a per-attempt binomial CI would be too
    narrow.
    """
    data_root = paths.data_root(benchmark)
    runs, notes = [], []
    for model in style.MODELS:
        for effort in sorted(models.EFFORT_ORDER, key=effort_key):
            samples = mod.read_samples(mod.run_dir(data_root, model, effort) / mod.SAMPLES_FILE)
            if not samples:
                continue
            correct: dict[int, list[bool]] = defaultdict(list)
            cost: dict[int, float] = defaultdict(float)
            tokens: dict[int, float] = defaultdict(float)
            reasoning: dict[int, float] = defaultdict(float)
            output: dict[int, float] = defaultdict(float)
            measured: dict[int, float] = defaultdict(float)   # wall clock, TTFT included
            refused: list[int] = []
            for s in samples:
                p, usage = s[idx_key], s["usage"]
                correct[p].append(bool(s["correct"]))
                if refusals is not None and refusals(s):
                    refused.append(p)
                cost[p] += mod.cost_of(model, usage) or 0.0
                tokens[p] += (usage["input_tokens"] or 0) + (usage["output_tokens"] or 0)
                reasoning[p] += usage["reasoning_tokens"] or 0
                output[p] += usage["output_tokens"] or 0
                measured[p] += s.get("latency_s") or 0.0
            if refused:
                # Refusals score as wrong; `gpqa_diamond/report.py` prints the other readings.
                notes.append(
                    f"{style.label_of(model)} {effort}: {len(refused)} attempt(s) across "
                    f"{len(set(refused))} {unit_noun}(s) were refused by the provider and score "
                    f"as wrong, capping this level at "
                    f"{(1 - len(refused) / len(samples)) * 100:.1f}%")
            if len(correct) < MIN_UNITS:
                notes.append(f"skipped {style.label_of(model)} {effort}: only {len(correct)} "
                             f"{unit_noun}s")
                continue
            rates = [sum(v) / len(v) for v in correct.values()]
            mean = statistics.mean(rates)
            se = statistics.stdev(rates) / len(rates) ** 0.5 if len(rates) > 1 else 0.0
            values = with_time(model, {
                "cost": {str(p): cost[p] / len(correct[p]) for p in correct},
                "tokens": {str(p): tokens[p] / len(correct[p]) for p in correct},
                "reasoning": {str(p): reasoning[p] / len(correct[p]) for p in correct},
                "latency": {str(p): measured[p] / len(correct[p]) for p in correct},
            }, {str(p): output[p] / len(correct[p]) for p in correct})
            total_cost = sum(cost.values())
            total_time_s, total_full_cost = run_time_totals(model, total_cost,
                                                            sum(output.values()))
            runs.append(Run(
                model=model, effort=effort, n_units=len(correct),
                n_resolved=sum(1 for v in correct.values() if any(v)),
                accuracy=(100 * mean, 100 * max(mean - 1.96 * se, 0.0),
                          100 * min(mean + 1.96 * se, 1.0)),
                total_cost=total_cost,
                resolved_units={str(p) for p, v in correct.items() if any(v)},
                attempts_per_unit=round(len(samples) / len(correct)),
                values=values, total_time_s=total_time_s, total_full_cost=total_full_cost))
    return runs, notes


def load_aime() -> tuple[list[Run], list[str]]:
    return _load_single_call("aime-2026", _import_benchmark("aime_2026", "aime"),
                             "problem_idx", "problem")


def load_gpqa() -> tuple[list[Run], list[str]]:
    gpqa = _import_benchmark("gpqa_diamond", "gpqa")
    return _load_single_call("gpqa-diamond", gpqa, "question_idx", "question", gpqa.is_refusal)


# --------------------------------------------------------------------------- registry

DATASETS: dict[str, DatasetSpec] = {
    "swebench_pro_pilot": DatasetSpec(
        key="swebench_pro_pilot", title="SWE-Bench-Pro (pilot)", unit="instance",
        units="instances",
        scaffold="mini-swe-agent scaffold, k=1, 44-instance pilot subset",
        metrics=("cost", "turns", "tokens"),
        value_unit="instance", panel_title="{name} per easy instance",
        accuracy_label="Instances resolved  (%)", accuracy_kind="wilson",
        cost_axis=lambda rows: f"Pilot cost — {max(r['n'] for r in rows)} instances, k=1  (USD)",
        loader=lambda: load_pro(subset=_pilot_ids())),
    "terminal_bench": DatasetSpec(
        key="terminal_bench", title="Terminal-Bench 2.1", unit="task", units="tasks",
        scaffold="harbor Terminus-2, k=1",
        metrics=("cost", "turns", "tokens"),
        value_unit="task", panel_title="{name} per easy task",
        accuracy_label="Tasks resolved  (%)", accuracy_kind="wilson",
        cost_axis=lambda rows: f"Full-run cost — {max(r['n'] for r in rows)} tasks, k=1  (USD)",
        loader=load_tb),
    "aime_2026": DatasetSpec(
        key="aime_2026", title="AIME 2026", unit="problem", units="problems",
        scaffold="one API call per attempt, no tools",
        metrics=("cost", "tokens"),
        # A per-unit value is a mean over the problem's attempts, so the axis says "attempt"
        # while the easy set is counted in problems, the clustering unit for the interval.
        value_unit="attempt", panel_title="{name} per attempt on an easy problem",
        accuracy_label="pass@1  (%)", accuracy_kind="cluster-se",
        cost_axis=lambda rows: f"Effort-level cost — {max(r['n'] for r in rows)} problems x "
                               f"{max(r['attempts'] for r in rows)} attempts  (USD)",
        loader=load_aime, zoom_floor=88.0),
    "gpqa_diamond": DatasetSpec(
        key="gpqa_diamond", title="GPQA Diamond", unit="question", units="questions",
        scaffold="one API call per attempt, no tools",
        metrics=("cost", "tokens"),
        value_unit="attempt", panel_title="{name} per attempt on an easy question",
        accuracy_label="pass@1  (%)", accuracy_kind="cluster-se",
        cost_axis=lambda rows: f"Effort-level cost — {max(r['n'] for r in rows)} questions x "
                               f"{max(r['attempts'] for r in rows)} attempts  (USD)",
        loader=load_gpqa),
}
