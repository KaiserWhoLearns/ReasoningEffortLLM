#!/usr/bin/env python3
"""Ask every model, for every instance, which reasoning effort it wants (the self-router).

    python routing/run_selection.py --dry-run                       # one prompt + a cost estimate
    python routing/run_selection.py --dataset gpqa_diamond --limit 5
    python routing/run_selection.py --dataset terminal_bench --all

One API call per (model, instance), at the judge effort (`low`, or the cheapest level a model
has). The menu is the model's routing pool on that benchmark, read from the sweeps on disk, so
the self-router and the oracle choose between the same levels. Selections are appended to

    DATA_ROOT/self_routing/<source>/<model>.jsonl

and scored by make_self_routing.py. A selection already on disk is not asked again, so an
interrupted run resumes; `--refresh` re-asks. The full plan runs only with `--limit N` or `--all`.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE.parent / "figures", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import selection  # noqa: E402
import style  # noqa: E402
import tasks  # noqa: E402
import vanilla  # noqa: E402
from common import pricing  # noqa: E402
from datasets import DATASETS  # noqa: E402

DATASET_KEYS = [k for k in DATASETS if k in tasks.SOURCE_OF]

# Assumed output tokens per call and characters per prompt token, for the estimate only.
EST_OUTPUT_TOKENS = 300
CHARS_PER_TOKEN = 3.6


@dataclass
class Job:
    source: str
    model: str
    unit: str
    menu: list[str]
    to_level: dict[str, str]
    task: str


def plan(datasets: list[str], wanted_models: list[str] | None,
         refresh: bool) -> tuple[list[Job], list[str]]:
    """Every selection that is missing, deduplicated across datasets that share a source."""
    notes: list[str] = []
    wanted: dict[tuple[str, str, tuple[str, ...]], set[str]] = {}
    menus: dict[tuple[str, str, tuple[str, ...]], dict[str, str]] = {}

    for key in datasets:
        source = tasks.SOURCE_OF[key]
        pools, pool_notes, _ = vanilla.pools_for(DATASETS[key])
        notes.extend(pool_notes)
        for model, pool in pools.items():
            if wanted_models and model not in wanted_models:
                continue
            menu, to_level = selection.build_menu(pool.pool)
            slot = (source, model, tuple(menu))
            wanted.setdefault(slot, set()).update(pool.units)
            menus[slot] = to_level

    jobs: list[Job] = []
    for (source, model, menu), units in sorted(wanted.items()):
        text = tasks.tasks_for(source)
        have: set[str] = set()
        if not refresh:
            judge = selection.judge_effort_for(model)
            have = {r["unit"] for r in selection.read_records(selection.record_path(source, model))
                    if selection.is_scored(r, list(menu), judge)}
        missing_text = sorted(u for u in units if u not in text)
        if missing_text:
            notes.append(f"{source}/{style.label_of(model)}: no task text for "
                         f"{len(missing_text)} unit(s), not asked (e.g. {missing_text[0]})")
        for unit in sorted(units):
            if unit in text and unit not in have:
                jobs.append(Job(source=source, model=model, unit=unit, menu=list(menu),
                                to_level=menus[(source, model, menu)], task=text[unit]))
    return jobs, notes


def prompt_for(job: Job, max_task_chars: int) -> tuple[str, bool]:
    text, cut = tasks.clip(job.task, max_task_chars)
    src = tasks.SOURCES[job.source]
    return selection.build_prompt(src.setting, src.unit, text, job.menu), cut


def estimate(jobs: list[Job], max_task_chars: int) -> float:
    """A priced guess at the plan, from the prompts it would send."""
    total = 0.0
    for job in jobs:
        prompt, _ = prompt_for(job, max_task_chars)
        cost = pricing.cost_from_usage(job.model,
                                       prompt_tokens=int(len(prompt) / CHARS_PER_TOKEN),
                                       output_tokens=EST_OUTPUT_TOKENS)
        total += cost or 0.0
    return total


def run(jobs: list[Job], *, max_task_chars: int, max_tokens: int, workers: int) -> None:
    by_model: dict[str, list[Job]] = {}
    for job in jobs:
        by_model.setdefault(job.model, []).append(job)

    spent = 0.0
    for model, model_jobs in by_model.items():
        judge = selection.judge_effort_for(model)
        client = selection.make_client(model)
        done = failed = 0
        started = time.time()

        def one(job: Job) -> dict | None:
            src = tasks.SOURCES[job.source]
            try:
                record = selection.select_one(
                    client, model=model, source=job.source, unit=job.unit, task=job.task,
                    unit_noun=src.unit, setting=src.setting, menu=job.menu,
                    to_level=job.to_level, judge_effort=judge, max_task_chars=max_task_chars,
                    max_tokens=max_tokens)
            except Exception as exc:                      # provider-agnostic
                print(f"  ! {style.label_of(model)} {job.unit}: {type(exc).__name__}: "
                      f"{str(exc)[:200]}")
                selection.append_record(selection.error_path(job.source, model),
                                        selection.error_record(
                                            model=model, source=job.source, unit=job.unit,
                                            menu=job.menu, judge_effort=judge, exc=exc))
                return None
            selection.append_record(selection.record_path(job.source, model), record)
            return record

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, job) for job in model_jobs]
            for future in as_completed(futures):
                record = future.result()
                if record is None:
                    failed += 1
                    continue
                done += 1
                spent += record["cost_usd"] or 0.0
                if done % 25 == 0 or done == len(model_jobs):
                    print(f"  {style.label_of(model)}: {done}/{len(model_jobs)} "
                          f"({failed} failed)  ${spent:.3f} spent  "
                          f"{time.time() - started:.0f}s", flush=True)
        for source in sorted({job.source for job in model_jobs}):
            print(f"  {style.label_of(model)}: wrote {selection.record_path(source, model)}")
    print(f"\nspent ${spent:.3f} on selection calls")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=DATASET_KEYS,
                    help="benchmark to ask about (repeatable; default: all)")
    ap.add_argument("--model", action="append",
                    help="model slug (repeatable; default: every model with a routing pool)")
    ap.add_argument("--max-task-chars", type=int, default=tasks.DEFAULT_MAX_CHARS,
                    help="clip the task text shown to the router (0 = never clip)")
    ap.add_argument("--max-tokens", type=int, default=selection.DEFAULT_MAX_TOKENS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="ask about at most this many instances per model")
    ap.add_argument("--all", action="store_true",
                    help="ask about every instance (required without --limit)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-ask instances that already have a selection on disk")
    ap.add_argument("--dry-run", action="store_true",
                    help="print one full prompt and the priced plan, then stop")
    args = ap.parse_args()

    jobs, notes = plan(args.dataset or DATASET_KEYS, args.model, args.refresh)
    for note in notes:
        print(f"  note: {note}")
    if not jobs:
        print("nothing to ask: every selection in this plan is on disk (--refresh re-asks)")
        return

    if args.limit:
        capped: list[Job] = []
        seen: dict[str, int] = {}
        for job in jobs:
            if seen.get(job.model, 0) < args.limit:
                seen[job.model] = seen.get(job.model, 0) + 1
                capped.append(job)
        jobs = capped

    per_model: dict[str, int] = {}
    for job in jobs:
        per_model[job.model] = per_model.get(job.model, 0) + 1
    print(f"\n{len(jobs)} selection call(s):")
    for model, n in sorted(per_model.items(), key=lambda kv: style.model_sort_key(kv[0])):
        print(f"  {style.label_of(model):<20} {n}  (judge effort "
              f"{selection.judge_effort_for(model)})")
    print(f"estimated ${estimate(jobs, args.max_task_chars):.2f} "
          f"(assuming {EST_OUTPUT_TOKENS} output tokens per call)")

    if args.dry_run:
        job = jobs[0]
        prompt, cut = prompt_for(job, args.max_task_chars)
        print(f"\n--- example prompt: {style.label_of(job.model)} / {job.source} / {job.unit}"
              f"{' (task text clipped)' if cut else ''} ---\n{prompt}\n--- end ---")
        return
    if not args.limit and not args.all:
        raise SystemExit("refusing to run the full plan implicitly; pass --limit N or --all")

    run(jobs, max_task_chars=args.max_task_chars, max_tokens=args.max_tokens,
        workers=args.workers)


if __name__ == "__main__":
    main()
