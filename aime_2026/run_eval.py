#!/usr/bin/env python3
"""Generate and grade AIME 2026 attempts, one API call per attempt.

    python aime_2026/run_eval.py --model claude-sonnet-5 --full
    python aime_2026/run_eval.py --model gpt-5.6-terra --efforts medium,xhigh --limit 5
    python aime_2026/run_eval.py --model gpt-5.6-terra --full --dry-run

One arm is a (model, effort) pair; one sample is a (problem, run) attempt. Each sample is
appended to `<data>/aime_2026/<model>/<effort>/samples.jsonl` as it lands, so re-running the
same command resumes an interrupted arm (`--redo-all` starts it over). report.py re-grades
from the stored response text.

Defaults are the paper's setup: every effort level the model distinguishes, 10 attempts per
problem, a 128,000-token output cap, no sampling parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, for `common`
import aime  # noqa: E402
from common import models, paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help=f"one of {sorted(models.MODELS)}")
    p.add_argument("--efforts",
                   help="comma-separated reasoning-effort levels (default: all the model has)")
    p.add_argument("--n", type=int, default=10, help="attempts per problem (default: 10)")
    p.add_argument("--max-tokens", type=int, default=128_000,
                   help="output cap per attempt (default: %(default)s)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="attempts in flight (default: %(default)s)")
    p.add_argument("--out-root", default=str(aime.DEFAULT_OUT_ROOT),
                   help="results tree (default: %(default)s)")
    p.add_argument("--full", action="store_true", help=f"run all {aime.N_PROBLEMS} problems")
    p.add_argument("--limit", type=int, help="run the first N problems only")
    p.add_argument("--problems", help="comma-separated 1-based problem_idx list")
    p.add_argument("--redo-all", action="store_true", help="discard existing samples for the arm")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and an assumed-token cost estimate; make no calls")
    p.add_argument("--assume-output-tokens", type=int, default=12_000,
                   help="output tokens per attempt assumed by --dry-run (default: %(default)s)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    paths.load_env()          # before the dataset load, so HF_TOKEN is in place for it
    spec = models.resolve(args.model)
    efforts = ([e.strip() for e in args.efforts.split(",") if e.strip()] if args.efforts
               else models.valid_efforts(args.model))
    models.check_efforts(args.model, efforts)

    if not (args.full or args.limit or args.problems):
        raise SystemExit("pass --full, or --limit N, or --problems 1,2,3")
    problem_ids = [int(x) for x in args.problems.split(",")] if args.problems else None
    problems = aime.load_problems(limit=args.limit, problems=problem_ids)
    out_root = Path(args.out_root)

    print(f"model {spec.slug} ({spec.family})  efforts {efforts}  "
          f"problems {len(problems)}  runs/problem {args.n}  "
          f"=> {len(problems) * args.n} attempts per arm")

    if args.dry_run:
        return dry_run(args, spec, efforts, problems)

    client = aime.make_client(spec)
    for effort in efforts:
        run_arm(args, spec, effort, problems, client, out_root)
    return 0


def dry_run(args, spec, efforts, problems) -> int:
    """Estimate the sweep's cost from an assumed output length."""
    rate = aime.PRICES.get(spec.slug)
    prompt_chars = sum(len(aime.build_prompt(p["problem"])) for p in problems)
    in_tokens = int(prompt_chars / 4) * args.n          # ~4 chars/token, per arm
    out_tokens = len(problems) * args.n * args.assume_output_tokens
    print(f"\nassumed {args.assume_output_tokens} output tokens/attempt "
          f"(thinking included); input estimated from prompt length")
    if rate is None:
        print(f"no rates for {spec.slug} in aime.PRICES")
        return 0
    per_arm = in_tokens * rate["input"] + out_tokens * rate["output"]
    for effort in efforts:
        print(f"  {effort:<7} ~{in_tokens/1e3:.1f}k in + {out_tokens/1e6:.2f}M out  "
              f"~${per_arm:,.2f}")
    print(f"  {'total':<7} {'':>30}~${per_arm * len(efforts):,.2f}")
    return 0


def run_arm(args, spec, effort: str, problems: list[dict], client, out_root: Path) -> None:
    """Run every missing (problem, run) attempt for one (model, effort) arm."""
    directory = aime.run_dir(out_root, args.model, effort)
    samples_path = directory / aime.SAMPLES_FILE
    directory.mkdir(parents=True, exist_ok=True)

    if args.redo_all and samples_path.exists():
        samples_path.unlink()
    done = {(s["problem_idx"], s["run_idx"]) for s in aime.read_samples(samples_path)}
    todo = [(p, run_idx) for p in problems for run_idx in range(args.n)
            if (p["problem_idx"], run_idx) not in done]

    (directory / aime.META_FILE).write_text(json.dumps({
        "dataset": aime.HF_DATASET,
        "model": spec.slug, "api_id": aime.provider_id(spec), "family": spec.family,
        "effort": effort, "runs_per_problem": args.n, "max_tokens": args.max_tokens,
        "thinking_display": "omitted" if spec.family == "anthropic" else None,
        "instruction": aime.INSTRUCTION,
        "sampling_params": "none",
        "n_problems": len(problems),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")

    print(f"\n[{effort}] {len(done)} sample(s) already present, {len(todo)} to run "
          f"-> {samples_path}")
    if not todo:
        return

    lock = Lock()
    counts = {"done": 0, "correct": 0, "failed": 0, "truncated": 0, "cost": 0.0}
    started = time.monotonic()

    def one(problem: dict, run_idx: int) -> dict:
        prompt = aime.build_prompt(problem["problem"])
        began = time.monotonic()

        def warn(attempt: int, exc: Exception) -> None:
            with lock:
                print(f"  retry {attempt} for problem {problem['problem_idx']} "
                      f"run {run_idx}: {type(exc).__name__}: {exc}")

        reply = aime.with_retries(
            lambda: aime.call(client, spec, prompt, effort, args.max_tokens), on_error=warn)
        answer, how = aime.extract_answer(reply.text)
        gold = int(problem["answer"])
        return {
            "model": spec.slug, "effort": effort,
            "problem_idx": problem["problem_idx"], "run_idx": run_idx,
            "gold": gold, "answer": answer, "correct": answer == gold,
            "extracted_by": how, "truncated": reply.truncated, "finish": reply.finish,
            "usage": reply.usage, "cost_usd": aime.cost_of(args.model, reply.usage),
            "latency_s": round(time.monotonic() - began, 1),
            "response": reply.text,
            "reasoning_summary": reply.reasoning_text,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, p, r): (p["problem_idx"], r) for p, r in todo}
        for future in as_completed(futures):
            problem_idx, run_idx = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                with lock:
                    counts["failed"] += 1
                    print(f"  FAILED problem {problem_idx} run {run_idx} after retries: "
                          f"{type(exc).__name__}: {exc}")
                continue
            with lock:
                aime.append_sample(samples_path, record)
                counts["done"] += 1
                counts["correct"] += bool(record["correct"])
                counts["truncated"] += bool(record["truncated"])
                counts["cost"] += record["cost_usd"] or 0.0
                print(f"  [{counts['done']}/{len(todo)}] p{problem_idx:02d} r{run_idx} "
                      f"{'OK ' if record['correct'] else 'x  '}"
                      f"answer={record['answer']} gold={record['gold']} "
                      f"out={record['usage'].get('output_tokens')} "
                      f"{record['latency_s']}s  running acc "
                      f"{counts['correct']}/{counts['done']} "
                      f"${counts['cost']:.2f}")

    minutes = (time.monotonic() - started) / 60
    print(f"[{effort}] {counts['done']} new sample(s) in {minutes:.1f} min, "
          f"{counts['failed']} failed, {counts['truncated']} hit the output cap, "
          f"${counts['cost']:.2f} spent (this invocation)")


if __name__ == "__main__":
    raise SystemExit(main())
