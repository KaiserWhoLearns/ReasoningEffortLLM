#!/usr/bin/env python3
"""Generate and grade GPQA Diamond attempts, one API call per attempt.

    python gpqa_diamond/run_eval.py --model claude-sonnet-5 --full
    python gpqa_diamond/run_eval.py --model gpt-5.6-terra --efforts medium,xhigh --limit 10
    python gpqa_diamond/run_eval.py --model gpt-5.6-terra --full --dry-run

One arm is a (model, effort) pair; one sample is a (question, run) attempt. Each sample is
appended to `<data>/gpqa_diamond/<model>/<effort>/samples.jsonl` as it lands, so re-running
the same command resumes an interrupted arm (`--redo-all` starts it over). report.py
re-grades from the stored response text.

Defaults are the paper's setup: every effort level the model distinguishes, 4 attempts per
question with independently shuffled choices, a 128,000-token output cap, no sampling
parameters.
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
import gpqa  # noqa: E402
from common import models, paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help=f"one of {sorted(models.MODELS)}")
    p.add_argument("--efforts",
                   help="comma-separated reasoning-effort levels (default: all the model has)")
    p.add_argument("--n", type=int, default=4, help="attempts per question (default: 4)")
    p.add_argument("--max-tokens", type=int, default=128_000,
                   help="output cap per attempt (default: %(default)s)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="attempts in flight (default: %(default)s)")
    p.add_argument("--out-root", default=str(gpqa.DEFAULT_OUT_ROOT),
                   help="results tree (default: %(default)s)")
    p.add_argument("--full", action="store_true", help=f"run all {gpqa.N_QUESTIONS} questions")
    p.add_argument("--limit", type=int, help="run the first N questions only")
    p.add_argument("--questions", help="comma-separated 1-based question_idx list")
    p.add_argument("--redo-all", action="store_true", help="discard existing samples for the arm")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and an assumed-token cost estimate; make no calls")
    p.add_argument("--assume-output-tokens", type=int, default=4_000,
                   help="output tokens per attempt assumed by --dry-run (default: %(default)s)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    paths.load_env()
    spec = models.resolve(args.model)
    efforts = ([e.strip() for e in args.efforts.split(",") if e.strip()] if args.efforts
               else models.valid_efforts(args.model))
    models.check_efforts(args.model, efforts)

    if not (args.full or args.limit or args.questions):
        raise SystemExit("pass --full, or --limit N, or --questions 1,2,3")
    question_ids = [int(x) for x in args.questions.split(",")] if args.questions else None
    questions = gpqa.load_questions(limit=args.limit, questions=question_ids)
    out_root = Path(args.out_root)

    print(f"model {spec.slug} ({spec.family})  efforts {efforts}  "
          f"questions {len(questions)}  runs/question {args.n}  "
          f"=> {len(questions) * args.n} attempts per arm")

    if args.dry_run:
        return dry_run(args, spec, efforts, questions)

    client = gpqa.make_client(spec)
    permutations = gpqa.permutation_table(args.n)
    for effort in efforts:
        run_arm(args, spec, effort, questions, permutations, client, out_root)
    return 0


def dry_run(args, spec, efforts, questions) -> int:
    """Estimate the sweep's cost from an assumed output length."""
    rate = gpqa.PRICES.get(spec.slug)
    permutations = gpqa.permutation_table(1)
    prompt_chars = sum(
        len(gpqa.build_prompt(q, permutations[(q["question_idx"], 0)])[0]) for q in questions)
    in_tokens = int(prompt_chars / 4) * args.n          # ~4 chars/token, per arm
    out_tokens = len(questions) * args.n * args.assume_output_tokens
    print(f"\nassumed {args.assume_output_tokens} output tokens/attempt "
          f"(thinking included); input estimated from prompt length")
    if rate is None:
        print(f"no rates for {spec.slug} in gpqa.PRICES")
        return 0
    per_arm = in_tokens * rate["input"] + out_tokens * rate["output"]
    for effort in efforts:
        print(f"  {effort:<7} ~{in_tokens/1e3:.1f}k in + {out_tokens/1e6:.2f}M out  "
              f"~${per_arm:,.2f}")
    print(f"  {'total':<7} {'':>30}~${per_arm * len(efforts):,.2f}")
    return 0


def run_arm(args, spec, effort: str, questions: list[dict], permutations: dict,
            client, out_root: Path) -> None:
    """Run every missing (question, run) attempt for one (model, effort) arm."""
    directory = gpqa.run_dir(out_root, args.model, effort)
    samples_path = directory / gpqa.SAMPLES_FILE
    directory.mkdir(parents=True, exist_ok=True)

    if args.redo_all and samples_path.exists():
        samples_path.unlink()
    done = {(s["question_idx"], s["run_idx"]) for s in gpqa.read_samples(samples_path)}
    todo = [(q, run_idx) for q in questions for run_idx in range(args.n)
            if (q["question_idx"], run_idx) not in done]

    (directory / gpqa.META_FILE).write_text(json.dumps({
        "dataset": gpqa.CSV_URL,
        "model": spec.slug, "api_id": gpqa.provider_id(spec), "family": spec.family,
        "effort": effort, "runs_per_question": args.n, "max_tokens": args.max_tokens,
        "thinking_display": "omitted" if spec.family == "anthropic" else None,
        "prompt_template": gpqa.QUERY_TEMPLATE,
        "choice_shuffle": "simple-evals random.Random(0), repeat-major draw order",
        "sampling_params": "none",
        "n_questions": len(questions),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")

    print(f"\n[{effort}] {len(done)} sample(s) already present, {len(todo)} to run "
          f"-> {samples_path}")
    if not todo:
        return

    lock = Lock()
    counts = {"done": 0, "correct": 0, "failed": 0, "truncated": 0, "unparsed": 0, "cost": 0.0}
    started = time.monotonic()

    def one(question: dict, run_idx: int) -> dict:
        permutation = permutations[(question["question_idx"], run_idx)]
        prompt, choices, correct_letter = gpqa.build_prompt(question, permutation)
        began = time.monotonic()

        def warn(attempt: int, exc: Exception) -> None:
            with lock:
                print(f"  retry {attempt} for question {question['question_idx']} "
                      f"run {run_idx}: {type(exc).__name__}: {exc}")

        reply = gpqa.with_retries(
            lambda: gpqa.call(client, spec, prompt, effort, args.max_tokens), on_error=warn)
        answer, how = gpqa.extract_answer(reply.text)
        return {
            "model": spec.slug, "effort": effort,
            "question_idx": question["question_idx"], "run_idx": run_idx,
            "record_id": question["record_id"],
            "domain": question["domain"], "subdomain": question["subdomain"],
            # The letters mean different options on different attempts, so the presented
            # order is stored with the record.
            "permutation": permutation, "choices": choices,
            "gold": correct_letter, "answer": answer, "correct": answer == correct_letter,
            "extracted_by": how, "truncated": reply.truncated, "finish": reply.finish,
            "raw": reply.raw,     # read by gpqa.is_refusal
            "usage": reply.usage, "cost_usd": gpqa.cost_of(args.model, reply.usage),
            "latency_s": round(time.monotonic() - began, 1),
            "response": reply.text,
            "reasoning_summary": reply.reasoning_text,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, q, r): (q["question_idx"], r) for q, r in todo}
        for future in as_completed(futures):
            question_idx, run_idx = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                with lock:
                    counts["failed"] += 1
                    print(f"  FAILED question {question_idx} run {run_idx} after retries: "
                          f"{type(exc).__name__}: {exc}")
                continue
            with lock:
                gpqa.append_sample(samples_path, record)
                counts["done"] += 1
                counts["correct"] += bool(record["correct"])
                counts["truncated"] += bool(record["truncated"])
                counts["unparsed"] += record["answer"] is None
                counts["cost"] += record["cost_usd"] or 0.0
                print(f"  [{counts['done']}/{len(todo)}] q{question_idx:03d} r{run_idx} "
                      f"{'OK ' if record['correct'] else 'x  '}"
                      f"answer={record['answer']} gold={record['gold']} "
                      f"out={record['usage'].get('output_tokens')} "
                      f"{record['latency_s']}s  running acc "
                      f"{counts['correct']}/{counts['done']} "
                      f"${counts['cost']:.2f}")

    minutes = (time.monotonic() - started) / 60
    print(f"[{effort}] {counts['done']} new sample(s) in {minutes:.1f} min, "
          f"{counts['failed']} failed, {counts['truncated']} hit the output cap, "
          f"{counts['unparsed']} had no 'Answer: X' line, "
          f"${counts['cost']:.2f} spent (this invocation)")


if __name__ == "__main__":
    raise SystemExit(main())
