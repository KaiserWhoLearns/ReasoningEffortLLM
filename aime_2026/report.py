#!/usr/bin/env python3
"""Score stored AIME 2026 attempts: accuracy, tokens and dollars per (model, effort) arm.

    python aime_2026/report.py --model claude-sonnet-5
    python aime_2026/report.py --model gpt-5.6-terra --efforts medium,xhigh --per-problem
    python aime_2026/report.py --model gpt-5.6-terra --csv

Answers are re-extracted from the stored response text and costs recomputed from the stored
token counts, so everything is derived from `samples.jsonl`.

    mean@k   mean per-problem solve rate over the stored attempts (the reported accuracy);
             +- is the standard error across problems.
    pass@k   probability that at least one of k attempts is correct, by the unbiased
             estimator 1 - C(n-c, k)/C(n, k) (Chen et al. 2021); a problem counts as solved
             when any of its k attempts is. Problems with fewer than k attempts are left out.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, for `common`
import aime  # noqa: E402
from common import models, paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--efforts", help="comma-separated; default: every arm found on disk")
    p.add_argument("--out-root", default=str(aime.DEFAULT_OUT_ROOT))
    p.add_argument("--per-problem", action="store_true",
                   help="also print the per-problem solve rate for each arm")
    p.add_argument("--k", type=int, default=10,
                   help="k for the pass@k column (default: %(default)s)")
    p.add_argument("--csv", action="store_true",
                   help="also write the arm table to <output>/aime_2026/<model>.csv")
    return p.parse_args()


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021), None if n < k."""
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(model: str, effort: str, samples: list[dict], k: int) -> dict:
    """One arm's row, recomputed from its sample records."""
    by_problem: dict[int, list[bool]] = defaultdict(list)
    unparsed = fallback = truncated = 0
    out_tokens: list[int] = []
    reasoning_tokens: list[int] = []
    cost = 0.0

    for s in samples:
        answer, how = aime.extract_answer(s.get("response") or "")
        by_problem[s["problem_idx"]].append(answer == s["gold"])
        unparsed += answer is None
        fallback += how == "last_int"
        truncated += bool(s.get("truncated"))

        usage = s.get("usage") or {}
        if usage.get("output_tokens") is not None:
            out_tokens.append(int(usage["output_tokens"]))
        if usage.get("reasoning_tokens") is not None:
            reasoning_tokens.append(int(usage["reasoning_tokens"]))
        cost += aime.cost_of(model, usage) or 0.0

    rates = [sum(v) / len(v) for v in by_problem.values()]
    stderr = statistics.stdev(rates) / len(rates) ** 0.5 if len(rates) > 1 else 0.0

    pass_k_vals = [est for v in by_problem.values()
                   if (est := pass_at_k(len(v), sum(v), k)) is not None]
    pass_k_skipped = len(by_problem) - len(pass_k_vals)
    pass_k = statistics.mean(pass_k_vals) if pass_k_vals else None
    pass_k_stderr = (statistics.stdev(pass_k_vals) / len(pass_k_vals) ** 0.5
                     if len(pass_k_vals) > 1 else 0.0)
    return {
        "effort": effort,
        "problems": len(by_problem),
        "samples": len(samples),
        "mean_at_k": statistics.mean(rates) if rates else 0.0,
        "stderr": stderr,
        "k": k,
        "pass_at_k": pass_k,
        "pass_k_stderr": pass_k_stderr,
        "pass_k_skipped": pass_k_skipped,
        "solved_all": sum(1 for r in rates if r == 1.0),
        "solved_none": sum(1 for r in rates if r == 0.0),
        "unparsed": unparsed,
        "fallback_parses": fallback,
        "truncated": truncated,
        "mean_output_tokens": statistics.mean(out_tokens) if out_tokens else None,
        "mean_reasoning_tokens": statistics.mean(reasoning_tokens) if reasoning_tokens else None,
        "cost_usd": cost,
        "per_problem": {idx: sum(v) / len(v) for idx, v in sorted(by_problem.items())},
    }


def main() -> int:
    args = parse_args()
    slug = models.name_slug(args.model)
    out_root = Path(args.out_root)
    model_dir = out_root / slug

    if args.efforts:
        efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
    else:
        efforts = sorted((d.name for d in model_dir.iterdir() if d.is_dir()),
                         key=lambda e: models.EFFORT_ORDER.get(e, 99)) \
                  if model_dir.exists() else []
    if not efforts:
        raise SystemExit(f"no arms found under {model_dir}; run run_eval.py first")

    rows = []
    for effort in efforts:
        samples = aime.read_samples(aime.run_dir(out_root, args.model, effort)
                                    / aime.SAMPLES_FILE)
        if not samples:
            print(f"(skipping {effort}: no samples)")
            continue
        rows.append(summarize(args.model, effort, samples, args.k))

    if not rows:
        return 1

    pass_k_hdr = f"pass@{args.k}"
    print(f"\nAIME 2026 ({aime.HF_DATASET}) — {slug}")
    print(f"{'effort':<8}{'probs':>6}{'n':>6}{'mean@k':>9}{'+-':>7}"
          f"{pass_k_hdr:>9}{'+-':>7}"
          f"{'all/none':>10}{'out tok':>10}{'think tok':>11}{'$':>9}{'$/attempt':>11}")
    for r in rows:
        think = f"{r['mean_reasoning_tokens']:,.0f}" if r["mean_reasoning_tokens"] is not None else "n/a"
        out = f"{r['mean_output_tokens']:,.0f}" if r["mean_output_tokens"] is not None else "n/a"
        extremes = f"{r['solved_all']}/{r['solved_none']}"
        per_attempt = r["cost_usd"] / max(r["samples"], 1)
        if r["pass_at_k"] is not None:
            pk = f"{r['pass_at_k'] * 100:>8.1f}%"
            pke = f"{r['pass_k_stderr'] * 100:>6.1f}%"
        else:
            pk, pke = f"{'n/a':>9}", f"{'':>7}"
        print(f"{r['effort']:<8}{r['problems']:>6}{r['samples']:>6}"
              f"{r['mean_at_k'] * 100:>8.1f}%{r['stderr'] * 100:>6.1f}%"
              f"{pk}{pke}"
              f"{extremes:>10}{out:>10}{think:>11}"
              f"{r['cost_usd']:>9.2f}{per_attempt:>11.3f}")

    print("\nmean@k = mean per-problem solve rate; +- = standard error across problems. "
          f"pass@{args.k} = share of problems solved by at least one of {args.k} attempts.")
    print("all/none = problems solved by every attempt / by none. "
          "think tok = the provider's metered reasoning tokens, a subset of out tok.")

    for r in rows:
        notes = []
        if r["unparsed"]:
            notes.append(f"{r['unparsed']} attempt(s) produced no readable integer")
        if r["fallback_parses"]:
            notes.append(f"{r['fallback_parses']} graded by the last-integer fallback "
                         f"rather than \\boxed{{}}")
        if r["truncated"]:
            notes.append(f"{r['truncated']} hit the output cap")
        if r["pass_at_k"] is not None and r["pass_k_skipped"]:
            notes.append(f"pass@{r['k']} leaves out {r['pass_k_skipped']} problem(s) "
                         f"with fewer than {r['k']} attempts")
        if notes:
            print(f"\n{r['effort']}: " + "; ".join(notes) + ".")

    if args.per_problem:
        for r in rows:
            print(f"\nper-problem solve rate — {r['effort']}")
            for idx, rate in r["per_problem"].items():
                print(f"  p{idx:02d} {rate * 100:5.0f}%")

    if args.csv:
        path = paths.output_dir("aime_2026", create=True) / f"{slug}.csv"
        fields = ["effort", "problems", "samples", "mean_at_k", "stderr",
                  "k", "pass_at_k", "pass_k_stderr", "solved_all",
                  "solved_none", "unparsed", "fallback_parses", "truncated",
                  "mean_output_tokens", "mean_reasoning_tokens", "cost_usd"]
        lines = [",".join(fields)]
        for r in rows:
            lines.append(",".join("" if r[f] is None else str(r[f]) for f in fields))
        path.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
