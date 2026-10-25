#!/usr/bin/env python3
"""Terminal-Bench 2.1, stage 2 of 2: turn harbor trial outputs into a cost_summary.json.

Walks every trial result.json under a run dir, extracts tokens, cost and verdict per trial,
prices the tokens with common/pricing.py, and writes <run dir>/cost_summary.json
({run_dir, aggregate, per_instance, litellm_capture}) and prints a summary.

Fields read from each trial result.json:
    task_name, trial_name, config.agent.model_name
    agent_result (or step_results[i].agent_result): n_input_tokens (incl. cache),
        n_cache_tokens, n_output_tokens, cost_usd
    verifier_result.rewards["reward"] == 1.0  -> task resolved
harbor's trial metrics have no reasoning-token or cache-creation field, so the per-trial
cost (`our_cost_usd_no_cachewrite`) omits Anthropic's cache-write premium. The run's
litellm_calls.jsonl (written by run_sweep.py's capture) has both and is summarized under
`litellm_capture`.

Usage
    python terminal_bench/collect_cost.py data/terminal_bench/anthropic_claude-sonnet-5/low
    python terminal_bench/collect_cost.py --all      # every <model>/<effort> under the data root
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import models, paths, pricing

BENCHMARK = "terminal-bench-2-1"
OUT_ROOT = paths.data_root(BENCHMARK)


def _rel(path: Path) -> str:
    """`path` relative to the data root when it lies inside it."""
    try:
        return str(path.relative_to(paths.DATA_ROOT))
    except ValueError:
        return str(path)


def run_model(run_dir: Path) -> str | None:
    """The registered model a <model>/<effort> run dir belongs to, from its directory name."""
    by_slug = {models.litellm_slug(m): spec.litellm_id for m, spec in models.MODELS.items()}
    return by_slug.get(run_dir.parent.name)


def recompute_cost(litellm_model: str, n_input, n_cache, n_output) -> float | None:
    """Cost of harbor's stored token counts at the rate table; None without rates or tokens.

    harbor stores cache reads but not cache creation, so for Anthropic this is a lower bound
    (no cache-write premium).
    """
    return pricing.cost_from_usage(litellm_model, prompt_tokens=n_input,
                                   output_tokens=n_output, cache_read=n_cache)


def _totals(result: dict) -> tuple[int | None, int | None, int | None, float | None]:
    """Sum (n_input, n_cache, n_output, cost_usd) over whichever agent contexts exist."""
    ctxs = []
    if result.get("agent_result") is not None:
        ctxs = [result["agent_result"]]
    elif result.get("step_results"):
        ctxs = [s["agent_result"] for s in result["step_results"] if s.get("agent_result")]
    n_in = n_cache = n_out = None
    cost = None
    for c in ctxs:
        if c.get("n_input_tokens") is not None:
            n_in = (n_in or 0) + c["n_input_tokens"]
        if c.get("n_cache_tokens") is not None:
            n_cache = (n_cache or 0) + c["n_cache_tokens"]
        if c.get("n_output_tokens") is not None:
            n_out = (n_out or 0) + c["n_output_tokens"]
        if c.get("cost_usd") is not None:
            cost = (cost or 0.0) + c["cost_usd"]
    return n_in, n_cache, n_out, cost


def _find_capture(leaf: Path) -> Path | None:
    """Walk up from a run dir to OUT_ROOT looking for the run's litellm_calls.jsonl."""
    for d in [leaf, *leaf.parents]:
        cand = d / "litellm_calls.jsonl"
        if cand.exists():
            return cand
        if d == OUT_ROOT:
            break
    return None


def _cap_int(d: dict | None, key: str) -> int:
    return int((d or {}).get(key) or 0)


def summarize_capture(cap_path: Path, litellm_model: str | None) -> dict:
    """Aggregate the LiteLLM per-call capture: tokens (incl. reasoning and cache creation),
    LiteLLM's own cost, and the rate-table cost of the full breakdown."""
    recs = []
    for line in cap_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # Dedupe by response_id (unique per API call): resumed jobs append to the same file.
    seen: set[str] = set()
    deduped = []
    for r in recs:
        rid = r.get("response_id")
        if rid and rid in seen:
            continue
        if rid:
            seen.add(rid)
        deduped.append(r)
    recs = deduped
    n_calls = len(recs)
    prompt = completion = reasoning = cache_read = cache_creation = 0
    litellm_cost = 0.0
    n_reasoning_summaries = 0
    # Calls that produced reasoning tokens; 0 is expected on a `none` run.
    calls_with_reasoning = 0
    for r in recs:
        u = r.get("usage") or {}
        pd = u.get("prompt_tokens_details") or {}
        cd = u.get("completion_tokens_details") or {}
        prompt += _cap_int(u, "prompt_tokens")
        completion += _cap_int(u, "completion_tokens")
        cache_read += _cap_int(pd, "cached_tokens")
        cache_creation += _cap_int(pd, "cache_creation_tokens") or _cap_int(pd, "cache_write_tokens")
        call_reasoning = _cap_int(cd, "reasoning_tokens")
        reasoning += call_reasoning
        calls_with_reasoning += bool(call_reasoning)
        if r.get("response_cost"):
            litellm_cost += r["response_cost"]
        if r.get("reasoning_content"):
            n_reasoning_summaries += 1

    # With cache-creation tokens the cost is exact (see pricing.cost_from_usage).
    our_exact = pricing.cost_from_usage(litellm_model, prompt_tokens=prompt,
                                        output_tokens=completion, cache_read=cache_read,
                                        cache_creation=cache_creation)
    return {
        "path": _rel(cap_path),
        "n_calls": n_calls,
        "tokens": {
            "prompt": prompt, "completion": completion, "reasoning": reasoning,
            "cache_read": cache_read, "cache_creation": cache_creation,
        },
        "n_reasoning_summaries": n_reasoning_summaries,
        "calls_with_reasoning": calls_with_reasoning,
        "thinking_call_share": round(calls_with_reasoning / n_calls, 4) if n_calls else None,
        "litellm_response_cost_usd": round(litellm_cost, 6),
        "our_exact_recompute_usd": round(our_exact, 6) if our_exact is not None else None,
        "exact_ratio_litellm_over_ours": round(litellm_cost / our_exact, 4) if our_exact else None,
    }


def _resolved(result: dict) -> bool | None:
    vr = result.get("verifier_result")
    if not vr or vr.get("rewards") is None:
        return None
    return vr["rewards"].get("reward") == 1.0


def summarize(run_dir: Path) -> dict:
    # Trials are priced at the rates of the run's model. A trial's own model_name can carry
    # another provider prefix (e.g. `openai/` when the model is served through an
    # OpenAI-compatible base URL); it is kept in `model` but not used for pricing.
    priced_as = run_model(run_dir)
    per = []
    for rj in sorted(run_dir.rglob("result.json")):
        d = json.loads(rj.read_text())
        if "task_name" not in d:  # skip job-level lock/aggregate files named result.json
            continue
        n_in, n_cache, n_out, cost = _totals(d)
        model_id = (d.get("config", {}).get("agent") or {}).get("model_name")
        our_cost = recompute_cost(priced_as or model_id, n_in, n_cache, n_out)
        per.append({
            "task_name": d.get("task_name"),
            "trial_name": d.get("trial_name"),
            "model": model_id,
            "resolved": _resolved(d),
            "n_input_tokens": n_in,
            "n_cache_tokens": n_cache,
            "uncached_input_tokens": (n_in - n_cache) if (n_in is not None and n_cache is not None) else None,
            "n_output_tokens": n_out,
            "reasoning_tokens": None,  # not in harbor's trial metrics
            "cost_usd": round(cost, 6) if cost is not None else None,              # harbor-reported
            "our_cost_usd_no_cachewrite": round(our_cost, 6) if our_cost is not None else None,  # rate table
        })
    costs = [r["cost_usd"] for r in per if r["cost_usd"] is not None]
    resolved = [r["resolved"] for r in per if r["resolved"] is not None]
    # harbor's reported cost vs the rate-table cost of the same trials
    pairs = [(r["cost_usd"], r["our_cost_usd_no_cachewrite"]) for r in per
             if r["cost_usd"] is not None and r["our_cost_usd_no_cachewrite"] is not None]
    harbor_sum = round(sum(h for h, _ in pairs), 6) if pairs else None
    our_sum = round(sum(o for _, o in pairs), 6) if pairs else None
    table_costs = [r["our_cost_usd_no_cachewrite"] for r in per
                   if r["our_cost_usd_no_cachewrite"] is not None]
    agg = {
        "n_trials": len(per),
        "n_with_cost": len(costs),
        "n_resolved": sum(1 for r in resolved if r),
        "resolve_rate": round(sum(1 for r in resolved if r) / len(resolved), 4) if resolved else None,
        # Tokens x common/pricing.py: the figure used in the analysis. harbor's own figure is
        # kept as a cross-check; it uses LiteLLM's rates unless model_info was registered.
        "total_cost_from_table_usd": round(sum(table_costs), 4) if table_costs else None,
        "mean_cost_from_table_usd": round(sum(table_costs) / len(table_costs), 4) if table_costs else None,
        "total_cost_usd": round(sum(costs), 4) if costs else None,       # harbor-reported
        "mean_cost_usd": round(sum(costs) / len(costs), 4) if costs else None,
        "reconciliation": {
            "harbor_reported_usd": harbor_sum,
            "our_recompute_usd": our_sum,
            # >= 1.0; above 1 by the Anthropic cache-write premium harbor bills.
            "ratio_harbor_over_ours": round(harbor_sum / our_sum, 4) if our_sum else None,
        },
    }
    out = {"run_dir": _rel(run_dir), "aggregate": agg, "per_instance": per}
    cap = _find_capture(run_dir)
    if cap is not None:
        model_id = priced_as or (per[0]["model"] if per else None)
        out["litellm_capture"] = summarize_capture(cap, model_id)
    return out


def _print(summary: dict) -> None:
    a = summary["aggregate"]
    rec = a["reconciliation"]
    print(f"\n{summary['run_dir']}")
    print(f"  trials={a['n_trials']}  cost_reported={a['n_with_cost']}  "
          f"resolved={a['n_resolved']}/{a['n_trials']} ({a['resolve_rate']})  "
          f"total=${a['total_cost_from_table_usd']}  mean=${a['mean_cost_from_table_usd']}  "
          f"(rate table)")
    print(f"  reconcile: harbor=${rec['harbor_reported_usd']}  table=${rec['our_recompute_usd']}  "
          f"harbor/table={rec['ratio_harbor_over_ours']} (expect 1.0-1.3; >1 = Anthropic "
          f"cache-write premium)")
    if a["n_trials"] <= 6:  # per-task detail for small runs only
        print(f"    {'resolved':8} {'harbor$':>9} {'ours$':>9}  in/cache/out  task")
        for r in summary["per_instance"]:
            c = f"${r['cost_usd']}" if r["cost_usd"] is not None else "—no cost"
            o = f"${r['our_cost_usd_no_cachewrite']}" if r["our_cost_usd_no_cachewrite"] is not None else "—"
            print(f"    {str(r['resolved']):8} {c:>9} {o:>9}  "
                  f"{r['n_input_tokens']}/{r['n_cache_tokens']}/{r['n_output_tokens']}  {r['task_name']}")
    if a["n_trials"] and a["n_with_cost"] == 0:
        print("  WARNING: no cost_usd on any trial (LiteLLM had no rates for the model); the "
              "rate-table totals above are unaffected.")
    if rec["ratio_harbor_over_ours"] is not None and not (0.98 <= rec["ratio_harbor_over_ours"] <= 1.35):
        print(f"  WARNING: harbor/table ratio {rec['ratio_harbor_over_ours']} is outside the "
              "expected band: harbor priced this run with other rates (e.g. --no-register-pricing). "
              "The rate-table totals above are unaffected.")
    lc = summary.get("litellm_capture")
    if lc:
        tk = lc["tokens"]
        print(f"  litellm_capture ({lc['n_calls']} calls): prompt={tk['prompt']} completion={tk['completion']} "
              f"reasoning={tk['reasoning']} cache_read={tk['cache_read']} cache_creation={tk['cache_creation']} "
              f"reasoning_summaries={lc['n_reasoning_summaries']}")
        share = lc.get("thinking_call_share")
        if share is not None:
            print(f"    effort check: {lc['calls_with_reasoning']}/{lc['n_calls']} calls produced "
                  f"reasoning ({share * 100:.1f}%)")
        if share and summary["run_dir"].rstrip("/").endswith("/none"):
            print(f"    WARNING: {share * 100:.1f}% of calls in this 'none' run produced reasoning.")
        print(f"    EXACT cost: litellm=${lc['litellm_response_cost_usd']}  ours(full breakdown)="
              f"${lc['our_exact_recompute_usd']}  litellm/ours={lc['exact_ratio_litellm_over_ours']}"
              "  (expect ~1.0)")


def _run_dirs(root: Path) -> list[Path]:
    """Return the <model>/<effort> run dirs under `root` that contain trials.

    `root` may be OUT_ROOT (all runs), a <model> dir (all its efforts), or a single
    <model>/<effort> dir; every result.json below a run dir belongs to that run.
    """
    if root == OUT_ROOT:
        cands = sorted(root.glob("*/*"))
    elif root.parent == OUT_ROOT:            # a model dir
        cands = sorted(root.glob("*"))
    else:                                    # a model/effort dir (or anything else specific)
        cands = [root]
    return [d for d in cands if d.is_dir() and any(d.rglob("result.json"))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", type=Path,
                    help="a <model>/<effort> run dir, a <model> dir, or the benchmark data root")
    ap.add_argument("--all", action="store_true",
                    help="summarize every <model>/<effort> run under the benchmark data root")
    args = ap.parse_args()

    if args.all:
        root = OUT_ROOT
    elif args.run_dir:
        root = (args.run_dir if args.run_dir.is_absolute() else (Path.cwd() / args.run_dir)).resolve()
    else:
        ap.error("pass a run dir or --all")
    if not root.exists():
        raise SystemExit(f"no such dir: {root}")

    runs = _run_dirs(root)
    if not runs:
        raise SystemExit(f"no trial result.json found under {root}")
    for run_dir in runs:
        summary = summarize(run_dir)
        (run_dir / "cost_summary.json").write_text(json.dumps(summary, indent=2))
        _print(summary)
    print(f"\nwrote cost_summary.json in {len(runs)} config(s)")


if __name__ == "__main__":
    main()
