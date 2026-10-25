#!/usr/bin/env python3
"""SWE-Bench Pro, stage 3 of 3: per-effort cost summaries and a quality/cost report.

For every model in the dataset's tree (or those given with --model) and every effort directory
with a manifest.json, this

1. rewrites `<effort>/cost_summary.json` from the stored trajectories: per-instance token counts
   (common/usage.py), `cost_from_tokens` = tokens x the rate table in common/pricing.py,
   api_calls, exit status, submission, and the corrected environment's per-instance evidence
   (`env_setup_ok`: the sanitizer printed STRIP-OK; `egress_policy`: the allow-list sha);
2. reads the evaluator's verdicts from `eval/<effort>/eval_results.json` and applies the cost cap:
   an instance that passes but costs more than `--cost-cap` (default $5) counts as a failure;
3. writes a table per model, plus a paired comparison of each lower effort against a baseline
   effort (`medium`, else the model's highest scored level) on the instances both solved, to
   `paths.output_dir(<benchmark tree>)/report.{md,json}`.

Tokens are the measurement and dollars are derived, so a rate change is a re-report, not a re-run.
`instance_cost` is what LiteLLM billed during the run; `harness/table` compares it with the table.
`output` tokens are the inclusive generated total and `reasoning` is a subset of them;
`reasoning_available` is false where no metered count exists (see common/usage.py).

Usage:
    python report.py --dataset pro
    python report.py --dataset pro-corrected --model gpt-5.6-sol --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for `common`
from common import models as model_registry, paths, pricing, usage

import variants

ORDER = model_registry.EFFORT_ORDER
DEFAULT_BASELINE = "medium"
DEFAULT_COST_CAP = 5.0


# --------------------------------------------------------------------------- cost_summary.json

def run_instances(run_dir: Path) -> list[str]:
    """The manifest's instances plus any instance with a trajectory on disk, sorted."""
    manifest = run_dir / "manifest.json"
    ids = set(json.loads(manifest.read_text())["instances"]) if manifest.exists() else set()
    ids |= {p.parent.name for p in run_dir.glob("*/*.traj.json")}
    return sorted(ids)


def summarize(run_dir: Path, instance_ids: list[str], model: str) -> dict:
    """Per-instance tokens, cost, api_calls, exit status and patch presence from the trajectories."""
    per = []
    for iid in instance_ids:
        traj = run_dir / iid / f"{iid}.traj.json"
        rec = {"instance_id": iid, "exit_status": None, "instance_cost": None,
               "api_calls": None, "submitted": False, "patch_len": 0,
               "cost_from_tokens": None, "env_setup_ok": None, "env_setup_note": None,
               "egress_policy": None}
        if traj.exists():
            d = json.loads(traj.read_text())
            info = d.get("info", {})
            stats = info.get("model_stats", {})
            rec["exit_status"] = info.get("exit_status")
            # Present only when the environment ran a setup command (the corrected dataset).
            # git can print to stderr ahead of the sanitizer's verdict, so look for the verdict
            # lines rather than at the first line.
            setup = info.get("setup_output")
            if isinstance(setup, str) and setup.strip():
                verdicts = [l.strip() for l in setup.splitlines()
                            if l.strip().startswith(("STRIP-OK", "STRIP-FAIL"))]
                rec["env_setup_ok"] = bool(verdicts) and all(v.startswith("STRIP-OK")
                                                             for v in verdicts)
                rec["env_setup_note"] = (verdicts[-1] if verdicts
                                         else setup.strip().splitlines()[-1])[:200]
            egress = info.get("egress_policy")
            if isinstance(egress, dict):
                rec["egress_policy"] = egress.get("policy_sha256_16")
            rec["instance_cost"] = stats.get("instance_cost")
            rec["api_calls"] = stats.get("api_calls")
            sub = info.get("submission") or ""
            rec["submitted"] = bool(sub.strip())
            rec["patch_len"] = len(sub)
            tokens = usage.from_trajectory(traj)
            rec.update(tokens)
            priced = pricing.cost_from_usage(
                model, prompt_tokens=tokens["input_tokens"],
                output_tokens=tokens["output_tokens"],
                cache_read=tokens["cached_input_tokens"],
                cache_creation=tokens["cache_creation_tokens"])
            rec["cost_from_tokens"] = round(priced, 6) if priced is not None else None
        per.append(rec)
    costs = [r["instance_cost"] for r in per if r["instance_cost"] is not None]
    priced = [r["cost_from_tokens"] for r in per if r["cost_from_tokens"] is not None]
    calls = [r["api_calls"] for r in per if r["api_calls"] is not None]
    paired = [(r["instance_cost"], r["cost_from_tokens"]) for r in per
              if r["instance_cost"] and r["cost_from_tokens"]]
    agg = {
        "n_instances": len(per),
        "n_trajectories": sum(1 for r in per if r["exit_status"] is not None),
        "n_submitted": sum(1 for r in per if r["submitted"]),
        "total_cost": round(sum(costs), 4) if costs else None,
        "mean_cost": round(sum(costs) / len(costs), 4) if costs else None,
        "total_cost_from_tokens": round(sum(priced), 4) if priced else None,
        "mean_cost_from_tokens": round(sum(priced) / len(priced), 4) if priced else None,
        # 1.0 when LiteLLM billed the run on this table's rates.
        "cost_ratio_harness_over_table": (
            round(sum(h for h, _ in paired) / sum(t for _, t in paired), 4) if paired else None),
        "mean_api_calls": round(sum(calls) / len(calls), 2) if calls else None,
        "n_env_setup_ok": sum(1 for r in per if r["env_setup_ok"]),
        "n_egress_sealed": sum(1 for r in per if r["egress_policy"]),
    }
    for field in usage.COUNT_FIELDS:
        values = [r[field] for r in per if r.get(field) is not None]
        agg[f"mean_{field}"] = round(sum(values) / len(values)) if values else None
    # Share of LM calls that produced reasoning; a `none` run should be 0.
    agg["n_lm_calls"] = sum(r.get("n_calls") or 0 for r in per) or None
    agg["calls_with_reasoning"] = sum(r.get("calls_with_reasoning") or 0 for r in per)
    agg["thinking_call_share"] = (round(agg["calls_with_reasoning"] / agg["n_lm_calls"], 4)
                                  if agg["n_lm_calls"] else None)
    return {"aggregate": agg, "per_instance": per}


def write_cost_summary(run_dir: Path, model: str) -> dict:
    """Rewrite run_dir/cost_summary.json from the trajectories and return it."""
    summary = summarize(run_dir, run_instances(run_dir), model)
    (run_dir / "cost_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- report

def effort_dirs(model_root: Path) -> list[Path]:
    return sorted((p for p in model_root.iterdir()
                   if p.is_dir() and p.name in ORDER and (p / "manifest.json").exists()),
                  key=lambda p: ORDER[p.name])


def collect(model: str, root_dir: Path, cost_cap: float = DEFAULT_COST_CAP) -> list[dict[str, Any]]:
    root = root_dir / model_registry.name_slug(model)
    if not root.exists():
        sys.exit(f"no runs at {root}")
    rows = []
    for run_dir in effort_dirs(root):
        summary = write_cost_summary(run_dir, model)
        agg = summary["aggregate"]
        per_instance = {r["instance_id"]: r for r in summary["per_instance"]}
        tokens = {i: r for i, r in per_instance.items() if r["exit_status"] is not None}

        row: dict[str, Any] = {
            "model": model,
            "effort": run_dir.name,
            **agg,
            "resolved": None,
            "resolve_rate": None,
            "cost_per_resolved": None,
            "resolved_ids": [],
            "per_instance": per_instance,
        }
        ev = root / "eval" / run_dir.name / "eval_results.json"
        if ev.exists():
            results = json.loads(ev.read_text())
            resolved_ids = sorted(i for i, passed in results.items() if passed)
            # Cost cap: an instance that passes but costs more than the cap counts as a failure.
            if cost_cap:
                over = {i for i in resolved_ids
                        if (tokens.get(i, {}).get("cost_from_tokens") or 0) > cost_cap}
                if over:
                    row["over_budget_unresolved"] = sorted(over)
                    resolved_ids = [i for i in resolved_ids if i not in over]
            row["resolved_ids"] = resolved_ids
            row["resolved"] = len(resolved_ids)
            row["n_scored"] = len(results)
            if results:
                row["resolve_rate"] = round(len(resolved_ids) / len(results), 4)

        available = list(tokens.values())
        if row["resolved_ids"] and row["total_cost_from_tokens"] is not None:
            row["cost_per_resolved"] = round(
                row["total_cost_from_tokens"] / len(row["resolved_ids"]), 4)
        reasoning_values = [int(v["reasoning_tokens"]) for v in available if v["reasoning_available"]]
        row["mean_reasoning_tokens"] = (
            round(statistics.mean(reasoning_values)) if reasoning_values else None)
        row["reasoning_instances_observed"] = len(reasoning_values)
        row["reasoning_source"] = next((v["reasoning_source"] for v in available
                                        if v["reasoning_source"]), None)
        rows.append(row)
    if not rows:
        sys.exit(f"no effort directory with a manifest.json under {root}; run run_sweep.py")
    return rows


def _baseline_effort(rows: list[dict[str, Any]], requested: str) -> str | None:
    """`requested` when the model has it scored, else its highest scored level."""
    scored = [r["effort"] for r in rows if r["resolved"] is not None]
    if requested in scored:
        return requested
    return max(scored, key=lambda e: ORDER[e]) if scored else None


def _paired_comparisons(rows: list[dict[str, Any]],
                        requested_baseline: str = DEFAULT_BASELINE) -> list[dict[str, Any]]:
    by_effort = {row["effort"]: row for row in rows}
    baseline_effort = _baseline_effort(rows, requested_baseline)
    baseline = by_effort.get(baseline_effort) if baseline_effort else None
    if not baseline:
        return []
    comparisons = []
    lower_efforts = sorted((e for e in by_effort if ORDER[e] < ORDER[baseline_effort]),
                           key=lambda e: ORDER[e])
    for effort in lower_efforts:
        lower = by_effort.get(effort)
        if not lower or lower["resolved"] is None:
            continue
        lower_solved = set(lower["resolved_ids"])
        common = sorted(lower_solved & set(baseline["resolved_ids"]))
        pairs = [(lower["per_instance"].get(i), baseline["per_instance"].get(i)) for i in common]
        pairs = [(a, b) for a, b in pairs if a and b and a.get("total_tokens") is not None
                 and b.get("total_tokens") is not None]

        def paired_metric(key: str) -> tuple[float | None, float | None, float | None, int]:
            valid = [(float(a[key]), float(b[key])) for a, b in pairs
                     if a.get(key) is not None and b.get(key) is not None]
            if not valid:
                return None, None, None, 0
            lower_mean = statistics.mean(a for a, _ in valid)
            baseline_mean = statistics.mean(b for _, b in valid)
            delta_pct = ((lower_mean / baseline_mean) - 1) * 100 if baseline_mean else None
            lower_wins = sum(a < b for a, b in valid)
            precision = 4 if key.startswith("cost") else 1
            return round(lower_mean, precision), round(baseline_mean, precision), (
                round(delta_pct, 1) if delta_pct is not None else None), lower_wins

        comparison: dict[str, Any] = {
            "model": lower["model"],
            "baseline_effort": baseline_effort,
            "lower_effort": effort,
            "lower_resolved": len(lower_solved),
            "also_resolved_by_baseline": len(common),
            "paired_n": len(pairs),
        }
        for key in ("input_tokens", "output_tokens", "total_tokens", "cost_from_tokens"):
            lo, base, pct, wins = paired_metric(key)
            comparison[f"lower_mean_{key}"] = lo
            comparison[f"baseline_mean_{key}"] = base
            comparison[f"delta_pct_{key}"] = pct
            comparison[f"lower_uses_less_{key}"] = wins
        comparisons.append(comparison)
    return comparisons


def _fmt(value: Any, key: str = "") -> str:
    if value is None:
        return "—"
    if key == "resolve_rate":
        return f"{value * 100:.1f}%"
    # Dollars before tokens: "mean_cost_from_tokens" ends in "tokens" but is money.
    if "cost" in key:
        return f"{value:,.4f}" if key.startswith("mean_") else str(value)
    if key.startswith("mean_") and "tokens" in key:
        return f"{value:,.0f}"
    return str(value)


def render_md(all_rows: dict[str, list[dict[str, Any]]], comparisons: list[dict[str, Any]],
              dataset: variants.Dataset, cost_cap: float) -> str:
    lines = [f"# Reasoning effort: cost and tokens on {dataset.title}", "",
             f"**Environment:** {dataset.env_note}.", "",
             "`output tok` is the inclusive generated total (reasoning is a subset of it); "
             "`total tok` = input + output.",
             "`mean $` is tokens x `common/pricing.py`; `harness/table` is what LiteLLM billed "
             "divided by that.",
             f"Instances that pass but cost more than ${cost_cap:g} count as unresolved.", ""]
    cols = [
        ("effort", "effort"), ("n_instances", "n"), ("n_scored", "scored"),
        ("resolved", "resolved"), ("resolve_rate", "resolve%"),
        ("mean_cost_from_tokens", "mean $"), ("cost_ratio_harness_over_table", "harness/table"),
        ("mean_input_tokens", "mean input tok"), ("mean_output_tokens", "mean output tok"),
        ("mean_reasoning_tokens", "mean reasoning tok"), ("mean_total_tokens", "mean total tok"),
        ("mean_api_calls", "mean calls"),
    ]
    for model, rows in all_rows.items():
        lines += [f"## {model}", ""]
        lines.append("| " + " | ".join(label for _, label in cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_fmt(row.get(key), key) for key, _ in cols) + " |")
        lines.append("")

    lines += ["## Lower effort versus the baseline effort, on instances both solved", ""]
    if not comparisons:
        lines += ["No scored baseline/lower-effort pairs are available.", ""]
    else:
        lines += [
            "A negative delta means the lower effort used fewer tokens or dollars on the same "
            "solved instances.", "",
            "| model | comparison | paired n | lower/base mean total tok | delta | lower used less (n) | lower/base mean output tok | output delta | lower/base mean $ | cost delta |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in comparisons:
            def pair(key: str) -> str:
                precision = 4 if key.startswith("cost") else 1
                return (f"{c[f'lower_mean_{key}']:,.{precision}f} / "
                        f"{c[f'baseline_mean_{key}']:,.{precision}f}")

            def pct(key: str) -> str:
                return f"{c[f'delta_pct_{key}']:+.1f}%"

            lines.append(
                f"| {c['model']} | {c['lower_effort']} vs {c['baseline_effort']} | {c['paired_n']} | "
                f"{pair('total_tokens')} | {pct('total_tokens')} | "
                f"{c['lower_uses_less_total_tokens']}/{c['paired_n']} | "
                f"{pair('output_tokens')} | {pct('output_tokens')} | "
                f"{pair('cost_from_tokens')} | {pct('cost_from_tokens')} |")
        lines.append("")

    leaks = [(r["model"], r["effort"], r["thinking_call_share"])
             for rows in all_rows.values() for r in rows
             if r["effort"] == "none" and (r.get("thinking_call_share") or 0) > 0]
    if leaks:
        lines += ["## `none` runs with reasoning calls", ""]
        lines += [f"* `{m}` / `{e}`: {share * 100:.1f}% of LM calls produced reasoning"
                  for m, e, share in leaks]
        lines.append("")
    return "\n".join(lines)


def _serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"resolved_ids", "per_instance"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    variants.add_argument(parser)
    parser.add_argument("--model", action="append", dest="models",
                        help="model to include; repeatable (default: every model in the tree)")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE,
                        help=f"effort the lower levels are paired against (default "
                             f"{DEFAULT_BASELINE}; falls back to the model's highest scored level)")
    parser.add_argument("--cost-cap", type=float, default=DEFAULT_COST_CAP,
                        help=f"an instance costing more than this counts as unresolved "
                             f"(default {DEFAULT_COST_CAP:g}; 0 disables)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output stem (default: <output>/<benchmark tree>/report)")
    args = parser.parse_args()

    dataset = variants.resolve(args.dataset)
    root_dir = paths.data_root(dataset.benchmark)
    if not root_dir.exists():
        sys.exit(f"no runs at {root_dir}")
    names = args.models or sorted(d.name for d in root_dir.iterdir() if d.is_dir())
    all_rows = {model: collect(model, root_dir, args.cost_cap) for model in names}
    comparisons = [c for rows in all_rows.values() for c in _paired_comparisons(rows, args.baseline)]
    payload = {
        "dataset": dataset.key,
        "dataset_title": dataset.title,
        "environment_note": dataset.env_note,
        "cost_cap": args.cost_cap,
        "rates": {model: pricing.rates(model) for model in names},
        "models": {model: [_serializable_row(row) for row in rows] for model, rows in all_rows.items()},
        "paired_lower_vs_baseline": comparisons,
    }
    stem = args.output or paths.output_dir(paths.BENCHMARK_DATA_SUBDIR[dataset.benchmark]) / "report"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    markdown = render_md(all_rows, comparisons, dataset, args.cost_cap)
    stem.with_suffix(".md").write_text(markdown + "\n")
    print(markdown)
    print(f"wrote {stem.with_suffix('.md')} and {stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
