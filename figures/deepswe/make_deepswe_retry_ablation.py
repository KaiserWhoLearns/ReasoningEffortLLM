#!/usr/bin/env python3
"""Table: two-attempt retry policies on DeepSWE v1.1, reconstructed from the recorded rollouts.

    python figures/deepswe/make_deepswe_retry_ablation.py

No inference or network access. Each policy makes one attempt and, after a failure, one fresh
retry: L->L, L->H and H->H, where L/H are `low`/`xhigh` (GLM-5.2: `high`/`max`). Per task, all
ordered distinct rollout pairs within an effort (and all cross-effort pairs for escalation) are
averaged, then tasks are weighted equally. Errors count as failures; absent rollouts are not
imputed. Costs use `common/pricing.py` with the recorded cache usage, without a long-context
surcharge or verifier/infrastructure cost. Bootstrap intervals (20,000 paired task resamples)
describe variation across tasks conditional on the recorded rollout pools.

Reads the snapshots in `paths.DEEPSWE_ROOT`. Writes:
    output/figures/deepswe_retries/{tasks,summary,contrasts}.csv
    output/tables/app_deepswe_retries.tex
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import paths, pricing  # noqa: E402

OUT = paths.output_dir("figures", "deepswe_retries")
TABLE = paths.output_dir("tables") / "app_deepswe_retries.tex"
MODELS = {
    "gpt-5-6-luna": ("gpt-5.6-luna", "GPT-5.6-luna"),
    "gpt-5-6-terra": ("gpt-5.6-terra", "GPT-5.6-terra"),
    "gpt-5-6-sol": ("gpt-5.6-sol", "GPT-5.6-sol"),
    "claude-sonnet-5": ("claude-sonnet-5", "Claude Sonnet 5"),
    "glm-5-2": ("glm-5.2", "GLM-5.2"),
}
POLICIES = {"LL": ("L", "L"), "LH": ("L", "H"), "HH": ("H", "H")}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimate(first, second, same):
    pairs = [(a, b) for a, b in product(first, second) if a[0] != b[0]]
    assert len(pairs) == len(first) * (len(second) - int(same))
    success = np.mean([a[1] or b[1] for a, b in pairs])
    cost = np.mean([a[2] + (not a[1]) * b[2] for a, b in pairs])
    if same:
        n, passed = len(first), sum(a[1] for a in first)
        assert math.isclose(success, 1 - (n-passed)*(n-passed-1)/(n*(n-1)), abs_tol=1e-12)
    else:
        assert math.isclose(success, 1 - (1-np.mean([a[1] for a in first])) *
                            (1-np.mean([a[1] for a in second])), abs_tol=1e-12)
    assert cost >= np.mean([a[2] for a in first]) - 1e-12
    return float(success), float(cost)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    trials = read_csv(paths.DEEPSWE_ROOT / "deepswe_v1_1_trials.csv")
    tokens = read_csv(paths.DEEPSWE_ROOT / "deepswe_v1_1_rollout_tokens.csv")
    assert len({r["trial_name"] for r in trials}) == len(trials)
    assert len({r["trial_name"] for r in tokens}) == len(tokens)
    tokens = {r["trial_name"]: r for r in tokens}
    task_rows, summary, contrasts = [], [], []
    rng = np.random.default_rng(20261023)
    for slug, (model, label) in MODELS.items():
        low, high = ("high", "max") if slug == "glm-5-2" else ("low", "xhigh")
        cells = defaultdict(lambda: defaultdict(list))
        for r in trials:
            if r["model"] != slug or r["reasoning_effort"] not in (low, high):
                continue
            t = tokens[r["trial_name"]]  # Fail rather than silently omit unpriced rollouts.
            assert all(r[k] == t[k] for k in ("model", "task_name", "reasoning_effort", "passed"))
            assert r["passed"] == str(r["outcome"] == "pass")
            cost = pricing.cost_from_usage(model, prompt_tokens=int(t["prompt_tokens"]),
                output_tokens=int(t["completion_tokens"]), cache_read=int(t["cache_read_tokens"]),
                cache_creation=int(t["cache_creation_tokens"]))
            assert cost is not None and math.isfinite(cost) and cost >= 0
            arm = "L" if r["reasoning_effort"] == low else "H"
            cells[r["task_name"]][arm].append((r["trial_name"], r["passed"] == "True", cost))
        assert len(cells) == 113
        values, complete = [], []
        for task, arms in sorted(cells.items()):
            assert set(arms) == {"L", "H"} and all(len(v) >= 2 for v in arms.values())
            complete.append(all(len(v) == 4 for v in arms.values()))
            row = {"model": label, "task": task, "low": low, "high": high,
                   "n_low": len(arms["L"]), "n_high": len(arms["H"])}
            estimates = []
            for policy, (a, b) in POLICIES.items():
                success, cost = estimate(arms[a], arms[b], a == b)
                row[f"{policy}_success"] = success
                row[f"{policy}_cost_usd"] = cost
                estimates.append((success, cost))
            values.append(estimates)
            task_rows.append(row)
        values = np.array(values)
        for scenario, mask in [("recorded", np.ones(113, dtype=bool)),
                               ("complete_cells", np.array(complete))]:
            means = values[mask].mean(axis=0)
            for idx, policy in enumerate(POLICIES):
                summary.append({"model": label, "scenario": scenario, "low": low, "high": high,
                    "n_tasks": int(mask.sum()), "policy": policy,
                    "success_pct": means[idx, 0] * 100, "mean_cost_usd": means[idx, 1]})
        draws = rng.integers(0, len(values), size=(20000, len(values)))
        for idx, control in [(0, "LL"), (2, "HH")]:
            delta = values[:, 1] - values[:, idx]
            boot = delta[draws].mean(axis=1)
            bounds = np.quantile(boot, [0.025, 0.975], axis=0)
            contrasts.append({"model": label, "contrast": f"LH-{control}",
                "success_diff_pp": delta[:, 0].mean()*100,
                "success_ci_low_pp": bounds[0, 0]*100, "success_ci_high_pp": bounds[1, 0]*100,
                "cost_diff_usd": delta[:, 1].mean(),
                "cost_ci_low_usd": bounds[0, 1], "cost_ci_high_usd": bounds[1, 1]})
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "tasks.csv", task_rows)
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "contrasts.csv", contrasts)
    lines = [r"% Generated by figures/deepswe/make_deepswe_retry_ablation.py; recorded-rollout estimates.",
        r"\begin{table}[t]", r"    \centering", r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \caption{Retry control study on DeepSWE v1.1, averaged equally over 113 tasks per model.",
        r"    Each policy makes one initial attempt and at most one fresh retry after failure.",
        r"    Success is reported in percent; cost is mean API dollars per task, including retries only when needed.",
        r"    $L/H$ denote \texttt{low}/\texttt{xhigh}, except for GLM-5.2 (\texttt{high}/\texttt{max}).}",
        r"    \label{tab:app-deepswe-retries}",
        r"    \begin{tabular}{@{}lrrrrrr@{}}", r"        \toprule",
        r"        & \multicolumn{2}{c}{$L\!\rightarrow\!L$} & \multicolumn{2}{c}{$L\!\rightarrow\!H$} & \multicolumn{2}{c}{$H\!\rightarrow\!H$} \\",
        r"        \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"        Model & Success & Cost (\$) & Success & Cost (\$) & Success & Cost (\$) \\",
        r"        \midrule"]
    for _, label in MODELS.values():
        rows = [r for r in summary if r["model"] == label and r["scenario"] == "recorded"]
        lines.append("        " + label + " & " + " & ".join(
            f"{r['success_pct']:.1f} & {r['mean_cost_usd']:.3f}" for r in rows) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n")
    for r in summary:
        if r["scenario"] == "recorded":
            print(f"{r['model']:16s} {r['policy']} {r['success_pct']:5.1f}% ${r['mean_cost_usd']:.3f}")
    for r in contrasts:
        print(f"{r['model']:16s} {r['contrast']}  success {r['success_diff_pp']:+.1f} pp "
              f"[{r['success_ci_low_pp']:+.1f}, {r['success_ci_high_pp']:+.1f}]  "
              f"cost ${r['cost_diff_usd']:+.3f} [{r['cost_ci_low_usd']:+.3f}, "
              f"{r['cost_ci_high_usd']:+.3f}]")


if __name__ == "__main__":
    main()
