#!/usr/bin/env python3
"""Snapshot the per-rollout token counts of the DeepSWE v1.1 rollouts into `paths.DEEPSWE_ROOT`.

    python figures/deepswe/snapshot_deepswe_tokens.py [TRAJECTORY_DIR]

`make_solvable_cost.py --deepswe` and `make_deepswe_retry_ablation.py` price each DeepSWE
rollout from its token counts at `common/pricing.py`'s rates. The leaderboard's `trials.json`
(`deepswe_v1_1_trials.csv`, see `snapshot_deepswe_trials.py`) carries the verdicts and totals
but no reasoning or cache-write counts, so both are read from the leaderboard's own trajectories
(`<TRAJECTORY_DIR>/<model>__<effort>/<trial_name>.json`, ~9 GB; default
`paths.DEEPSWE_ROOT/deepswe-v1.1-trajectories`). One row per rollout of the trials snapshot:

    prompt_tokens         final_metrics.total_prompt_tokens      (= trials.json n_input_tokens)
    cache_read_tokens     final_metrics.total_cached_tokens      (= trials.json n_cache_tokens)
    completion_tokens     final_metrics.total_completion_tokens  (= trials.json n_output_tokens)
    reasoning_tokens      final_metrics.extra.total_reasoning_tokens; empty where the trajectory
                          records none (Claude Sonnet 5's thinking is not metered)
    cache_creation_tokens sum of the agent steps' prompt_tokens_details.cache_creation_tokens
                          (recorded for Anthropic only, else 0)
    peak_prompt_tokens    the largest single call's prompt_tokens (calls above ~272k are billed
                          at OpenAI's long-context rate; the repository's price table has no
                          such tier)

The three totals are taken from `final_metrics`, which equal the site's own counts. The script
stops if prompt or completion tokens disagree with the trials snapshot. A rollout with no
trajectory file is left out and named. Standard library only.

Writes `deepswe_v1_1_rollout_tokens.csv` in `paths.DEEPSWE_ROOT`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import paths  # noqa: E402

TRIALS = paths.DEEPSWE_ROOT / "deepswe_v1_1_trials.csv"
OUT = paths.DEEPSWE_ROOT / "deepswe_v1_1_rollout_tokens.csv"
DEFAULT_TRAJECTORIES = paths.DEEPSWE_ROOT / "deepswe-v1.1-trajectories"
FIELDS = ["trial_name", "task_name", "model", "reasoning_effort", "passed", "prompt_tokens",
          "cache_read_tokens", "cache_creation_tokens", "completion_tokens", "reasoning_tokens",
          "peak_prompt_tokens"]


def details(step: dict, key: str) -> dict:
    return (((step.get("metrics") or {}).get("extra") or {}).get(key)) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory_dir", nargs="?", type=Path, default=DEFAULT_TRAJECTORIES,
                    help="the leaderboard's trajectories, one <model>__<effort>/ dir per cell")
    traj_root = ap.parse_args().trajectory_dir
    rows, missing, bad = [], [], []
    for t in csv.DictReader(TRIALS.open()):
        path = traj_root / f"{t['model']}__{t['reasoning_effort']}" / f"{t['trial_name']}.json"
        if not path.exists():
            missing.append(t["trial_name"])
            continue
        traj = json.loads(path.read_text())
        fm = traj["final_metrics"]
        agent = [s for s in traj["steps"] if s.get("source") == "agent"]
        row = {
            "trial_name": t["trial_name"], "task_name": t["task_name"], "model": t["model"],
            "reasoning_effort": t["reasoning_effort"], "passed": t["passed"],
            "prompt_tokens": fm["total_prompt_tokens"],
            "cache_read_tokens": fm.get("total_cached_tokens") or 0,
            "completion_tokens": fm["total_completion_tokens"],
            "reasoning_tokens": (fm.get("extra") or {}).get("total_reasoning_tokens", ""),
            "cache_creation_tokens": sum(details(s, "prompt_tokens_details")
                                         .get("cache_creation_tokens") or 0 for s in agent),
            "peak_prompt_tokens": max([(s.get("metrics") or {}).get("prompt_tokens") or 0
                                       for s in agent] or [0]),
        }
        for mine, site in (("prompt_tokens", "n_input_tokens"),
                           ("completion_tokens", "n_output_tokens")):
            if str(row[mine]) != t[site]:
                bad.append(f"{t['trial_name']}: {mine} {row[mine]} vs site {site} {t[site]}")
        rows.append(row)
    if bad:
        raise SystemExit("trajectory totals disagree with the trials snapshot:\n  " +
                         "\n  ".join(bad[:20]))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rollouts)")
    if missing:
        print(f"note: {len(missing)} rollout(s) in the trials snapshot have no trajectory and are "
              f"left out: {', '.join(missing)}")


if __name__ == "__main__":
    main()
