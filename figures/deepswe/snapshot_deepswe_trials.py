#!/usr/bin/env python3
"""Snapshot the DeepSWE v1.1 rollouts the figures read into `paths.DEEPSWE_ROOT`.

    python figures/deepswe/snapshot_deepswe_trials.py                 # downloads trials.json
    python figures/deepswe/snapshot_deepswe_trials.py trials.json     # or reads a saved copy

The DeepSWE figures and table use the public leaderboard's own rollouts, published at
https://deepswe.datacurve.ai/artifacts/v1.1/trials.json (the data behind
https://deepswe.datacurve.ai/data/v1.1, ~51 MB, every benchmark the site runs). This keeps only
the full DeepSWE benchmark (`source = deep-swe`, `eval_scope = full`), the study's models and
the fields below. The site can change, so a fresh download may differ from the paper's snapshot.

Writes `deepswe_v1_1_trials.csv` in `paths.DEEPSWE_ROOT`. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import paths  # noqa: E402

URL = "https://deepswe.datacurve.ai/artifacts/v1.1/trials.json"
OUT = paths.DEEPSWE_ROOT / "deepswe_v1_1_trials.csv"

# The site's slugs for the study's models (Inkling has no DeepSWE rollouts).
MODELS = ("gpt-5-6-luna", "gpt-5-6-terra", "gpt-5-6-sol", "claude-sonnet-5", "glm-5-2",
          "inkling", "deepseek-v4-flash")
FIELDS = ["trial_name", "task_name", "model", "reasoning_effort", "run_attempt", "passed",
          "errored", "outcome", "included_in_score", "reward", "n_agent_steps", "n_input_tokens",
          "n_output_tokens", "n_reasoning_tokens", "cost_usd"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trials_json", nargs="?", type=Path,
                    help="a saved copy of the site's trials.json (default: download it)")
    args = ap.parse_args()
    if args.trials_json:
        data = json.loads(args.trials_json.read_text())
    else:
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL) as resp:
            data = json.load(resp)
    rows = [r for r in data["rows"] if r.get("source") == "deep-swe"
            and r.get("eval_scope") == "full" and r.get("model") in MODELS]
    rows.sort(key=lambda r: (MODELS.index(r["model"]), r["reasoning_effort"], r["trial_name"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rollouts)")


if __name__ == "__main__":
    main()
