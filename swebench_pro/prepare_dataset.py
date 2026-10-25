#!/usr/bin/env python3
"""Prepare SWE-Bench Pro for mini-swe-agent.

Reads the test split of ScaleAI/SWE-bench_Pro (731 instances, from HuggingFace or a local export
given with --source) and writes `paths.DATASETS_ROOT/swebench_pro/mini_instances.jsonl`, one row
per instance with the fields mini's swebench runner expects:

  problem_statement  the task text: problem_statement + requirements + interface, composed as in
                     SWE-bench_Pro-os helper_code/create_problem_statement.py
  image_name         the official prebuilt image, jefzda/sweap-images:<dockerhub_tag>
  instance_id        output directory and preds.json key

plus the columns evaluate.py passes to the evaluator (fail_to_pass, pass_to_pass, base_commit,
before_repo_set_cmd, selected_test_files_to_run, ...). The repository is at /app in these images.

The pilot subset (`subsets/pilot.txt`, 44 ids) is 4 evenly spaced instances from the
lexicographically sorted ids of each of the 11 repositories: with n ids in a repository,
ids[::n // 4][:4]. This script recomputes it and checks it against the committed file (or writes
the file if it is missing).

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --source test.jsonl      # a local jsonl export of the split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for `common`
from common import paths

HF_DATASET = "ScaleAI/SWE-bench_Pro"
OUT = paths.DATASETS_ROOT / "swebench_pro" / "mini_instances.jsonl"
PILOT = Path(__file__).resolve().parent / "subsets" / "pilot.txt"
PILOT_PER_REPO = 4
DOCKERHUB_USER = "jefzda"

# Columns carried through for the evaluator.
KEEP = ["instance_id", "base_commit", "repo", "repo_language", "dockerhub_tag",
        "fail_to_pass", "pass_to_pass", "before_repo_set_cmd", "selected_test_files_to_run"]


def compose_problem_statement(row: dict) -> str:
    """As SWE-bench_Pro-os/helper_code/create_problem_statement.py."""
    return (f"{row['problem_statement']}\n\n"
            f"Requirements:\n{row['requirements']}\n\n"
            f"New interfaces introduced:\n{row['interface']}")


def load_rows(source: Path | None) -> list[dict]:
    if source is None:
        from datasets import load_dataset
        return list(load_dataset(HF_DATASET, split="test"))
    return [json.loads(line) for line in source.open()]


def pilot_ids(rows: list[dict], per_repo: int = PILOT_PER_REPO) -> list[str]:
    """Evenly spaced picks from each repository's sorted ids (deterministic, no RNG)."""
    by_repo: dict[str, list[str]] = {}
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r["instance_id"])
    pilot: list[str] = []
    for repo in sorted(by_repo):
        ids = sorted(by_repo[repo])
        k = min(per_repo, len(ids))
        stride = max(1, len(ids) // k)
        pilot.extend(ids[::stride][:k])
    return pilot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=None,
                    help=f"local .jsonl export of {HF_DATASET} (test split); "
                         "default: download it from HuggingFace")
    args = ap.parse_args()

    rows = load_rows(args.source)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            tag = r.get("dockerhub_tag")
            if not tag:
                raise SystemExit(f"instance {r['instance_id']} has no dockerhub_tag")
            inst = {k: r.get(k) for k in KEEP}
            inst["image_name"] = f"{DOCKERHUB_USER}/sweap-images:{tag}"
            inst["problem_statement"] = compose_problem_statement(r)
            inst["problem_statement_raw"] = r["problem_statement"]
            f.write(json.dumps(inst) + "\n")
    print(f"wrote {len(rows)} instances -> {OUT}")

    pilot = pilot_ids(rows)
    if not PILOT.exists():
        PILOT.parent.mkdir(parents=True, exist_ok=True)
        PILOT.write_text("\n".join(pilot) + "\n")
        print(f"wrote pilot subset ({len(pilot)} ids) -> {PILOT}")
    elif PILOT.read_text().split() == pilot:
        print(f"pilot subset matches {PILOT} ({len(pilot)} ids)")
    else:
        print(f"WARNING: the pilot rule gives a different id list from {PILOT}; "
              "the dataset differs from the one the paper used")


if __name__ == "__main__":
    main()
