#!/usr/bin/env python3
"""DeepSWE v1.1 per-task solve counts: rollouts passed per (model, task, effort level).

    python figures/deepswe/solve_counts.py      # writes output/figures/deepswe/deepswe_tasks.csv

Reads the leaderboard snapshot `paths.DEEPSWE_ROOT/deepswe_v1_1_trials.csv` (see
`snapshot_deepswe_trials.py`; the full DeepSWE benchmark, one row per rollout). One output row
per (model, task, effort level): how many rollouts passed, how many were run, and whether the
task counts as solved at that level (passed on at least one rollout). An errored rollout is not
a pass. `plotted` marks the levels a model's ladder draws (`plotted_levels`).

Also the shared pieces of the solve-count heatmaps (`load`, `CMAP`, `NORM`), used by
`make_app_deepswe_instances_gained.py`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import paths  # noqa: E402

TRIALS = paths.DEEPSWE_ROOT / "deepswe_v1_1_trials.csv"
OUT_DIR = paths.output_dir("figures", "deepswe")
TASKS_CSV = OUT_DIR / "deepswe_tasks.csv"

# The leaderboard's model slugs -> display names, in the figure's order.
MODELS = {"gpt-5-6-luna": "GPT-5.6-luna", "gpt-5-6-terra": "GPT-5.6-terra",
          "gpt-5-6-sol": "GPT-5.6-sol", "claude-sonnet-5": "Claude Sonnet 5",
          "glm-5-2": "GLM-5.2", "inkling": "Inkling", "deepseek-v4-flash": "DeepSeek V4 Flash"}
LEVELS = ["low", "medium", "high", "xhigh", "max"]
MAX_ROLLOUTS = 4

SHORT_DOT = "#E8710A"   # marks a cell with fewer than MAX_ROLLOUTS rollouts
# Five discrete shades of Blues, one per pass count 0..4.
CMAP = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.04, 1.0, MAX_ROLLOUTS + 1)))
NORM = BoundaryNorm(np.arange(-0.5, MAX_ROLLOUTS + 1), CMAP.N)


def plotted_levels(levels: set[str]) -> list[str]:
    """low..xhigh where run, and `max` only for a model with no `xhigh` (GLM-5.2), where it
    takes the top slot. The other models' `max` stays in the CSV but is not part of the ladder."""
    out = [e for e in LEVELS[:4] if e in levels]
    if "max" in levels and "xhigh" not in levels:
        out.append("max")
    return out


def build_rows(trials: Path = TRIALS) -> list[dict]:
    """One row per (model, task, effort level), in model / level / task order."""
    cell: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    with trials.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] in MODELS:
                cell[(r["model"], r["task_name"], r["reasoning_effort"])].append(
                    r["passed"] == "True")
    rows = []
    for site_model, label in MODELS.items():
        levels = {e for (m, _, e) in cell if m == site_model}
        drawn = plotted_levels(levels)
        for (m, task, effort), passes in cell.items():
            if m != site_model:
                continue
            rows.append({"model": label, "site_model": site_model, "task_name": task,
                         "effort": effort, "passed": sum(passes), "rollouts": len(passes),
                         "solved": any(passes), "plotted": effort in drawn})
    rows.sort(key=lambda r: (list(MODELS).index(r["site_model"]), LEVELS.index(r["effort"]),
                             r["task_name"]))
    return rows


def write_tasks_csv(rows: list[dict], path: Path = TASKS_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def load(rows: list[dict]) -> dict[str, dict]:
    """{model: {"levels", "tasks", "passed", "rollouts"}}, one heatmap's worth per model.

    Every level the model was run at is a row; columns are tasks sorted by total passes across
    all levels (ties: by how many levels solve the task, then by name).
    """
    passed: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    rollouts: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for r in rows:
        key = (r["effort"], r["task_name"])
        passed[r["model"]][key] = int(r["passed"])
        rollouts[r["model"]][key] = int(r["rollouts"])
    out = {}
    for model in passed:
        levels = [e for e in LEVELS if any(k[0] == e for k in passed[model])]
        tasks = sorted({k[1] for k in passed[model]})
        p = np.array([[passed[model].get((e, t), 0) for t in tasks] for e in levels])
        n = np.array([[rollouts[model].get((e, t), 0) for t in tasks] for e in levels])
        order = sorted(range(len(tasks)),
                       key=lambda j: (-p[:, j].sum(), -(p[:, j] > 0).sum(), tasks[j]))
        out[model] = {"levels": levels, "tasks": [tasks[j] for j in order],
                      "passed": p[:, order], "rollouts": n[:, order]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=Path, default=TRIALS, help="trials snapshot CSV")
    args = ap.parse_args()
    rows = build_rows(args.trials)
    print(f"wrote {write_tasks_csv(rows)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
