"""On-disk layout for run results and analysis products.

Results (raw run output) live under DATA_ROOT, one subdirectory per benchmark:

    data/
        aime_2026/<model>/<effort>/               single-call runs
        gpqa_diamond/<model>/<effort>/
        swebench_pro/<model>/<effort>/            mini-swe-agent runs, original environment
        swebench_pro/<model>/eval/<effort>/       official evaluator verdicts
        swebench_pro_corrected/<model>/...        corrected environment, same layout
        terminal_bench/<litellm-slug>/<effort>/   harbor jobs
        actonomy/<bench>/                         behavior-annotation inputs and outputs
        deepswe/                                  DeepSWE v1.1 leaderboard snapshot
        datasets/                                 downloaded benchmark inputs
        logs/<benchmark>/                         captured stdout of long runs

Derived products (tables, figures, routing results) live under OUTPUT_ROOT, in a
subdirectory named after the stage that writes them (see `output_dir`).

Both roots can be moved with the REL_DATA_DIR and REL_OUTPUT_DIR environment variables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
DATA_ROOT = Path(os.environ.get("REL_DATA_DIR", REPO_ROOT / "data")).resolve()
OUTPUT_ROOT = Path(os.environ.get("REL_OUTPUT_DIR", REPO_ROOT / "output")).resolve()
# External checkouts fetched on demand (e.g. the official SWE-Bench Pro evaluator).
VENDOR_DIR = REPO_ROOT / "vendor"

# benchmark id -> subdirectory of DATA_ROOT
BENCHMARK_DATA_SUBDIR: dict[str, str] = {
    "aime-2026": "aime_2026",
    "gpqa-diamond": "gpqa_diamond",
    "swebench-pro": "swebench_pro",
    "swebench-pro-corrected": "swebench_pro_corrected",
    "terminal-bench-2-1": "terminal_bench",
}

ANNOTATION_ROOT = DATA_ROOT / "actonomy"
DEEPSWE_ROOT = DATA_ROOT / "deepswe"
DATASETS_ROOT = DATA_ROOT / "datasets"
LOGS_ROOT = DATA_ROOT / "logs"


def data_root(benchmark: str) -> Path:
    """Root of a benchmark's result tree."""
    if benchmark not in BENCHMARK_DATA_SUBDIR:
        raise KeyError(f"unknown benchmark {benchmark!r}; known: {list(BENCHMARK_DATA_SUBDIR)}")
    return DATA_ROOT / BENCHMARK_DATA_SUBDIR[benchmark]


def log_dir(benchmark: str, create: bool = False) -> Path:
    """Where to redirect a long run's stdout."""
    d = LOGS_ROOT / benchmark
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(*parts: str, create: bool = False) -> Path:
    """OUTPUT_ROOT/<parts...>, e.g. output_dir("figures") or output_dir("self_routing")."""
    d = OUTPUT_ROOT.joinpath(*parts)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(benchmark: str, model_slug: str, effort: str) -> Path:
    """The directory one (benchmark, model, effort) configuration writes into."""
    return data_root(benchmark) / model_slug / effort


def load_env(required: list[str] | tuple[str, ...] = ()) -> None:
    """Load the repository's .env and exit if any required variable is missing."""
    load_dotenv(ENV_FILE)
    missing = sorted(k for k in required if not os.environ.get(k))
    if missing:
        sys.exit(f"missing environment variable(s): {', '.join(missing)} "
                 f"(set them in the shell or in {ENV_FILE})")
