"""Where the Act-onomy annotation inputs and outputs live.

    <ANNOTATION_ROOT>/<bench>/                 bench = tb | pro-corrected
        sample.json                            the fixed instance sample shared by every cell
        selection.json                         the instances annotated (here: the whole sample)
        skeletons/_manifest.json               de-blinding map (model, effort, outcome)
        skeletons/<model>__<effort>/<instance>.skeleton.json     extract.py
        annotations/<model>__<effort>/<instance>.annotated.json  run_batch.py
        runs/<model>__<effort>/<instance>/     raw annotator output per trajectory
        runs/_batch_log.jsonl                  one line per annotation attempt

A cell is one `<model>__<effort>` directory; `<model>` is the result-tree directory name
(the LiteLLM slug on Terminal-Bench, the model slug on SWE-Bench Pro).
"""
from __future__ import annotations

import sys
from pathlib import Path

ACTONOMY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ACTONOMY_DIR.parents[1]))     # repository root, for `common`
from common import paths  # noqa: E402

CODEBOOK_MD = ACTONOMY_DIR / "skill" / "trace-qualitative-analyst" / "references" / "codebook.md"

# CLI benchmark name -> benchmark id in common/paths.py
BENCHMARKS: dict[str, str] = {
    "tb": "terminal-bench-2-1",
    "pro-corrected": "swebench-pro-corrected",
}
TB_BENCHMARKS = {"tb"}

# Models whose corrected SWE-Bench Pro prose is annotated on reasoning_content + content
# (`--substrate both`): they keep most of their narration in reasoning_content, so their
# content field alone is empty on most turns. Every other cell is annotated on content only.
BOTH_FIELD_MODELS = {"claude-sonnet-5", "deepseek-v4-flash", "kimi-k3"}

# The ten Act-onomy v4.2 groups (the codebook is the authority for subgroups and leaves).
GROUPS = (
    "Retrieval", "Memory", "Planning", "Reasoning", "Evaluate",
    "Deciding", "Grounding", "Executing", "Learning", "Reflection",
)

PHASES = ("reproduce", "localize", "recover", "edit", "verify", "submit", "other")

TODO = "TODO"


def default_substrate(bench: str, model: str) -> str:
    """The annotated text field used for a cell (see BOTH_FIELD_MODELS)."""
    return "both" if bench == "pro-corrected" and model in BOTH_FIELD_MODELS else "content"


def rel(path: Path) -> str:
    """Path relative to DATA_ROOT (or the repository) for logs and records."""
    for root in (paths.DATA_ROOT, paths.REPO_ROOT):
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def data_root(bench: str) -> Path:
    """Result tree that a benchmark's trajectories are read from."""
    return paths.data_root(BENCHMARKS[bench])


def bench_root(bench: str) -> Path:
    return paths.ANNOTATION_ROOT / bench


def skeleton_root(bench: str) -> Path:
    return bench_root(bench) / "skeletons"


def manifest_path(bench: str) -> Path:
    """Kept at the skeleton root, outside every cell, so it is never shown to the annotator."""
    return skeleton_root(bench) / "_manifest.json"


def annotation_root(bench: str) -> Path:
    return bench_root(bench) / "annotations"


def runs_root(bench: str) -> Path:
    return bench_root(bench) / "runs"


def sample_path(bench: str) -> Path:
    return bench_root(bench) / "sample.json"


def selection_path(bench: str) -> Path:
    return bench_root(bench) / "selection.json"


def submission(model: str, effort: str) -> str:
    return f"{model}__{effort}"


def split_submission(name: str) -> tuple[str, str]:
    model, _, effort = name.partition("__")
    return model, effort


def codebook_index():
    """The codebook hierarchy, parsed from the vendored codebook.md."""
    from codebook_index import CodebookIndex
    return CodebookIndex.from_file(CODEBOOK_MD)
