"""The task text a self-router is shown, per benchmark: what the model sees at the start of a run.

    swebench_pro     `problem_statement` from DATASETS_ROOT/swebench_pro/mini_instances.jsonl,
                     the statement mini-swe-agent hands the agent (built by
                     swebench_pro/prepare_dataset.py).
    terminal_bench   the "Task Description:" block of the first user message of a recorded
                     Terminus-2 trajectory. Harbor renders it into the prompt and stores it
                     nowhere else, so it is read back from any recorded trial of the task.
    aime_2026        the problem statement from MathArena/aime_2026 (`aime.load_problems`;
                     needs the network on a cold cache).
    gpqa_diamond     the question and its four choices as `gpqa.build_prompt` renders them with
                     attempt 0's shuffle, so the correct answer is not always option A.

Texts are cached under OUTPUT_ROOT/self_routing/cache/<source>.json; delete a file to
re-extract it.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import paths  # noqa: E402

CACHE_DIR = paths.output_dir("self_routing", "cache")

# dataset key (figures/datasets.py) -> task source. A source is also the subdirectory of
# DATA_ROOT/self_routing/ that holds the selection calls.
SOURCE_OF: dict[str, str] = {
    "swebench_pro_pilot": "swebench_pro",
    "terminal_bench": "terminal_bench",
    "aime_2026": "aime_2026",
    "gpqa_diamond": "gpqa_diamond",
}


@contextmanager
def _unshadowed_datasets():
    """Let `import datasets` find Hugging Face's package instead of figures/datasets.py.

    The routing scripts put figures/ on sys.path, and its `datasets` module shadows the Hugging
    Face package that `aime.load_problems` imports. Both are hidden for one import and restored.
    """
    figures_dir = (REPO_ROOT / "figures").resolve()
    saved_path, saved_modules = list(sys.path), {}
    for name in [n for n in sys.modules if n == "datasets" or n.startswith("datasets.")]:
        saved_modules[name] = sys.modules.pop(name)
    sys.path = [p for p in sys.path if Path(p).resolve() != figures_dir]
    try:
        yield
    finally:
        sys.path = saved_path
        for name in [n for n in sys.modules if n == "datasets" or n.startswith("datasets.")]:
            del sys.modules[name]
        sys.modules.update(saved_modules)


@dataclass(frozen=True)
class Source:
    """One task source: how to load its texts, and what the router is told about the setting."""
    key: str
    unit: str
    setting: str
    load: Callable[[], dict[str, str]]


# --------------------------------------------------------------------------- SWE-Bench Pro

PRO_INSTANCES = paths.DATASETS_ROOT / "swebench_pro" / "mini_instances.jsonl"


def load_pro_tasks() -> dict[str, str]:
    if not PRO_INSTANCES.exists():
        raise SystemExit(f"missing {PRO_INSTANCES}; run swebench_pro/prepare_dataset.py")
    out = {}
    with PRO_INSTANCES.open() as fh:
        for line in fh:
            row = json.loads(line)
            out[row["instance_id"]] = row["problem_statement"]
    return out


# --------------------------------------------------------------------------- Terminal-Bench

TB_ROOT = paths.data_root("terminal-bench-2-1")
TB_START = "Task Description:\n"
TB_END = "\n\nCurrent terminal state:"


def _tb_description(message: str) -> str | None:
    """The task block of a Terminus-2 opening prompt, without the scaffold instructions and the
    initial screen, which are the same on every task."""
    start = message.find(TB_START)
    if start < 0:
        return None
    body = message[start + len(TB_START):]
    end = body.find(TB_END)
    return (body[:end] if end >= 0 else body).strip() or None


def load_tb_tasks() -> dict[str, str]:
    """task_name -> description, from the first recorded trial of each task."""
    if not TB_ROOT.exists():
        raise SystemExit(f"missing {TB_ROOT}; no Terminal-Bench runs to read task text from")
    out: dict[str, str] = {}
    # <model>/<effort>/<job>/<trial>/config.json
    for config in sorted(TB_ROOT.glob("*/*/*/*/config.json")):
        try:
            task_name = json.loads(config.read_text())["task"]["name"]
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if task_name in out:
            continue
        trajectory = config.parent / "agent" / "trajectory.json"
        if not trajectory.exists():
            continue
        try:
            steps = json.loads(trajectory.read_text()).get("steps") or []
        except (OSError, json.JSONDecodeError):
            continue
        for step in steps[:1]:
            text = _tb_description(step.get("message") or "")
            if text:
                out[task_name] = text
    return out


# --------------------------------------------------------------------------- AIME 2026

def load_aime_tasks() -> dict[str, str]:
    """problem_idx -> problem statement."""
    sys.path.insert(0, str(REPO_ROOT / "aime_2026"))
    with _unshadowed_datasets():
        import aime  # noqa: PLC0415

        return {str(row["problem_idx"]): row["problem"] for row in aime.load_problems()}


# --------------------------------------------------------------------------- GPQA Diamond

def load_gpqa_tasks() -> dict[str, str]:
    """question_idx -> the question and its four options, in attempt 0's shuffle.

    The answer-format instruction at the head of simple-evals' template is dropped: it tells the
    model how to answer, which contradicts the router prompt's "do not attempt the task".
    """
    sys.path.insert(0, str(REPO_ROOT / "gpqa_diamond"))
    import gpqa  # noqa: PLC0415

    questions = gpqa.load_questions()
    table = gpqa.permutation_table(1)
    # The template is <instruction>\n\n<question>\n\nA) ...; strip the first paragraph only if
    # it is the instruction.
    head = gpqa.QUERY_TEMPLATE.split("\n\n", 1)[0]
    out = {}
    for q in questions:
        prompt, _, _ = gpqa.build_prompt(q, table[(q["question_idx"], 0)])
        out[str(q["question_idx"])] = (prompt.split("\n\n", 1)[1].strip()
                                       if prompt.startswith(head) else prompt)
    return out


# --------------------------------------------------------------------------- registry

SOURCES: dict[str, Source] = {
    "swebench_pro": Source(
        key="swebench_pro", unit="instance",
        setting=("You will work on a real GitHub issue inside the repository's Docker container, "
                 "as an autonomous agent driving a shell one command at a time (the "
                 "mini-swe-agent scaffold). You get one attempt. You succeed only if the "
                 "repository's own fail-to-pass and pass-to-pass tests both pass on the patch "
                 "you leave behind."),
        load=load_pro_tasks),
    "terminal_bench": Source(
        key="terminal_bench", unit="task",
        setting=("You will solve a command-line task inside a Linux container, as an autonomous "
                 "agent sending batches of shell keystrokes to a terminal and reading the screen "
                 "back (the Terminus-2 scaffold). You get one attempt. You succeed only if the "
                 "task's own verifier script passes afterwards."),
        load=load_tb_tasks),
    "aime_2026": Source(
        key="aime_2026", unit="problem",
        setting=("You will answer this AIME competition problem in a single response, with no "
                 "tools, no code execution and no second chance to revise. The answer is an "
                 "integer between 0 and 999 and is marked right or wrong on that integer alone."),
        load=load_aime_tasks),
    "gpqa_diamond": Source(
        key="gpqa_diamond", unit="question",
        setting=("You will answer this graduate-level multiple-choice science question in a "
                 "single response, with no tools, no search and no second chance to revise. "
                 "Only the letter you pick is marked."),
        load=load_gpqa_tasks),
}


def tasks_for(source: str, refresh: bool = False) -> dict[str, str]:
    """Every unit's task text for one source, through the on-disk cache."""
    if source not in SOURCES:
        raise SystemExit(f"unknown task source {source!r}; known: {sorted(SOURCES)}")
    cache = CACHE_DIR / f"{source}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    tasks = SOURCES[source].load()
    if not tasks:
        raise SystemExit(f"no task text recovered for {source!r}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(tasks, indent=1, sort_keys=True))
    return tasks


# Long statements are cut in the middle, keeping the opening (the issue) and the closing (the
# requirements). Every selection records whether its task text was cut.
DEFAULT_MAX_CHARS = 12_000
ELISION = "\n\n[... {n:,} characters of this task description omitted ...]\n\n"


def clip(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    """(text at most `max_chars` long, whether anything was cut)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.6)
    tail = max_chars - head
    cut = len(text) - head - tail
    return text[:head] + ELISION.format(n=cut) + text[-tail:], True
