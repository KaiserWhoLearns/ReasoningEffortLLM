"""GPQA Diamond: dataset, prompt, grading, API calls and sample I/O for run_eval and report.

198 four-way multiple-choice science questions. One attempt is one API call with no tools: a
single user message in, one reply out, graded by its `Answer: <letter>` line.

    Dataset    the Diamond CSV OpenAI publishes for simple-evals (ungated, no token needed),
               cached under <data>/datasets/gpqa_diamond/.
    Prompt     openai/simple-evals QUERY_TEMPLATE_MULTICHOICE, sent as a single user message
               with no system prompt.
    Shuffle    the answer choices are reordered independently on every attempt, drawn from
               `random.Random(0)` in simple-evals' order (see `permutation_table`).
    Grading    simple-evals ANSWER_PATTERN_MULTICHOICE, first match. A provider refusal has no
               answer and counts as incorrect.
    Effort     OpenAI: Responses API `reasoning.effort`. Anthropic: `none` disables thinking,
               every other level is adaptive thinking + `output_config.effort`. Together:
               `reasoning_effort` in the request body (`models.together_effort_body`).
               No sampling parameters are sent; current Anthropic models reject them.

Each provider is called through its own SDK rather than LiteLLM, which drops Anthropic's
thinking-token count. Samples store the provider's token counts; dollars are derived from
PRICES at report time.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from common import models, paths

# ---------------------------------------------------------------- dataset/prompt

CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
DATASET_DIR = paths.DATASETS_ROOT / "gpqa_diamond"
N_QUESTIONS = 198
LETTERS = "ABCD"

# Verbatim from openai/simple-evals `common.py` QUERY_TEMPLATE_MULTICHOICE.
QUERY_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def dataset_csv(cache_dir: Path = DATASET_DIR) -> Path:
    """The Diamond CSV, downloaded on first use."""
    path = Path(cache_dir) / "gpqa_diamond.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CSV_URL, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def load_questions(cache_dir: Path = DATASET_DIR, limit: int | None = None,
                   questions: list[int] | None = None) -> list[dict]:
    """The 198 questions as dicts with question_idx (1-based, CSV order).

    `--limit` takes the first n rows, which keeps the choice shuffles identical to a full run.
    """
    with dataset_csv(cache_dir).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != N_QUESTIONS:
        raise SystemExit(f"expected {N_QUESTIONS} rows in the Diamond CSV, read {len(rows)}")

    out = []
    for i, row in enumerate(rows, start=1):
        out.append({
            "question_idx": i,
            "record_id": row.get("Record ID", ""),
            "question": row["Question"],
            "correct": row["Correct Answer"],
            "incorrect": [row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                          row["Incorrect Answer 3"]],
            "domain": row.get("High-level domain", ""),
            "subdomain": row.get("Subdomain", ""),
        })
    if questions:
        wanted = set(questions)
        out = [r for r in out if r["question_idx"] in wanted]
        missing = wanted - {r["question_idx"] for r in out}
        if missing:
            raise SystemExit(f"no such question_idx: {sorted(missing)}")
    if limit is not None:
        out = out[:limit]
    return out


def permutation_table(n_runs: int) -> dict[tuple[int, int], list[int]]:
    """(question_idx, run_idx) -> the choice order, drawn as simple-evals draws it.

    simple-evals repeats the question list n_repeats times and draws one
    `rng.sample(range(4), 4)` per copy from `random.Random(0)`, so the draws are repeat-major:
    every question once, then every question again. The first r*198 draws therefore do not
    depend on n_runs, and raising --n leaves existing attempts' orders unchanged.
    """
    rng = random.Random(0)
    table: dict[tuple[int, int], list[int]] = {}
    for run_idx in range(n_runs):
        for question_idx in range(1, N_QUESTIONS + 1):
            table[(question_idx, run_idx)] = rng.sample(range(4), 4)
    return table


def build_prompt(question: dict, permutation: list[int]) -> tuple[str, list[str], str]:
    """(single user message, the four choices as presented, the correct letter).

    `permutation` indexes [correct, wrong1, wrong2, wrong3], simple-evals' ordering.
    """
    pool = [question["correct"], *question["incorrect"]]
    choices = [pool[i] for i in permutation]
    correct_letter = LETTERS[choices.index(question["correct"])]
    prompt = QUERY_TEMPLATE.format(Question=question["question"], A=choices[0],
                                   B=choices[1], C=choices[2], D=choices[3])
    return prompt, choices, correct_letter


# ---------------------------------------------------------------------- grading

# openai/simple-evals ANSWER_PATTERN_MULTICHOICE, verbatim.
ANSWER_PATTERN = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")


def extract_answer(text: str) -> tuple[str | None, str]:
    """(letter, how): the first `Answer: X` in the text, as simple-evals' `re.search` finds it."""
    if not text:
        return None, "empty"
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper(), "answer_line"
    return None, "no_answer_line"


def is_refusal(sample: dict) -> bool:
    """Did the provider decline to answer (rather than answer wrongly or unreadably)?

    Anthropic sets `stop_reason: "refusal"` and returns an empty body; OpenAI and Together
    report a content-filter stop, in the finish string or in `raw`. `raw` may be absent from a
    record, so the finish string is checked on its own first.
    """
    finish = (sample.get("finish") or "").lower()
    if finish in {"refusal", "content_filter"}:
        return True
    raw = " ".join(str(v) for v in (sample.get("raw") or {}).values()).lower()
    return "content_filter" in raw or "refusal" in raw


# ---------------------------------------------------------------------- pricing

# USD per token, from the providers' published rates (Anthropic and OpenAI pricing pages;
# Together's GET /v1/models `pricing` block). Cache reads are carried for completeness; GPQA
# prompts are short and distinct, so almost nothing caches.
PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input": 3.0e-6, "output": 15.0e-6, "cache_read": 0.3e-6},
    "gpt-5.6-sol": {"input": 5.0e-6, "output": 30.0e-6, "cache_read": 0.5e-6},
    "gpt-5.6-terra": {"input": 2.0e-6, "output": 12.0e-6, "cache_read": 0.2e-6},
    "gpt-5.6-luna": {"input": 0.2e-6, "output": 1.2e-6, "cache_read": 0.02e-6},
    "glm-5.2": {"input": 1.4e-6, "output": 4.4e-6, "cache_read": 0.26e-6},
    "inkling": {"input": 1.0e-6, "output": 4.05e-6, "cache_read": 0.17e-6},
    "kimi-k3": {"input": 3.0e-6, "output": 15.0e-6, "cache_read": 0.3e-6},
    "deepseek-v4-flash": {"input": 0.14e-6, "output": 0.28e-6, "cache_read": 0.03e-6},
}


def cost_of(model: str, usage: dict) -> float | None:
    """tokens x PRICES, or None when the model has no rates or the tokens are missing.

    `input_tokens` includes the cached part, which is re-priced at the cache rate.
    """
    rate = PRICES.get(models.name_slug(model))
    if rate is None or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        return None
    cached = usage.get("cached_input_tokens") or 0
    fresh = max(int(usage["input_tokens"]) - cached, 0)
    return (fresh * rate["input"]
            + cached * rate.get("cache_read", rate["input"])
            + int(usage["output_tokens"]) * rate["output"])


# ------------------------------------------------------------------------ paths

DEFAULT_OUT_ROOT = paths.data_root("gpqa-diamond")
SAMPLES_FILE = "samples.jsonl"
META_FILE = "run_meta.json"


def run_dir(out_root: Path, model: str, effort: str) -> Path:
    """<out_root>/<model>/<effort>, the directory one arm writes into."""
    return Path(out_root) / models.name_slug(model) / effort


# ------------------------------------------------------------------- API calls

TOGETHER_BASE_URL = "https://api.together.xyz/v1"


def provider_id(spec: models.ModelSpec) -> str:
    """The model id the provider's own SDK expects (the litellm id without its prefix)."""
    return spec.litellm_id.split("/", 1)[1]


@dataclass
class Reply:
    text: str
    usage: dict[str, Any]
    finish: str                       # provider's stop/status string
    truncated: bool                   # hit the output cap
    reasoning_text: str | None = None  # only when the provider returns one
    raw: dict[str, Any] = field(default_factory=dict)


def call_openai(client, model: models.ModelSpec, prompt: str, effort: str,
                max_tokens: int) -> Reply:
    """One Responses API call."""
    response = client.responses.create(
        model=provider_id(model),
        input=prompt,
        max_output_tokens=max_tokens,
        reasoning={"effort": effort},
    )
    usage = response.usage
    details_in = getattr(usage, "input_tokens_details", None)
    details_out = getattr(usage, "output_tokens_details", None)
    return Reply(
        text=response.output_text or "",
        usage={
            "input_tokens": getattr(usage, "input_tokens", None),
            "cached_input_tokens": getattr(details_in, "cached_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", None),
            # A subset of output_tokens.
            "reasoning_tokens": getattr(details_out, "reasoning_tokens", None),
        },
        finish=response.status or "",
        truncated=response.status == "incomplete",
        raw={"incomplete_details": str(response.incomplete_details or "")},
    )


def call_anthropic(client, model: models.ModelSpec, prompt: str, effort: str, max_tokens: int,
                   timeout: float = 3600.0) -> Reply:
    """One Messages API call.

    Non-streaming, because the streamed final message carries no thinking-token count; the
    explicit `timeout` is what the SDK requires for a non-streaming call this large.
    """
    kwargs: dict[str, Any] = {
        "model": provider_id(model),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": timeout,
    }
    if effort == "none":
        kwargs["thinking"] = {"type": "disabled"}
    else:
        kwargs["thinking"] = {"type": "adaptive", "display": "omitted"}
        kwargs["output_config"] = {"effort": effort}
    message = client.messages.create(**kwargs)

    text = "".join(b.text for b in message.content if b.type == "text")
    thinking = "".join(getattr(b, "thinking", "") or ""
                       for b in message.content if b.type == "thinking")
    usage = message.usage
    details_out = getattr(usage, "output_tokens_details", None)
    return Reply(
        text=text,
        usage={
            "input_tokens": getattr(usage, "input_tokens", None),
            "cached_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", None),
            # A subset of output_tokens.
            "reasoning_tokens": getattr(details_out, "thinking_tokens", None),
        },
        finish=message.stop_reason or "",
        truncated=message.stop_reason == "max_tokens",
        reasoning_text=thinking or None,
        raw={"stop_reason": message.stop_reason or ""},
    )


def call_together(client, model: models.ModelSpec, prompt: str, effort: str,
                  max_tokens: int) -> Reply:
    """One chat-completions call to Together's OpenAI-compatible endpoint.

    The effort travels in `extra_body` so it reaches the provider verbatim (`max` is not an
    OpenAI effort value).
    """
    response = client.chat.completions.create(
        model=provider_id(model),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body=models.together_effort_body(model.slug, effort),
    )
    choice = response.choices[0]
    message = choice.message
    usage = response.usage
    details_in = getattr(usage, "prompt_tokens_details", None)
    details_out = getattr(usage, "completion_tokens_details", None)
    # The reasoning trace arrives on a non-standard field whose name depends on the serving
    # stack; some models return none at all while still metering reasoning tokens.
    extra = getattr(message, "model_extra", None) or {}
    return Reply(
        text=message.content or "",
        usage={
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "cached_input_tokens": getattr(details_in, "cached_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", None),
            # A subset of output_tokens.
            "reasoning_tokens": getattr(details_out, "reasoning_tokens", None),
        },
        finish=choice.finish_reason or "",
        truncated=choice.finish_reason == "length",
        reasoning_text=extra.get("reasoning") or extra.get("reasoning_content") or None,
        raw={"finish_reason": choice.finish_reason or ""},
    )


def call(client, model: models.ModelSpec, prompt: str, effort: str, max_tokens: int) -> Reply:
    """One attempt, dispatched on the model's provider family."""
    if model.family == "openai":
        return call_openai(client, model, prompt, effort, max_tokens)
    if model.family == "together":
        return call_together(client, model, prompt, effort, max_tokens)
    return call_anthropic(client, model, prompt, effort, max_tokens)


def make_client(model: models.ModelSpec):
    """The provider SDK client, with SDK retries off (with_retries handles them)."""
    paths.load_env([model.key_env])
    if model.family == "together":
        from openai import OpenAI
        return OpenAI(base_url=TOGETHER_BASE_URL, api_key=os.environ[model.key_env],
                      timeout=3600, max_retries=0)
    if model.family == "openai":
        from openai import OpenAI
        return OpenAI(timeout=3600, max_retries=0)
    from anthropic import Anthropic
    return Anthropic(timeout=3600, max_retries=0)


def with_retries(fn: Callable[[], Reply], attempts: int = 5,
                 on_error: Callable[[int, Exception], None] | None = None) -> Reply:
    """Up to `attempts` tries with jittered exponential backoff."""
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:                      # provider-agnostic
            if attempt == attempts:
                raise
            if on_error:
                on_error(attempt, exc)
            time.sleep(delay + random.uniform(0, delay / 2))
            delay = min(delay * 2, 120.0)
    raise AssertionError("unreachable")


# ------------------------------------------------------------------- sample I/O

def read_samples(path: Path) -> list[dict]:
    """Every sample record in a samples.jsonl, skipping a partially written line."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append_sample(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
