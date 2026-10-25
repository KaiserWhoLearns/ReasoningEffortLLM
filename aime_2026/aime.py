"""AIME 2026: dataset, prompt, grading, API calls and sample I/O for run_eval.py and report.py.

One attempt is one API call with no tools: a single user message in, one reply out, graded by
the integer in its last \\boxed{}.

    Prompt     MathArena (github.com/eth-sri/matharena): the instruction string of
               `configs/competitions/aime/aime_2026.yaml`, then "\\n\\n" and the problem, sent as
               a single user message with no system prompt (`solvers/pure_model_solver.py`).
    Grading    the last \\boxed{} in the response, falling back to the last integer anywhere in
               it (MathArena's `strict_parsing: false`; OpenAI's gpt-oss `aime_eval.py` does
               the same).
    Effort     OpenAI: Responses API `reasoning.effort`. Anthropic: `none` disables thinking,
               every other level is adaptive thinking + `output_config.effort`. Together:
               `reasoning_effort` in the request body (`models.together_effort_body`).
               No sampling parameters are sent; current Anthropic models reject them.

Each provider is called through its own SDK rather than LiteLLM, which drops Anthropic's
thinking-token count. Samples store the provider's token counts; dollars are derived from
PRICES at report time.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from common import models, paths

# ---------------------------------------------------------------- dataset/prompt

HF_DATASET = "MathArena/aime_2026"   # 30 problems, integer answers 0-999
N_PROBLEMS = 30

# MathArena's instruction as the model sees it (the YAML doubles the braces for str.format).
INSTRUCTION = (
    "Put your final answer within \\boxed{}.\n"
    "The answer is an integer between 0 and 999 inclusive."
)


def build_prompt(problem: str) -> str:
    """The single user message."""
    return f"{INSTRUCTION}\n\n{problem}"


def load_problems(limit: int | None = None, problems: list[int] | None = None) -> list[dict]:
    """The 30 problems as dicts with problem_idx (1-based), problem, answer (int)."""
    from datasets import load_dataset  # only the run stage needs it

    rows = [dict(r) for r in load_dataset(HF_DATASET, split="train")]
    rows.sort(key=lambda r: r["problem_idx"])
    if problems:
        wanted = set(problems)
        rows = [r for r in rows if r["problem_idx"] in wanted]
        missing = wanted - {r["problem_idx"] for r in rows}
        if missing:
            raise SystemExit(f"no such problem_idx: {sorted(missing)}")
    if limit is not None:
        rows = rows[:limit]
    return rows


# ---------------------------------------------------------------------- grading

_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")


def _last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}/\\fbox{...}, with nested braces balanced by scanning."""
    starts = [m.end() for m in _BOXED.finditer(text)]
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
    return None


def _to_int(s: str) -> int | None:
    """Strip LaTeX decoration and read one integer."""
    s = s.strip()
    for cmd in (r"\text", r"\mathrm", r"\textbf", r"\mathbf"):
        s = s.replace(cmd, "")
    s = s.replace("$", "").replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("{", "").replace("}", "").replace(",", "").replace("_", "")
    s = s.strip().rstrip(".")
    m = re.fullmatch(r"[+-]?\d+", s)
    return int(m.group(0)) if m else None


def extract_answer(text: str) -> tuple[int | None, str]:
    """(answer, how): the graded integer and the rule that produced it.

    The last \\boxed{} first; without one, the last integer anywhere in the response.
    """
    if not text:
        return None, "empty"
    boxed = _last_boxed(text)
    if boxed is not None:
        value = _to_int(boxed)
        if value is not None:
            return value, "boxed"
        return None, "boxed_unparsed"
    ints = re.findall(r"\d+", text)
    if ints:
        return int(ints[-1]), "last_int"
    return None, "none"


# ---------------------------------------------------------------------- pricing

# USD per token, from the providers' published rates (Anthropic and OpenAI pricing pages;
# Together's GET /v1/models `pricing` block). Cache reads are carried for completeness; AIME
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

DEFAULT_OUT_ROOT = paths.data_root("aime-2026")
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
