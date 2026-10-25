"""One self-routing selection: ask a model which reasoning effort it wants for one task.

A selection is one API call. The prompt holds the task text, the effort levels the model may
choose between (its routing pool on that benchmark, see `vanilla.pools_for`) and one paragraph
describing the scaffold. Nothing is attempted: the routed run is priced from the sweeps on disk.

The selection call runs at `low` effort, or at the cheapest level a model has when it has no
`low` (GLM-5.2: `high`). Its cost is recorded and charged to the policy.

Calls use the vendors' SDKs directly. For Together the effort is sent through `extra_body`
(`common.models.together_effort_body`), as in the sweeps.

Records are appended to DATA_ROOT/self_routing/<source>/<model>.jsonl; calls that failed after
every retry go to <model>.errors.jsonl beside it and are re-asked on the next run.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import models, paths, pricing  # noqa: E402

SELECTION_DIR = paths.DATA_ROOT / "self_routing"

# The reply is one line; the cap only stops a runaway reasoning chain. A call that hits it is
# recorded as truncated.
DEFAULT_MAX_TOKENS = 16384

JUDGE_EFFORT = "low"
# Stored in every record; the only prompt variant.
VARIANT = "plain"


def _effort_key(effort: str) -> float:
    return models.EFFORT_ORDER.get(effort, 99.0)


def build_menu(pool: list[str]) -> tuple[list[str], dict[str, str]]:
    """(option names shown to the model, option name -> pooled effort level), in ladder order."""
    names = sorted(pool, key=_effort_key)
    return names, {name: name for name in names}


def judge_effort_for(model: str) -> str:
    """The level the selection call runs at: `low`, or the model's cheapest level above `none`."""
    valid = [e for e in models.valid_efforts(model) if e != "none"]
    return JUDGE_EFFORT if JUDGE_EFFORT in valid else min(valid, key=_effort_key)


# --------------------------------------------------------------------------- the prompt

LADDER_GLOSS = {
    "low": "the least thinking — fastest and cheapest",
    "medium": "moderate thinking",
    "high": "extended thinking",
    "xhigh": "the most thinking this model offers — slowest and dearest",
    "max": "the most thinking this model offers — slowest and dearest",
}

ANSWER_LINE = re.compile(r"(?im)^\s*EFFORT\s*[:=]\s*[`\"']?([a-z_]+)")


def build_prompt(setting: str, unit: str, task: str, menu: list[str]) -> str:
    """The single user message one selection sends (the paper's prompt table)."""
    lines = [f"  {name} — {LADDER_GLOSS.get(name, 'more thinking than the level below')}"
             for name in menu]
    return (
        f"You are about to be given one {unit} to solve on your own.\n\n"
        f"Setting: {setting}\n\n"
        f"Before you start, choose the reasoning effort you will run at. Your options, from "
        f"least to most thinking:\n"
        + "\n".join(lines) + "\n\n"
        "Thinking longer costs more money and more time, and on a hard task it is often what "
        "makes the difference between solving it and not. Thinking longer on an easy task buys "
        "nothing. Choose the LOWEST level you believe will still let you solve this particular "
        f"{unit}.\n\n"
        "Do not attempt the task now. Do not explain your choice.\n\n"
        f"--- BEGIN {unit.upper()} ---\n{task}\n--- END {unit.upper()} ---\n\n"
        "Reply with exactly one line, nothing else:\n"
        f"EFFORT: <one of: {', '.join(menu)}>"
    )


def parse_choice(text: str, menu: list[str]) -> tuple[str | None, str]:
    """(the level chosen, the rule that read it), or (None, reason) when no menu level is named.

    Rules, in order: the last `EFFORT:` line naming a menu level ("line"), else the last menu
    word anywhere in the reply ("scan").
    """
    if not text:
        return None, "empty"
    on_menu = {name.lower() for name in menu}
    matches = ANSWER_LINE.findall(text)
    for value in reversed(matches):
        if value.lower() in on_menu:
            return value.lower(), "line"
    hits = [(m.start(), m.group(0).lower())
            for m in re.finditer(r"(?i)\b(" + "|".join(re.escape(n) for n in menu) + r")\b", text)]
    if hits:
        return hits[-1][1], "scan"
    return None, "unparsed"


# --------------------------------------------------------------------------- provider calls

@dataclass
class Reply:
    text: str
    usage: dict[str, Any]
    finish: str
    truncated: bool
    raw: dict[str, Any] = field(default_factory=dict)


def call_openai(client, model: str, prompt: str, effort: str, max_tokens: int) -> Reply:
    """One Responses API call; no sampling parameters are sent."""
    response = client.responses.create(
        model=models.api_id(model),
        input=prompt,
        max_output_tokens=max_tokens,
        reasoning={"effort": effort},
    )
    usage = response.usage
    details_in = getattr(usage, "input_tokens_details", None)
    details_out = getattr(usage, "output_tokens_details", None)
    return Reply(
        text=response.output_text or "",
        usage={"input_tokens": getattr(usage, "input_tokens", None),
               "cached_input_tokens": getattr(details_in, "cached_tokens", 0) or 0,
               "output_tokens": getattr(usage, "output_tokens", None),
               "reasoning_tokens": getattr(details_out, "reasoning_tokens", None)},
        finish=response.status or "",
        truncated=response.status == "incomplete",
        raw={"incomplete_details": str(response.incomplete_details or "")})


def call_anthropic(client, model: str, prompt: str, effort: str, max_tokens: int) -> Reply:
    """One Messages API call, non-streaming with an explicit timeout.

    Streaming drops the thinking-token count, and a large `max_tokens` without an explicit
    timeout trips the SDK's ten-minute guard. `none` requires thinking={'type': 'disabled'}.
    """
    kwargs: dict[str, Any] = {
        "model": models.api_id(model),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": 900.0,
    }
    if effort == "none":
        kwargs["thinking"] = {"type": "disabled"}
    else:
        kwargs["thinking"] = {"type": "adaptive", "display": "omitted"}
        kwargs["output_config"] = {"effort": effort}
    message = client.messages.create(**kwargs)
    usage = message.usage
    details_out = getattr(usage, "output_tokens_details", None)
    return Reply(
        text="".join(b.text for b in message.content if b.type == "text"),
        usage={"input_tokens": getattr(usage, "input_tokens", None),
               "cached_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
               "output_tokens": getattr(usage, "output_tokens", None),
               "reasoning_tokens": getattr(details_out, "thinking_tokens", None)},
        finish=message.stop_reason or "",
        truncated=message.stop_reason == "max_tokens",
        raw={"stop_reason": message.stop_reason or ""})


TOGETHER_BASE_URL = "https://api.together.xyz/v1"


def together_model_id(model: str) -> str:
    """Together's own model name: the litellm id without its `together_ai/` prefix."""
    return models.resolve(model).litellm_id.removeprefix("together_ai/")


def call_together(client, model: str, prompt: str, effort: str, max_tokens: int) -> Reply:
    """One Together chat-completions call, with the effort in `extra_body`."""
    response = client.chat.completions.create(
        model=together_model_id(model),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body=models.together_effort_body(model, effort))
    choice = response.choices[0]
    usage = response.usage
    details_in = getattr(usage, "prompt_tokens_details", None)
    details_out = getattr(usage, "completion_tokens_details", None)
    return Reply(
        text=choice.message.content or "",
        usage={"input_tokens": getattr(usage, "prompt_tokens", None),
               "cached_input_tokens": getattr(details_in, "cached_tokens", 0) or 0,
               "output_tokens": getattr(usage, "completion_tokens", None),
               "reasoning_tokens": getattr(details_out, "reasoning_tokens", None)},
        finish=choice.finish_reason or "",
        truncated=choice.finish_reason == "length",
        raw={"finish_reason": choice.finish_reason or ""})


def make_client(model: str):
    spec = models.resolve(model)
    paths.load_env([spec.key_env])
    if spec.family == "together":
        from openai import OpenAI  # noqa: PLC0415
        return OpenAI(base_url=TOGETHER_BASE_URL, api_key=os.environ[spec.key_env],
                      timeout=900, max_retries=0)
    if spec.family == "openai":
        from openai import OpenAI  # noqa: PLC0415
        return OpenAI(timeout=900, max_retries=0)
    from anthropic import Anthropic  # noqa: PLC0415
    return Anthropic(timeout=900, max_retries=0)


def call(client, model: str, prompt: str, effort: str, max_tokens: int) -> Reply:
    family = models.resolve(model).family
    if family == "openai":
        return call_openai(client, model, prompt, effort, max_tokens)
    if family == "anthropic":
        return call_anthropic(client, model, prompt, effort, max_tokens)
    return call_together(client, model, prompt, effort, max_tokens)


def with_retries(fn: Callable[[], Reply], attempts: int = 4) -> Reply:
    """Jittered exponential backoff."""
    delay = 4.0
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception:                               # provider-agnostic
            if attempt == attempts:
                raise
            time.sleep(delay + random.uniform(0, delay / 2))
            delay = min(delay * 2, 90.0)
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- records

def record_path(source: str, model: str) -> Path:
    return SELECTION_DIR / source / f"{models.name_slug(model)}.jsonl"


def error_path(source: str, model: str) -> Path:
    """Calls that failed after every retry. Counted as unanswered and re-asked on the next run."""
    return SELECTION_DIR / source / f"{models.name_slug(model)}.errors.jsonl"


def is_scored(record: dict, menu: list[str], judge_effort: str) -> bool:
    """Whether a stored record is a paper-configuration selection for this menu.

    Selection files may also hold other prompt variants, judge efforts, menus or repeated draws
    of the same question; only the first draw of the plain prompt at the model's judge effort,
    over exactly this menu, is scored.
    """
    return (record.get("variant") == VARIANT and record.get("judge_effort") == judge_effort
            and list(record.get("menu") or []) == list(menu) and record.get("draw", 0) == 0)


def error_record(*, model: str, source: str, unit: str, menu: list[str], judge_effort: str,
                 exc: BaseException) -> dict:
    return {
        "source": source, "unit": unit, "model": models.name_slug(model), "menu": menu,
        "variant": VARIANT, "judge_effort": judge_effort, "draw": 0, "choice": None,
        "parsed_by": "api_error", "error": f"{type(exc).__name__}: {str(exc)[:400]}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def read_records(path: Path) -> list[dict]:
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
            continue                                    # interrupted write
    return out


_WRITE_LOCK = threading.Lock()


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def select_one(client, *, model: str, source: str, unit: str, task: str, unit_noun: str,
               setting: str, menu: list[str], to_level: dict[str, str], judge_effort: str,
               max_task_chars: int, max_tokens: int) -> dict:
    """One selection, as the record it is stored as. Raises only if every retry failed."""
    from tasks import clip  # noqa: PLC0415

    text, cut = clip(task, max_task_chars)
    prompt = build_prompt(setting, unit_noun, text, menu)
    started = time.time()
    reply = with_retries(lambda: call(client, model, prompt, judge_effort, max_tokens))
    name, how = parse_choice(reply.text, menu)
    return {
        "source": source, "unit": unit, "model": models.name_slug(model),
        "menu": menu, "variant": VARIANT, "judge_effort": judge_effort, "draw": 0,
        "choice": name, "chosen_effort": to_level.get(name) if name else None,
        "parsed_by": how,
        "cost_usd": pricing.cost_from_usage(
            model, prompt_tokens=reply.usage.get("input_tokens"),
            output_tokens=reply.usage.get("output_tokens"),
            cache_read=reply.usage.get("cached_input_tokens")),
        "usage": reply.usage, "finish": reply.finish, "truncated": reply.truncated,
        "latency_s": round(time.time() - started, 2),
        "prompt_chars": len(prompt), "task_truncated": cut,
        "reply": reply.text[:2000],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
