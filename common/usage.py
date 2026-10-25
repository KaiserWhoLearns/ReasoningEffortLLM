"""Token accounting recovered from a mini-swe-agent trajectory.

Sums the provider usage object of every assistant call; dollars are derived from these counts
by common/pricing.py.

Token terminology follows the provider usage objects:
    output / completion_tokens   inclusive generated total, what output is billed on.
    reasoning / thinking         a SUBSET of output, never an addition to it.
    input / prompt_tokens        total input, inclusive of its cached part.

Which reasoning-token field is used, per provider:
    Anthropic native   usage.output_tokens_details.thinking_tokens, metered. Present only on
                       the native-SDK path (common/anthropic_native_textbased.py).
    Anthropic/LiteLLM  completion_tokens_details.reasoning_tokens is NOT metered: LiteLLM
                       estimates it from the returned summary text and emits 0 when thinking
                       is omitted. Treated as unavailable.
    OpenAI, Together   completion_tokens_details.reasoning_tokens, metered.
`reasoning_source` records which rule applied, so "0 reasoning tokens" is distinguishable
from "no count available". `calls_with_reasoning` counts the calls that produced reasoning
tokens; on a `none` run it should be 0.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# reasoning_source values
NATIVE = "anthropic_native"      # Anthropic's own thinking_tokens breakdown
METERED = "provider_metered"     # OpenAI / Together reasoning_tokens
UNAVAILABLE = None               # Anthropic via LiteLLM: field exists but is an estimate

COUNT_FIELDS = ("input_tokens", "cached_input_tokens", "cache_creation_tokens",
                "output_tokens", "reasoning_tokens", "total_tokens")


def _is_anthropic_response(model_name: str) -> bool:
    """Name-based, so this module needs no registry lookup and never raises on a new id."""
    m = (model_name or "").lower()
    return "claude" in m or m.startswith("anthropic/")


def from_trajectory(path: Path) -> dict[str, Any]:
    """Sum every provider usage object across the assistant calls in one trajectory."""
    data = json.loads(Path(path).read_text())
    result: dict[str, Any] = {f: 0 for f in COUNT_FIELDS}
    result.update(n_calls=0, calls_with_reasoning=0, reasoning_available=False,
                  reasoning_source=UNAVAILABLE, has_thinking_blocks=False)
    for message in data.get("messages", []):
        if message.get("role") != "assistant":
            continue
        response = (message.get("extra") or {}).get("response") or {}
        usage = response.get("usage") or {}
        if not usage:
            continue
        result["n_calls"] += 1
        result["input_tokens"] += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        result["output_tokens"] += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0

        input_details = usage.get("prompt_tokens_details", usage.get("input_tokens_details", {})) or {}
        result["cached_input_tokens"] += (
            input_details.get("cached_tokens", usage.get("cache_read_input_tokens", 0)) or 0
        )
        # Anthropic only (native path); billed at 1.25x input, so kept apart from cache reads.
        result["cache_creation_tokens"] += usage.get("cache_creation_input_tokens", 0) or 0

        output_details = usage.get("output_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        call_reasoning = None
        if "thinking_tokens" in output_details:
            call_reasoning = output_details["thinking_tokens"] or 0
            result["reasoning_source"] = NATIVE
        elif ("reasoning_tokens" in completion_details
                and not _is_anthropic_response(response.get("model", ""))):
            call_reasoning = completion_details["reasoning_tokens"] or 0
            # Keep NATIVE if an earlier call had it.
            if result["reasoning_source"] is UNAVAILABLE:
                result["reasoning_source"] = METERED
        if call_reasoning is not None:
            result["reasoning_tokens"] += call_reasoning
            result["calls_with_reasoning"] += bool(call_reasoning)

        choices = response.get("choices") or []
        response_message = (choices[0].get("message") or {}) if choices else {}
        if response_message.get("thinking_blocks"):
            result["has_thinking_blocks"] = True
    result["total_tokens"] = int(result["input_tokens"]) + int(result["output_tokens"])
    result["reasoning_available"] = result["reasoning_source"] is not UNAVAILABLE
    return result
