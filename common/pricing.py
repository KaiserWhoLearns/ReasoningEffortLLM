"""API rates and cost arithmetic, shared by every benchmark.

Runs record token counts (common/usage.py); every dollar figure is tokens x this table, so a
rate change is a re-report, never a re-run. The table is used to

  1. register prices with LiteLLM (`register_litellm()` for mini-swe-agent, the Terminus-2
     `model_info` kwarg for harbor), whose bundled registry lacks some of these ids and
     prices others differently (see `litellm_conflicts`);
  2. recompute a run's cost from its token counts (`cost_from_usage`).

Entries use LiteLLM's `model_info` schema (USD per token), keyed by the provider-prefixed
litellm id, so they can be passed to `litellm.register_model` unchanged.

Rate sources (USD per million tokens, input / output)
    Anthropic   Claude pricing page. Sonnet 5 at its standard rate, $3 / $15; cache reads
                0.1x input, cache writes 1.25x input.
    OpenAI      developers.openai.com/api/docs/pricing. Sol $5 / $30, Terra $2 / $12,
                Luna $0.20 / $1.20; cache reads 0.1x input, no cache-write premium.
    Together    the `pricing` block of Together's `GET /v1/models`. GLM-5.2 $1.40 / $4.40
                (cache read $0.26), Inkling $1.00 / $4.05 ($0.17), Kimi K3 $3.00 / $15.00
                ($0.30), DeepSeek V4 Flash $0.14 / $0.28 ($0.03). Together caches prompts
                automatically and bills no cache-write premium.
"""
from __future__ import annotations

import math

from . import models


def _openai(in_per_mtok: float, out_per_mtok: float, **extra) -> dict:
    """An OpenAI entry from its published $/Mtok; cache reads are billed at 0.1x input."""
    return {
        "litellm_provider": "openai", "mode": "chat",
        "input_cost_per_token": in_per_mtok / 1e6,
        "output_cost_per_token": out_per_mtok / 1e6,
        "cache_read_input_token_cost": in_per_mtok / 10 / 1e6,
        **extra,
    }


_GPT56_LIMITS = {"max_input_tokens": 1_050_000, "max_output_tokens": 128_000}

PRICE_OVERRIDES: dict[str, dict] = {
    "anthropic/claude-sonnet-5": {
        "litellm_provider": "anthropic", "mode": "chat",
        "input_cost_per_token": 3.0e-6, "output_cost_per_token": 15.0e-6,
        "cache_read_input_token_cost": 0.3e-6, "cache_creation_input_token_cost": 3.75e-6,
        "max_input_tokens": 1_000_000, "max_output_tokens": 128_000,
    },
    "openai/gpt-5.6-sol": _openai(5.0, 30.0, **_GPT56_LIMITS),
    "openai/gpt-5.6-terra": _openai(2.0, 12.0, **_GPT56_LIMITS),
    "openai/gpt-5.6-luna": _openai(0.2, 1.2, **_GPT56_LIMITS),
    # Together models carry no `max_output_tokens`: Together does not publish one, and each
    # harness sets its own generation limit.
    "together_ai/zai-org/GLM-5.2": {
        "litellm_provider": "together_ai", "mode": "chat",
        "input_cost_per_token": 1.4e-6, "output_cost_per_token": 4.4e-6,
        "cache_read_input_token_cost": 0.26e-6,
        "max_input_tokens": 512_000,
    },
    # `supports_function_calling` is required for the models that use mini-swe-agent's native
    # tool calling: with `drop_params: True`, LiteLLM forwards `tools` only to models its
    # registry marks as supporting them.
    "together_ai/thinkingmachines/Inkling": {
        "litellm_provider": "together_ai", "mode": "chat",
        "supports_function_calling": True, "supports_tool_choice": True,
        "input_cost_per_token": 1.0e-6, "output_cost_per_token": 4.05e-6,
        "cache_read_input_token_cost": 0.17e-6,
        "max_input_tokens": 524_288,
    },
    "together_ai/moonshotai/Kimi-K3": {
        "litellm_provider": "together_ai", "mode": "chat",
        "input_cost_per_token": 3.0e-6, "output_cost_per_token": 15.0e-6,
        "cache_read_input_token_cost": 0.3e-6,
        "max_input_tokens": 1_000_000,
    },
    "together_ai/deepseek-ai/DeepSeek-V4-Flash-0731": {
        "litellm_provider": "together_ai", "mode": "chat",
        "supports_function_calling": True, "supports_tool_choice": True,
        "input_cost_per_token": 0.14e-6, "output_cost_per_token": 0.28e-6,
        "cache_read_input_token_cost": 0.03e-6,
        "max_input_tokens": 1_048_576,
    },
}


def rates(model: str) -> dict | None:
    """Price entry for a model (slug or litellm id), or None if we have no rates for it."""
    if not model:
        return None
    if model in PRICE_OVERRIDES:
        return PRICE_OVERRIDES[model]
    try:
        return PRICE_OVERRIDES.get(models.resolve(model).litellm_id)
    except SystemExit:      # unknown provider -> certainly no rates
        return None


def rate_triplet(model: str) -> tuple[float, float, float]:
    """(input, cache_read, output) $/token."""
    r = rates(model)
    if r is None:
        raise SystemExit(f"no rates for {model!r}; add it to PRICE_OVERRIDES in common/pricing.py")
    return (r["input_cost_per_token"],
            r.get("cache_read_input_token_cost", r["input_cost_per_token"]),
            r["output_cost_per_token"])


def cost_from_usage(model: str, *, prompt_tokens: int | None, output_tokens: int | None,
                    cache_read: int | None = 0, cache_creation: int | None = 0) -> float | None:
    """Cost of a token breakdown at this table's rates. None if rates or tokens are missing.

    `prompt_tokens` is the TOTAL input, inclusive of its cached parts (the LiteLLM and harbor
    convention), and its composition is provider-specific:
        Anthropic  prompt = new + cache_read + cache_creation; creation billed at 1.25x input.
        OpenAI     prompt = new + cache_read; no separate write premium.
        Together   prompt = new + cache_read; no write premium.
    Pass `cache_creation` only when the source records it. Harbor's trial metrics store cache
    reads only, so a cost computed from them is a lower bound on Anthropic spend; the LiteLLM
    per-call capture records creation and gives the exact figure.
    """
    r = rates(model)
    if r is None or prompt_tokens is None or output_tokens is None:
        return None
    read = cache_read or 0
    creation = cache_creation or 0
    cache_read_rate = r.get("cache_read_input_token_cost", r["input_cost_per_token"])
    if models.resolve(model).family == "anthropic":
        uncached = max(prompt_tokens - read - creation, 0)
        creation_cost = creation * r.get("cache_creation_input_token_cost",
                                         1.25 * r["input_cost_per_token"])
    else:
        uncached = max(prompt_tokens - read, 0)
        creation_cost = 0.0
    return (uncached * r["input_cost_per_token"]
            + read * cache_read_rate
            + creation_cost
            + output_tokens * r["output_cost_per_token"])


def register_litellm() -> list[str]:
    """Register this table with LiteLLM's cost calculator; return the divergences it replaced.

    Models absent from LiteLLM's registry make `litellm.completion_cost` raise, which
    mini-swe-agent treats as fatal, so a sweep must call this before running. Each model is
    registered under both its litellm id and its bare slug, because mini calls LiteLLM with
    the `--model` string (`claude-sonnet-5`) while harbor uses the prefixed id. The
    divergences are computed before the overwrite, after which there is nothing to compare.
    """
    import litellm  # local: this module stays importable without litellm installed

    # register_model() warns about the provider of a bare slug; the entry itself names it.
    litellm.suppress_debug_info = True
    replaced = litellm_conflicts()
    entries: dict[str, dict] = {}
    for litellm_id, rate in PRICE_OVERRIDES.items():
        entries[litellm_id] = rate
        entries[models.resolve(litellm_id).slug] = rate
    litellm.register_model(entries)
    return replaced


def litellm_conflicts() -> list[str]:
    """Models this table prices differently from LiteLLM's current registry, as readable lines.

    Only meaningful before `register_litellm()`. Models absent from LiteLLM's registry are not
    reported.
    """
    import litellm

    out = []
    for litellm_id, rate in PRICE_OVERRIDES.items():
        for key in ("input_cost_per_token", "output_cost_per_token"):
            for name in (litellm_id, models.resolve(litellm_id).slug):
                theirs = (litellm.model_cost.get(name) or {}).get(key)
                # isclose: $/Mtok divided down to $/token can differ from a literal by a few ulps.
                if (theirs is not None and rate.get(key) is not None
                        and not math.isclose(theirs, rate[key], rel_tol=1e-9)):
                    out.append(f"{name} {key.replace('_cost_per_token', '')}: "
                               f"ours ${rate[key] * 1e6:g}/Mtok vs litellm ${theirs * 1e6:g}/Mtok")
    return out
