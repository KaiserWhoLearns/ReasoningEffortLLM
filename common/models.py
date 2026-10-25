"""Model registry and reasoning-effort levels, shared by every benchmark.

The legal effort levels depend on the model (family defaults, narrowed per model where the
model distinguishes fewer levels); the request parameter that sets them depends on the
scaffold, so each harness builds it:

    swebench_pro/run_sweep.py    mini-swe-agent: OpenAI `reasoning_effort`; Anthropic
                                 `output_config.effort` + adaptive thinking; Together
                                 `extra_body.reasoning_effort`
    terminal_bench/run_sweep.py  harbor Terminus-2: `reasoning_effort` for OpenAI and
                                 Anthropic; Together through `extra_body`

Together-hosted models receive the effort through `extra_body` because LiteLLM's
`together_ai` provider does not list `reasoning_effort` as a supported parameter, and with
`drop_params: true` it would be silently discarded.

Two run-directory naming conventions are used:

    name_slug()     "claude-sonnet-5"            SWE-Bench Pro, AIME, GPQA
    litellm_slug()  "anthropic_claude-sonnet-5"  Terminal-Bench
"""
from __future__ import annotations

from dataclasses import dataclass

# Effort levels each provider family accepts.
#   'none' disables reasoning. For Anthropic this requires sending
#   thinking={'type': 'disabled'} explicitly; each harness does so.
#   OpenAI GPT-5.x rejects 'minimal', so it is not listed.
EFFORTS: dict[str, list[str]] = {
    "openai": ["none", "low", "medium", "high", "xhigh"],
    "anthropic": ["none", "low", "medium", "high", "xhigh", "max"],
    # Permissive superset for Together-hosted models; models narrow it via `efforts`.
    "together": ["none", "low", "medium", "high", "max"],
}

# Increasing reasoning spend; for sorting effort directories and rows.
EFFORT_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}

FAMILY_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                  "together": "TOGETHER_API_KEY"}


@dataclass(frozen=True)
class ModelSpec:
    slug: str          # short name (what you pass to --model)
    family: str        # "openai" | "anthropic" | "together" — decides the effort knob
    litellm_id: str    # provider-prefixed id for litellm / harbor `-m`
    key_env: str       # environment variable holding the API key
    registered: bool = True   # False = inferred, not in MODELS (no price table entry either)
    efforts: tuple[str, ...] | None = None
    """Effort levels this model distinguishes, when narrower than its family's."""


def _spec(slug: str, family: str, litellm_id: str | None = None,
          efforts: tuple[str, ...] | None = None) -> ModelSpec:
    return ModelSpec(slug=slug, family=family,
                     litellm_id=litellm_id or f"{family}/{slug}",
                     key_env=FAMILY_KEY_ENV[family], efforts=efforts)


MODELS: dict[str, ModelSpec] = {
    m.slug: m for m in [
        _spec("gpt-5.6-sol", "openai"),
        _spec("gpt-5.6-terra", "openai"),
        _spec("gpt-5.6-luna", "openai"),
        _spec("claude-sonnet-5", "anthropic"),
        # GLM-5.2's chat template maps `high` to "Reasoning Effort: High" and every other
        # value to "Max", so `low` and `medium` would be aliases of `max`.
        _spec("glm-5.2", "together", "together_ai/zai-org/GLM-5.2",
              efforts=("none", "high", "max")),
        _spec("inkling", "together", "together_ai/thinkingmachines/Inkling"),
        # Kimi K3 documents low | high | max and always thinks; there is no `none`.
        _spec("kimi-k3", "together", "together_ai/moonshotai/Kimi-K3",
              efforts=("low", "high", "max")),
        # DeepSeek V4 Flash documents low | high | max; `none` disables thinking.
        _spec("deepseek-v4-flash", "together", "together_ai/deepseek-ai/DeepSeek-V4-Flash-0731",
              efforts=("none", "low", "high", "max")),
    ]
}


# Agent wall-clock headroom per model on SWE-Bench Pro, as a multiple of the default
# (applied to --wall-time and the per-call timeout). Recorded in each run's manifest.
WALL_CLOCK_SCALE: dict[str, float] = {"deepseek-v4-flash": 3.0}


def wall_clock_scale(model: str) -> float:
    """How much longer than the default this model's agent may run (1.0 = unchanged)."""
    return WALL_CLOCK_SCALE.get(name_slug(model), 1.0)


def together_effort_body(model: str, effort: str) -> dict:
    """The `extra_body` that sets reasoning effort for a Together-hosted model.

    For `none`, both `reasoning_effort=none` and `enable_thinking=false` are sent: different
    models honour different switches, and sending both disables thinking on all of them.
    """
    body: dict = {"reasoning_effort": effort}
    if effort == "none":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


# Models that reach the shell through mini-swe-agent's native bash tool instead of the
# text-based scaffold (one ```bash block per reply). These models answer in their native
# tool-call syntax, which the text scaffold rejects as a format error.
REQUIRES_TOOLCALL: frozenset[str] = frozenset({"inkling", "deepseek-v4-flash", "kimi-k3"})


def requires_toolcall(model: str) -> bool:
    """True when this model must reach the shell through mini's native bash tool."""
    return name_slug(model) in REQUIRES_TOOLCALL


def infer_family(model: str) -> str:
    """Family of an unregistered model name, from the name itself."""
    m = model.lower()
    # Together first: it serves other vendors' models.
    if m.startswith("together_ai/"):
        return "together"
    if "claude" in m or m.startswith("anthropic/"):
        return "anthropic"
    if "gpt" in m or m.startswith("openai/"):
        return "openai"
    raise SystemExit(f"cannot tell which provider {model!r} belongs to — add it to "
                     "common/models.py MODELS")


def resolve(model: str) -> ModelSpec:
    """Look up a model by slug or by provider-prefixed litellm id.

    Unregistered names are accepted (family inferred, `registered=False`); they have no
    price table entry.
    """
    if model in MODELS:
        return MODELS[model]
    for spec in MODELS.values():
        if spec.litellm_id == model:
            return spec
    family = infer_family(model)
    return ModelSpec(slug=model.split("/", 1)[-1], family=family,
                     litellm_id=model if "/" in model else f"{family}/{model}",
                     key_env=FAMILY_KEY_ENV[family], registered=False)


def valid_efforts(model: str) -> list[str]:
    """Effort levels this model distinguishes — its own list if it has one, else its family's."""
    spec = resolve(model)
    return list(spec.efforts) if spec.efforts else EFFORTS[spec.family]


def check_efforts(model: str, efforts: list[str]) -> None:
    """Exit with a clear message if any effort level is invalid for this model."""
    allowed = valid_efforts(model)
    bad = [e for e in efforts if e not in allowed]
    if bad:
        raise SystemExit(f"effort(s) {bad} invalid for {model}; valid: {allowed}")


def is_anthropic(model: str) -> bool:
    return resolve(model).family == "anthropic"


def is_together(model: str) -> bool:
    return resolve(model).family == "together"


def api_id(model: str) -> str:
    """The model string to hand the API client.

    OpenAI and Anthropic ids are callable bare; Together ids need the provider prefix, or
    LiteLLM cannot infer the provider.
    """
    spec = resolve(model)
    return spec.litellm_id if spec.family == "together" else model


def name_slug(model: str) -> str:
    """Filesystem-safe short name (SWE-Bench Pro, AIME and GPQA run directories).

    A registered model collapses to its slug, so `glm-5.2` and `together_ai/zai-org/GLM-5.2`
    write to one directory.
    """
    for spec in MODELS.values():
        if model in (spec.slug, spec.litellm_id):
            return spec.slug
    return model.replace("/", "_")


def litellm_slug(model: str) -> str:
    """Filesystem-safe provider-prefixed id (Terminal-Bench run directories)."""
    return resolve(model).litellm_id.replace("/", "_")
