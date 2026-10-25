"""Waiting time as a second cost axis: decode time priced at an hourly wage.

The full cost of a run is

    full cost = API cost + decode seconds / 3600 x hourly wage
    decode seconds = output tokens / output tokens-per-second

Output tokens are the provider's inclusive generated count (reasoning tokens are a subset of
them), so a level that reasons longer waits longer. Prefill, queueing and tool/environment
execution time are not included, so the waiting time is a lower bound.

Throughput source
    ArtificialAnalysis (artificialanalysis.ai) "output speed": median output tokens/s, for
    the reasoning variant at its top effort level, one figure per model. For GLM-5.2,
    Inkling and Kimi K3 the figure is Together's row on the providers page, since Together
    served these runs. ArtificialAnalysis does not track Together for DeepSeek V4 Flash, so
    its figure is DeepSeek's first-party API.

Wage source
    BLS Occupational Employment and Wage Statistics, May 2025 national estimates, Table 1
    (mean hourly wages). The default is the all-occupations mean, $33.54; the software
    developer mean, $71.20, is available through `set_wage("developer")`.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import models


@dataclass(frozen=True)
class Throughput:
    tokens_per_s: float
    variant: str     # the ArtificialAnalysis model page the figure is read from
    provider: str    # whose API the figure was measured against


# Output tokens per second, keyed by model slug.
OUTPUT_TPS: dict[str, Throughput] = {
    "gpt-5.6-sol":     Throughput(59.0,  "GPT-5.6 Sol (high)",   "OpenAI"),
    "gpt-5.6-terra":   Throughput(92.9,  "GPT-5.6 Terra (max)",  "OpenAI"),
    "gpt-5.6-luna":    Throughput(114.8, "GPT-5.6 Luna (max)",   "OpenAI"),
    "claude-sonnet-5": Throughput(71.0,  "Claude Sonnet 5 (adaptive reasoning, max effort)",
                                  "Anthropic"),
    "glm-5.2":         Throughput(211.0, "GLM-5.2 (max)", "Together AI"),
    "inkling":         Throughput(108.8, "Inkling (xhigh)", "Together AI"),
    "kimi-k3":         Throughput(60.0,  "Kimi K3 (max)",   "Together AI"),
    "deepseek-v4-flash": Throughput(218.0, "DeepSeek V4 Flash 0731 (max)", "DeepSeek"),
}


@dataclass(frozen=True)
class Wage:
    key: str
    usd_per_hour: float
    who: str         # whose hour it is, for labels
    source: str


WAGES: dict[str, Wage] = {
    "all": Wage("all", 33.54, "an average worker",
                "BLS OEWS May 2025, all occupations (SOC 00-0000), national mean hourly wage"),
    "developer": Wage("developer", 71.20, "a software developer",
                      "BLS OEWS May 2025, Software Developers (SOC 15-1252), national mean "
                      "hourly wage"),
}
DEFAULT_WAGE = "all"

# The wage every derived number uses in this process; change it with `set_wage`.
WAGE: Wage = WAGES[DEFAULT_WAGE]


def set_wage(key: str) -> Wage:
    """Select the wage the wait is priced at, for this process. Returns it."""
    global WAGE
    if key not in WAGES:
        raise SystemExit(f"unknown wage {key!r}; one of {sorted(WAGES)}")
    WAGE = WAGES[key]
    return WAGE


def throughput(model: str) -> Throughput | None:
    """The throughput entry for a model slug or litellm id, or None if it has none."""
    if not model:
        return None
    if model in OUTPUT_TPS:
        return OUTPUT_TPS[model]
    try:
        return OUTPUT_TPS.get(models.resolve(model).slug)
    except SystemExit:
        return None


def decode_seconds(model: str, output_tokens: float | None) -> float | None:
    """Seconds spent waiting for `output_tokens` at the model's table throughput."""
    t = throughput(model)
    if t is None or output_tokens is None:
        return None
    return output_tokens / t.tokens_per_s


def wait_cost_usd(seconds: float | None, wage: float | None = None) -> float | None:
    """What `seconds` of waiting costs at `wage` dollars per hour (default: the active WAGE)."""
    if seconds is None:
        return None
    return seconds / 3600.0 * (WAGE.usd_per_hour if wage is None else wage)


def full_cost_usd(api_cost: float | None, seconds: float | None,
                  wage: float | None = None) -> float | None:
    """API cost plus the wage cost of waiting for it."""
    w = wait_cost_usd(seconds, wage)
    if api_cost is None or w is None:
        return None
    return api_cost + w


def fmt_seconds(s: float) -> str:
    """'42s' / '3.5 min' / '1.20 h'."""
    if s < 60:
        return f"{s:.0f}s" if s >= 10 else f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"
