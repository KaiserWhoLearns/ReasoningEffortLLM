"""Restore Anthropic's metered thinking-token count in LiteLLM's usage object.

Anthropic returns reasoning spend in `usage.output_tokens_details.thinking_tokens`. LiteLLM's
`AnthropicConfig.calculate_usage` ignores it and estimates `reasoning_tokens` from the visible
reasoning text, which is 0 when thinking is not displayed. harbor calls LiteLLM internally, so
this patches `calculate_usage` in harbor's interpreter (via sitecustomize.py) to overwrite
the estimate with the native count. Anthropic only; idempotent; any failure leaves LiteLLM's
behavior unchanged.
"""
from __future__ import annotations


def register() -> bool:
    """Patch AnthropicConfig.calculate_usage; False if LiteLLM's Anthropic config is missing."""
    try:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    except Exception:
        return False
    if getattr(AnthropicConfig, "_tb_thinking_fix", False):
        return True

    _orig = AnthropicConfig.calculate_usage

    def calculate_usage(self, usage_object, reasoning_content, completion_response=None, speed=None):
        usage = _orig(
            self, usage_object, reasoning_content,
            completion_response=completion_response, speed=speed,
        )
        try:
            otd = (usage_object or {}).get("output_tokens_details") or {}
            native = otd.get("thinking_tokens")
            if native is not None:
                native = int(native)
                ctd = getattr(usage, "completion_tokens_details", None)
                if ctd is None:
                    from litellm.types.utils import CompletionTokensDetailsWrapper
                    ctd = CompletionTokensDetailsWrapper()
                    usage.completion_tokens_details = ctd
                ctd.reasoning_tokens = native
                try:  # keep the text/reasoning split consistent
                    ctd.text_tokens = max(int(usage.completion_tokens) - native, 0)
                except Exception:
                    pass
                # Keep the native field too (Usage allows extra fields).
                setattr(usage, "output_tokens_details", {"thinking_tokens": native})
        except Exception:
            pass
        return usage

    calculate_usage.__wrapped__ = _orig
    AnthropicConfig.calculate_usage = calculate_usage
    AnthropicConfig._tb_thinking_fix = True
    return True
