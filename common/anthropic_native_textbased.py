"""mini-swe-agent text-based model that calls the Anthropic Messages API directly (Claude only).

LiteLLM does not pass through Anthropic's metered `usage.output_tokens_details.thinking_tokens`;
it estimates `reasoning_tokens` from the visible reasoning text, which is 0 when thinking is
not displayed. This class replaces only `LitellmTextbasedModel._query()` with a native
Anthropic call, so the real thinking-token count is recorded while the rest of the scaffold
(agent loop, prompts, `_prepare_messages_for_api`, action parsing, `litellm.completion_cost`,
retries) is inherited unchanged. `_query` receives the prepared OpenAI-shaped messages, and

  1. translates them into a native Anthropic request (system blocks, cache_control, replayed
     thinking blocks), passing `thinking` and `output_config` through from model_kwargs;
  2. calls `anthropic.Anthropic().messages.create(...)` without streaming (the streamed final
     message lacks `output_tokens_details`), with an explicit timeout so the SDK accepts a
     large `max_tokens`;
  3. wraps the response in a `litellm.ModelResponse` whose usage carries
     `output_tokens_details.thinking_tokens` and the cache fields in LiteLLM's convention, so
     the cost computation is unchanged.

`max_tokens` is required by the Anthropic API; set it in the config's model_kwargs (default
32000). swebench_pro/run_sweep.py selects this class for Anthropic models via
`model_class: common.anthropic_native_textbased.AnthropicNativeTextbasedModel`; it needs the
`anthropic` package and ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

import anthropic
import litellm
from litellm.types.utils import (
    Choices,
    CompletionTokensDetailsWrapper,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)

from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

_DEFAULT_MAX_TOKENS = 32000
# Seconds; LiteLLM's default. An explicit timeout lets non-streaming create() accept large
# max_tokens.
_REQUEST_TIMEOUT = 600

# Key under Message.provider_specific_fields holding the raw ordered Anthropic content
# blocks, so a replayed assistant turn is byte-identical (see _to_anthropic).
_RAW_CONTENT_KEY = "anthropic_raw_content"

# Anthropic stop_reason -> OpenAI finish_reason (what _parse_actions reads).
_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "stop",
}


class AnthropicNativeTextbasedModel(LitellmTextbasedModel):
    """Text-based mini model that calls the Anthropic SDK directly to keep the real
    thinking-token count. Claude models only."""

    # Anthropic's terminal errors abort the retry loop, as LiteLLM's do in the parent class.
    abort_exceptions = LitellmTextbasedModel.abort_exceptions + [
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.BadRequestError,
    ]

    _client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    # --- request translation: prepared OpenAI messages -> native Anthropic request ----------
    def _to_anthropic(self, messages: list[dict], kwargs: dict) -> dict:
        merged = {**self.config.model_kwargs, **kwargs}
        system_blocks: list[dict] = []
        conv: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                system_blocks.extend(_content_to_blocks(content))
                continue
            # Replay a prior assistant turn from its raw Anthropic content when available.
            # Anthropic rejects a replayed turn whose thinking blocks differ from the originals,
            # so they must go back byte-identical and in their original positions, including
            # redacted_thinking blocks; rebuilding from `thinking_blocks` would not preserve that.
            psf = msg.get("provider_specific_fields")
            raw = psf.get(_RAW_CONTENT_KEY) if isinstance(psf, dict) else None
            if role == "assistant" and raw:
                # Anthropic emits empty text blocks but rejects them on input, so drop them;
                # only thinking/redacted_thinking blocks are signature-protected.
                kept = [
                    b for b in copy.deepcopy(raw)
                    if b.get("type") != "text" or (b.get("text") or "").strip()
                ]
                if kept:
                    conv.append({"role": "assistant", "content": kept})
                    continue

            blocks: list[dict] = []
            # Turns without raw content: thinking blocks first (Anthropic ordering, as LiteLLM).
            for tb in msg.get("thinking_blocks") or []:
                if isinstance(tb, dict) and tb.get("type") in ("thinking", "redacted_thinking"):
                    blocks.append(tb)
            blocks.extend(
                b for b in _content_to_blocks(content)
                if b.get("type") not in ("thinking", "redacted_thinking")
            )
            conv.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})

        req: dict[str, Any] = {
            "model": self.config.model_name.split("/", 1)[-1],  # strip any "anthropic/" prefix
            "max_tokens": int(merged.get("max_tokens", _DEFAULT_MAX_TOKENS)),
            "messages": conv,
        }
        if system_blocks:
            req["system"] = system_blocks
        # Reasoning settings pass straight through (set by swebench_pro/run_sweep.py).
        if "thinking" in merged:
            req["thinking"] = merged["thinking"]
        if "output_config" in merged:
            req["output_config"] = merged["output_config"]
        return req

    # --- the only override -------------------------------------------------------------------
    def _query(self, messages: list[dict[str, str]], **kwargs):
        # Non-streaming: only the non-streamed response carries output_tokens_details.
        resp = self.client.messages.create(
            **self._to_anthropic(messages, kwargs), timeout=_REQUEST_TIMEOUT
        )
        return _to_model_response(resp, self.config.model_name)


def _content_to_blocks(content) -> list[dict]:
    """Normalize an OpenAI-shaped message `content` (str | list) to Anthropic content blocks,
    preserving cache_control (same {"type": "ephemeral"} shape mini's set_cache_control emits)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("thinking", "redacted_thinking"):
            blocks.append(b)
            continue
        block: dict[str, Any] = {"type": "text", "text": b.get("text", "")}
        if b.get("cache_control"):
            block["cache_control"] = b["cache_control"]
        blocks.append(block)
    return blocks


def _to_model_response(resp, model_name: str) -> ModelResponse:
    """Wrap a native Anthropic Message as a litellm ModelResponse (with the real thinking-token
    count) so the inherited mini code runs unchanged."""
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    thinking_blocks = [
        {"type": "thinking", "thinking": b.thinking, "signature": getattr(b, "signature", "")}
        for b in resp.content
        if getattr(b, "type", None) == "thinking"
    ]
    reasoning = "".join(tb["thinking"] for tb in thinking_blocks) or None
    # The full ordered content, kept so _to_anthropic() can replay this turn verbatim.
    raw_content = [
        b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b)
        for b in resp.content
    ]

    u = resp.usage
    input_tok = int(getattr(u, "input_tokens", 0) or 0)          # uncached input
    output_tok = int(getattr(u, "output_tokens", 0) or 0)        # inclusive of thinking
    cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
    cache_creation = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    # LiteLLM convention: prompt_tokens is INCLUSIVE of cache tokens (so completion_cost matches).
    prompt_tok = input_tok + cache_read + cache_creation

    otd = getattr(u, "output_tokens_details", None)
    thinking_tok = getattr(otd, "thinking_tokens", None) if otd is not None else None

    usage_kwargs: dict[str, Any] = dict(
        prompt_tokens=prompt_tok,
        completion_tokens=output_tok,
        total_tokens=prompt_tok + output_tok,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cache_read),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=int(thinking_tok) if thinking_tok is not None else 0
        ),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )
    if thinking_tok is not None:
        # Read first by common/usage.py; Usage has extra='allow', so it survives model_dump().
        usage_kwargs["output_tokens_details"] = {"thinking_tokens": int(thinking_tok)}
    usage = Usage(**usage_kwargs)

    message = Message(
        role="assistant",
        content=text or None,
        thinking_blocks=thinking_blocks or None,
        reasoning_content=reasoning,
        provider_specific_fields={_RAW_CONTENT_KEY: raw_content},
    )
    return ModelResponse(
        id=resp.id,
        created=int(time.time()),
        model=model_name,
        choices=[Choices(index=0, message=message,
                         finish_reason=_FINISH.get(resp.stop_reason, "stop"))],
        usage=usage,
    )
