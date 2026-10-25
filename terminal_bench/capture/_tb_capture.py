"""LiteLLM success callback that records the full usage of every call harbor makes.

Terminus-2 keeps only prompt/completion/cached tokens and cost per trial; it drops
`completion_tokens_details.reasoning_tokens` and the cache-creation tokens. This callback reads
LiteLLM's response objects directly and appends one JSON record per successful call to the
file named by TB_LITELLM_CAPTURE (`<run dir>/litellm_calls.jsonl`). Registered by
sitecustomize.py; any failure is recorded rather than raised.
"""
from __future__ import annotations

import json
import os
import threading

import litellm
from litellm.integrations.custom_logger import CustomLogger

_PATH = os.environ.get("TB_LITELLM_CAPTURE")
_LOCK = threading.Lock()


def _reasoning_content(response_obj) -> str | None:
    try:
        choices = getattr(response_obj, "choices", None)
        if choices:
            msg = getattr(choices[0], "message", None)
            # Providers differ in the field name: Together answers on `reasoning`, others on
            # `reasoning_content`.
            for field in ("reasoning_content", "reasoning"):
                rc = getattr(msg, field, None)
                if isinstance(rc, str) and rc:
                    return rc
    except Exception:
        pass
    return None


def _extract(kwargs, response_obj, start_time, end_time) -> dict:
    usage = getattr(response_obj, "usage", None)
    usage_d = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None
    hidden = getattr(response_obj, "_hidden_params", {}) or {}
    cost = hidden.get("response_cost")
    if cost is None:
        cost = kwargs.get("response_cost")
    rc = _reasoning_content(response_obj)
    return {
        "response_id": getattr(response_obj, "id", None),
        "model": getattr(response_obj, "model", None) or kwargs.get("model"),
        "call_id": kwargs.get("litellm_call_id"),
        "response_cost": cost,                 # LiteLLM's per-call cost (what harbor sums)
        "usage": usage_d,                      # full breakdown incl. reasoning and cache creation
        "reasoning_content": rc,               # reasoning text when the provider returns one
        "reasoning_content_len": len(rc) if rc else 0,
        # The reasoning settings as sent, so each call's effort can be checked against the
        # request. The messages are not recorded; they are in the trial trajectory.
        "sent_extra_body": (kwargs.get("optional_params") or {}).get("extra_body"),
        "sent_reasoning_effort": (kwargs.get("optional_params") or {}).get("reasoning_effort"),
        "start": str(start_time), "end": str(end_time),
    }


def _write(kwargs, response_obj, start_time, end_time) -> None:
    if not _PATH:
        return
    try:
        rec = _extract(kwargs, response_obj, start_time, end_time)
    except Exception as e:
        rec = {"capture_error": repr(e)}
    line = json.dumps(rec, default=str)
    with _LOCK:
        with open(_PATH, "a") as f:
            f.write(line + "\n")


class TBCapture(CustomLogger):
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        _write(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        _write(kwargs, response_obj, start_time, end_time)


def register() -> None:
    """Add the callback to LiteLLM once."""
    if not any(isinstance(cb, TBCapture) for cb in litellm.callbacks):
        litellm.callbacks.append(TBCapture())
