"""Auto-loaded at interpreter startup when this directory is on PYTHONPATH.

When TB_LITELLM_CAPTURE is set (terminal_bench/run_sweep.py sets it), registers the LiteLLM
per-call capture (_tb_capture.py) and the Anthropic thinking-token fix
(_litellm_anthropic_fix.py) inside harbor's interpreter. Failures are reported and never stop
a harbor run.
"""
import os
import sys

if os.environ.get("TB_LITELLM_CAPTURE"):
    try:
        import _tb_capture

        _tb_capture.register()
    except Exception as e:  # pragma: no cover
        print(f"[tb-capture] disabled (registration failed): {e!r}", file=sys.stderr)

    try:
        import _litellm_anthropic_fix

        if not _litellm_anthropic_fix.register():
            print("[tb-capture] anthropic thinking-token fix not applied", file=sys.stderr)
    except Exception as e:  # pragma: no cover
        print(f"[tb-capture] anthropic thinking-token fix failed: {e!r}", file=sys.stderr)
