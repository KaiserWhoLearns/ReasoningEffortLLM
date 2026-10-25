#!/usr/bin/env python3
"""Terminal-Bench 2.1, stage 1 of 2: run Terminus-2 at each reasoning-effort level.

For each (model, effort) this runs the official harbor CLI with the benchmark's reference
agent, one trial per task:

    harbor run -d terminal-bench/terminal-bench-2-1 -a terminus-2 -m <litellm id>
               --ak <effort kwarg> --ak model_info=<rates> -k 1 -n <workers>
               --jobs-dir <run dir> --job-name tb2.1_<effort>
               [--agent-timeout-multiplier <x>] --extra-docker-compose configs/runtime-runc.yaml

Terminus-2 keeps its defaults (temperature 0.7, no turn limit); each trial is bounded only
by the task's agent wall-clock timeout, scaled per model by AGENT_TIMEOUT_MULTIPLIER.

Results go to `paths.run_dir("terminal-bench-2-1", models.litellm_slug(model), effort)`,
e.g. data/terminal_bench/openai_gpt-5.6-sol/high/. Each run dir also gets a manifest.json
and litellm_calls.jsonl, a per-call usage record written by capture/ inside harbor's
interpreter (reasoning and cache-creation tokens, which harbor's trial metrics drop). Turn
the trials into a cost_summary.json with collect_cost.py.

Requirements
    harbor, installed as a uv tool so that `harbor` is on PATH (the paper's runs used harbor
    0.20 and 0.22):    uv tool install harbor==0.20.0
    a running Docker daemon (harbor builds and runs each task's containers locally), and
    OPENAI_API_KEY / ANTHROPIC_API_KEY / TOGETHER_API_KEY in the environment or the repo .env.

Examples
    python terminal_bench/run_sweep.py --model claude-sonnet-5 --efforts low --limit 1
    python terminal_bench/run_sweep.py --model gpt-5.6-terra --full --workers 6
    python terminal_bench/run_sweep.py --dry-run --full          # print every command
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import models, paths, pricing

BENCHMARK = "terminal-bench-2-1"
DATASET = "terminal-bench/terminal-bench-2-1"
AGENT = "terminus-2"
N_ATTEMPTS = 1                # one trial per task

HERE = Path(__file__).resolve().parent
RUNC_OVERLAY = HERE / "configs" / "runtime-runc.yaml"
CAPTURE_DIR = HERE / "capture"   # sitecustomize.py + LiteLLM per-call usage capture

# harbor bounds each trial by the task's agent wall-clock timeout, not by a turn count. The
# Together-hosted models get a larger budget because provider latency, not agent progress,
# limits them; the multiplier is fixed across a model's efforts and recorded in manifest.json.
# Every other model keeps harbor's default.
AGENT_TIMEOUT_MULTIPLIER: dict[str, float] = {
    "glm-5.2": 3.0,
    "inkling": 3.0,
    "kimi-k3": 5.0,
    "deepseek-v4-flash": 9.0,
}


def default_efforts(model: str) -> list[str]:
    """The effort levels the paper evaluates: all of the model's levels, except `max` for the
    proprietary models."""
    efforts = models.valid_efforts(model)
    if not models.is_together(model):
        efforts = [e for e in efforts if e != "max"]
    return efforts


def harbor_bin(required: bool = True) -> str:
    exe = shutil.which("harbor") or str(Path.home() / ".local" / "bin" / "harbor")
    if not Path(exe).exists():
        if not required:
            return "harbor"
        sys.exit("harbor not found; install it with:  uv tool install harbor==0.20.0")
    return exe


def effort_kwargs(model: str, effort: str) -> list[str]:
    """The harbor `--ak` pair that sets reasoning effort for this model.

    Terminus-2 forwards `reasoning_effort` to LiteLLM, which is correct for OpenAI at every
    level and for Anthropic at every level except `none`:

    * Anthropic `none`: LiteLLM maps `reasoning_effort=none` to omitting `thinking`, and an
      adaptive-thinking Claude then thinks at its default effort. Thinking must be disabled
      explicitly with `thinking={'type': 'disabled'}`, sent alone: alongside
      `reasoning_effort`, LiteLLM would drop it.
    * Together: LiteLLM's together_ai provider does not support `reasoning_effort` and harbor
      sets `drop_params: True`, so the parameter would be silently discarded. Terminus-2's
      `llm_call_kwargs` reach `litellm.completion` unfiltered, so the effort goes in
      `extra_body` (see `models.together_effort_body`).
    """
    if models.is_together(model):
        return ["--ak", "llm_call_kwargs=" + json.dumps(
            {"extra_body": models.together_effort_body(model, effort)},
            separators=(",", ":"))]
    if models.is_anthropic(model) and effort == "none":
        return ["--ak", "llm_call_kwargs=" + json.dumps(
            {"thinking": {"type": "disabled"}}, separators=(",", ":"))]
    return ["--ak", f"reasoning_effort={effort}"]


def _wire_params(model: str, effort: str) -> dict:
    """The Anthropic request params LiteLLM derives from effort_kwargs(), with drop_params."""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    key, val = effort_kwargs(model, effort)[1].split("=", 1)
    non_default = json.loads(val) if key == "llm_call_kwargs" else {key: val}
    bare = models.resolve(model).litellm_id.split("/", 1)[-1]   # litellm strips the provider
    return AnthropicConfig().map_openai_params(non_default, {}, bare, drop_params=True)


def preflight_anthropic(selected: list[str], efforts: dict[str, list[str]]) -> None:
    """Exit if LiteLLM would not send a Claude run's effort as intended.

    `none` must carry thinking={'type': 'disabled'}; every other level adaptive thinking with
    output_config.effort equal to the level. No API call is made.
    """
    for model in (m for m in selected if models.is_anthropic(m)):
        for effort in efforts[model]:
            p = _wire_params(model, effort)
            thinking, oc = p.get("thinking"), p.get("output_config") or {}
            if effort == "none":
                ok = thinking == {"type": "disabled"}
            else:
                ok = (thinking or {}).get("type") == "adaptive" and oc.get("effort") == effort
            if not ok:
                sys.exit(f"preflight failed [{model} {effort}]: LiteLLM would send "
                         f"thinking={thinking!r} output_config={oc!r}; see effort_kwargs()")
        print(f"  preflight OK: {model} {efforts[model]} carry their effort")


def agent_timeout_multiplier(model: str, args) -> float | None:
    """Agent wall-clock multiplier for this model, or None for harbor's default."""
    if args.agent_timeout_multiplier is not None:
        return args.agent_timeout_multiplier
    return AGENT_TIMEOUT_MULTIPLIER.get(models.name_slug(model))


def build_cmd(model: str, effort: str, args) -> tuple[list[str], Path]:
    spec = models.resolve(model)
    run_dir = paths.run_dir(BENCHMARK, models.litellm_slug(model), effort)
    cmd = [
        harbor_bin(required=not args.dry_run), "run",
        "-d", DATASET,
        "-a", AGENT,
        "-m", spec.litellm_id,
        *effort_kwargs(model, effort),
        "-k", str(N_ATTEMPTS),
        "-n", str(args.workers),
        "--jobs-dir", str(run_dir),
        "--job-name", f"tb2.1_{effort}",
    ]
    if args.register_pricing and pricing.rates(spec.litellm_id):
        # One-line JSON; harbor parses it into Terminus-2's `model_info` kwarg.
        cmd += ["--ak", "model_info=" + json.dumps(pricing.rates(spec.litellm_id),
                                                   separators=(",", ":"))]
    timeout_multiplier = agent_timeout_multiplier(model, args)
    if timeout_multiplier is not None:
        cmd += ["--agent-timeout-multiplier", str(timeout_multiplier)]
    if args.env:
        cmd += ["-e", args.env]
    if args.docker_runtime == "runc":
        cmd += ["--extra-docker-compose", str(RUNC_OVERLAY)]
    if args.task:
        cmd += ["--task", args.task]
    elif args.limit:
        cmd += ["-l", str(args.limit)]
    cmd += args.harbor_arg
    return cmd, run_dir


def run_one(model: str, effort: str, args) -> None:
    cmd, run_dir = build_cmd(model, effort, args)
    print(f"\n=== {model}  effort={effort}  -> {run_dir} ===")
    print("  " + " ".join(cmd))
    if args.dry_run:
        return

    if args.fresh and run_dir.exists():
        # harbor refuses to reuse a job dir written with a different configuration.
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "stage": "generation", "benchmark": BENCHMARK, "agent": AGENT,
        "model": model, "litellm_model": models.resolve(model).litellm_id,
        "reasoning_effort": effort, "n_attempts": N_ATTEMPTS,
        "task": args.task, "limit": args.limit, "env": args.env or "docker (local)",
        "agent_timeout_multiplier": agent_timeout_multiplier(model, args),
        "priced_from_overrides": bool(args.register_pricing
                                      and pricing.rates(model) is not None),
        "cmd": cmd, "created_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    env = os.environ.copy()
    if args.capture_litellm:
        # capture/ on PYTHONPATH: its sitecustomize.py registers the LiteLLM callback in
        # harbor's interpreter, which appends to TB_LITELLM_CAPTURE.
        env["PYTHONPATH"] = os.pathsep.join([str(CAPTURE_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env["TB_LITELLM_CAPTURE"] = str(run_dir / "litellm_calls.jsonl")

    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", dest="models",
                    help="repeatable; slug or litellm id (default: every model in common/models.py)")
    ap.add_argument("--efforts", default=None,
                    help="comma list of effort levels for every selected model (default: the "
                         "paper's levels per model: all valid levels, without `max` for the "
                         "OpenAI and Anthropic models)")
    ap.add_argument("--full", action="store_true", help="run all 89 tasks")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N tasks (-l)")
    ap.add_argument("--task", default=None, help="run exactly one task by name")
    ap.add_argument("--workers", type=int, default=4, help="concurrent trials (-n)")
    ap.add_argument("--env", default=None,
                    help="harbor sandbox environment (-e), e.g. daytona; default: local Docker")
    ap.add_argument("--docker-runtime", choices=["runc", "default"], default="runc",
                    help="'runc' passes configs/runtime-runc.yaml (needed when Docker's default "
                         "runtime is nvidia); 'default' uses the daemon default")
    ap.add_argument("--agent-timeout-multiplier", type=float, default=None,
                    help="agent wall-clock multiplier for every model (default: "
                         f"{AGENT_TIMEOUT_MULTIPLIER}, harbor's default for the others)")
    ap.add_argument("--no-register-pricing", dest="register_pricing", action="store_false",
                    help="do not pass common/pricing.py rates to Terminus-2 as model_info "
                         "(harbor's reported cost then uses LiteLLM's registry)")
    ap.add_argument("--no-capture-litellm", dest="capture_litellm", action="store_false",
                    help="do not record per-call LiteLLM usage to <run dir>/litellm_calls.jsonl")
    ap.add_argument("--fresh", action="store_true",
                    help="delete each run dir before launching; omit to let harbor resume an "
                         "existing job with the same configuration")
    ap.add_argument("--dry-run", action="store_true", help="print the harbor commands without running them")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the check that each Claude effort reaches the API as intended")
    ap.add_argument("--harbor-arg", action="append", default=[],
                    help="repeatable flag appended verbatim to `harbor run` (e.g. --harbor-arg=--force-build)")
    args = ap.parse_args()

    selected = args.models or list(models.MODELS)
    if not (args.full or args.limit or args.task):
        sys.exit("refusing to run the whole task set implicitly: pass --limit, --task, or --full")

    efforts: dict[str, list[str]] = {}
    for m in selected:
        if not models.resolve(m).registered:
            sys.exit(f"unknown model {m!r}; known: {', '.join(models.MODELS)}")
        efforts[m] = ([e.strip() for e in args.efforts.split(",") if e.strip()]
                      if args.efforts else default_efforts(m))
        models.check_efforts(m, efforts[m])

    if not args.dry_run:
        paths.load_env(sorted({models.resolve(m).key_env for m in selected}))
    print(f"dataset={DATASET}  agent={AGENT}")
    if not args.dry_run and not args.skip_preflight:
        preflight_anthropic(selected, efforts)
    for model in selected:
        for effort in efforts[model]:
            run_one(model, effort, args)
    print(f"\ndone -> {paths.data_root(BENCHMARK)}")
    print("next: collect_cost.py to turn the trials into cost_summary.json")


if __name__ == "__main__":
    main()
