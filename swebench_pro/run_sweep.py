#!/usr/bin/env python3
"""SWE-Bench Pro, stage 1 of 3: run mini-swe-agent across reasoning-effort levels.

For each effort level, merges the dataset's config stack (see variants.py), the model's scaffold
(configs/toolcall.yaml for models in common.models.REQUIRES_TOOLCALL) and the effort setting, then
runs mini over the selected instances. The effort is set per provider family:

    OpenAI     model_kwargs.reasoning_effort
    Anthropic  output_config.effort + adaptive thinking; `none` = thinking disabled
               (LiteLLM's reasoning_effort only switches Claude's thinking on or off)
    Together   extra_body from common.models.together_effort_body (LiteLLM drops a plain
               reasoning_effort for together_ai)

Results land in `paths.run_dir(<benchmark>, <model>, <effort>)`:

    <instance_id>/<instance_id>.traj.json   full trajectory
    preds.json                               {instance_id: {model_name_or_path, instance_id, model_patch}}
    config.merged.yaml                       the exact config this effort ran with
    manifest.json                            dataset, scaffold, model, effort, limits, rates, instances
    cost_summary.json                        per-instance tokens, cost, api_calls, exit status

Instances already in preds.json are skipped, so re-running the same command resumes. Score the
patches with evaluate.py, then summarize with report.py.

Usage (needs Docker and the provider's API key in .env):
    python prepare_dataset.py                                   # once
    python run_sweep.py --model gpt-5.6-sol --efforts low --limit 1 --dry-run
    python run_sweep.py --model gpt-5.6-sol --workers 8         # 44-instance pilot, every effort
    python verify_sanitization.py && python verify_egress.py    # before the corrected runs
    python run_sweep.py --model gpt-5.6-sol --dataset pro-corrected --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name, process_instance
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import recursive_merge
from rich.live import Live

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for `common`
from common import models, paths, pricing
# The config names these classes by dotted path; importing them here fails at startup rather than
# once per instance inside a worker thread.
from common import anthropic_native_textbased, sanitized_docker, sealed_docker  # noqa: F401

import report
import variants

HERE = Path(__file__).resolve().parent
INSTANCES = paths.DATASETS_ROOT / "swebench_pro" / "mini_instances.jsonl"
PILOT = HERE / "subsets" / "pilot.txt"
NATIVE_ANTHROPIC_CLASS = "common.anthropic_native_textbased.AnthropicNativeTextbasedModel"

# Per-instance limits used for every run in the paper.
DEFAULT_COST_LIMIT = 5.0
DEFAULT_WALL_TIME = 3600

# Output ceiling per call. Anthropic requires max_tokens; Together's default truncates long
# reasoning turns, and a turn that spends the whole budget thinking returns empty visible content,
# which the text scaffold reads as a format error. OpenAI calls keep the model's default ceiling.
MAX_TOKENS = 32000
# Together: throttling shows up as queue latency rather than 429s, so allow a long single call and
# retry at the LiteLLM level (inside mini's own retry loop).
TOGETHER_NUM_RETRIES = 5
TOGETHER_TIMEOUT_S = 900
# OpenAI: bound each call, and leave retries to mini's logged retry loop by zeroing LiteLLM's and
# the SDK's silent ones.
OPENAI_TIMEOUT_S = 300
OPENAI_NUM_RETRIES = 0
OPENAI_MAX_RETRIES = 0

# Prompt edits for the tool-calling scaffold: (template field, text-scaffold snippet, replacement).
# The wording follows mini's own tool-calling config ("at least one bash tool call").
TOOLCALL_PROMPT_SUBS = [
    ("system_template",
     "Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).\n"
     "Include a THOUGHT section before your command where you explain your reasoning process.\n"
     "Format your response as shown in <format_example>.\n"
     "\n"
     "<format_example>\n"
     "Your reasoning and analysis here. Explain why you want to perform the action.\n"
     "\n"
     "```bash\n"
     "your_command_here\n"
     "```\n"
     "</format_example>\n"
     "\n"
     "Failure to follow these rules will cause your response to be rejected.",
     "You interact with the computer by calling the `bash` tool.\n"
     "Your response MUST include AT LEAST ONE call to that tool, and SHOULD include reasoning\n"
     "text explaining what you are doing and why. If several commands are needed, you may make\n"
     "several tool calls in the same response."),
    ("instance_template",
     "1. Every response must contain exactly one action\n"
     "2. The action must be enclosed in triple backticks",
     "1. Every response must contain at least one `bash` tool call\n"
     "2. Commands go in the tool call's argument, not in the text of your reply"),
    ("instance_template",
     "## Formatting your response\n"
     "\n"
     "Here is an example of a correct response:\n"
     "\n"
     "<example_response>\n"
     "THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.\n"
     "\n"
     "```bash\n"
     "ls -la\n"
     "```\n"
     "</example_response>",
     "## Formatting your response\n"
     "\n"
     "Explain your reasoning in the text of your reply, then call the `bash` tool with the\n"
     "command you want to run. For example: say that you need to see the repository layout,\n"
     "and call `bash` with the command `ls -la`."),
]


def reasoning_overlay(model: str, effort: str) -> dict:
    """model_kwargs that set this effort level for the model's provider."""
    if models.is_together(model):
        return {"extra_body": models.together_effort_body(model, effort)}
    if models.is_anthropic(model):
        if effort == "none":
            return {"thinking": {"type": "disabled"}}
        # display="summarized" returns readable thinking summaries (Sonnet 5 defaults to
        # "omitted", which leaves the thinking blocks empty in the trajectory).
        return {"thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": effort}}
    return {"reasoning_effort": effort}


def apply_toolcall_prompt(config: dict) -> dict:
    """Rewrite the prompt's format rules from ```bash blocks to tool calls, in place.

    Exits if a snippet is missing from base.yaml, so the tool-calling scaffold never runs with
    text-format instructions.
    """
    agent = config.setdefault("agent", {})
    for field, old, new in TOOLCALL_PROMPT_SUBS:
        text = agent.get(field) or ""
        if old not in text:
            raise SystemExit(f"toolcall prompt: expected snippet not found in agent.{field} "
                             f"(starts {old.splitlines()[0][:80]!r}); update TOOLCALL_PROMPT_SUBS")
        agent[field] = text.replace(old, new)
    # The tool-calling model class has no action_regex.
    config.get("model", {}).pop("action_regex", None)
    return config


def build_config(dataset: variants.Dataset, model: str, effort: str,
                 cost_limit: float = DEFAULT_COST_LIMIT, wall_time: int = DEFAULT_WALL_TIME,
                 max_tokens: int | None = None) -> dict:
    """The full mini-swe-agent config for one (dataset, model, effort) run."""
    base = variants.merged_config(dataset, model)
    model_kwargs = reasoning_overlay(model, effort)
    if max_tokens:
        model_kwargs["max_tokens"] = max_tokens
    model_ov: dict = {"model_name": models.api_id(model), "model_kwargs": model_kwargs}
    if models.is_anthropic(model):
        # Native Anthropic client: LiteLLM drops the metered thinking-token count.
        model_ov["model_class"] = NATIVE_ANTHROPIC_CLASS
        model_kwargs.setdefault("max_tokens", MAX_TOKENS)
    elif models.is_together(model):
        model_kwargs.setdefault("max_tokens", MAX_TOKENS)
        model_kwargs.setdefault("num_retries", TOGETHER_NUM_RETRIES)
        model_kwargs.setdefault("timeout", int(TOGETHER_TIMEOUT_S * models.wall_clock_scale(model)))
    else:
        model_kwargs.setdefault("timeout", OPENAI_TIMEOUT_S)
        model_kwargs.setdefault("num_retries", OPENAI_NUM_RETRIES)
        model_kwargs.setdefault("max_retries", OPENAI_MAX_RETRIES)
    agent_ov: dict = {"cost_limit": cost_limit}
    if wall_time:
        agent_ov["wall_time_limit_seconds"] = int(wall_time * models.wall_clock_scale(model))
    config = recursive_merge(base, {"model": model_ov, "agent": agent_ov})
    if variants.scaffold(model) == variants.SCAFFOLD_TOOLCALL:
        apply_toolcall_prompt(config)
    return config


def process_instance_with_image(instance: dict, run_dir: Path, config: dict, pm) -> None:
    """mini's `process_instance`, with the container image set for any environment class.

    mini injects `environment.image` only for its registered environment names ("docker", ...),
    not for a dotted class path like the corrected dataset's, so it is set here with mini's own
    helper. The config is copied because instances run in threads.
    """
    cfg = copy.deepcopy(config)
    cfg.setdefault("environment", {})["image"] = get_swebench_docker_image_name(instance)
    process_instance(instance, run_dir, cfg, pm)


def load_instances(subset: Path | None, limit: int | None, filt: str) -> list[dict]:
    if not INSTANCES.exists():
        sys.exit(f"missing {INSTANCES}; run prepare_dataset.py first")
    rows = [json.loads(line) for line in INSTANCES.open()]
    if subset:
        wanted = {x.strip() for x in subset.read_text().splitlines() if x.strip()}
        rows = [r for r in rows if r["instance_id"] in wanted]
        missing = wanted - {r["instance_id"] for r in rows}
        if missing:
            logger.warning(f"{len(missing)} subset id(s) not in the dataset, e.g. {sorted(missing)[:3]}")
    if filt:
        rows = [r for r in rows if re.search(filt, r["instance_id"])]
    rows.sort(key=lambda r: r["instance_id"])
    if limit:
        rows = rows[:limit]
    if not rows:
        sys.exit("no instances selected")
    return rows


def manifest(dataset: variants.Dataset, model: str, effort: str, config: dict,
             instances: list[dict]) -> dict:
    return {
        "stage": "generation",
        "benchmark": dataset.benchmark,
        "dataset": dataset.key,
        "dataset_title": dataset.title,
        "environment_note": dataset.env_note,
        **({"egress_policy": sealed_docker.policy()} if dataset.egress_sealed else {}),
        **dataset.tags,
        "scaffold": variants.scaffold(model),
        "model": model,
        "litellm_model": models.api_id(model),
        "effort": effort,
        "cost_limit": config["agent"].get("cost_limit"),
        "step_limit": config["agent"].get("step_limit"),
        "wall_time_limit_seconds": config["agent"].get("wall_time_limit_seconds"),
        "wall_clock_scale": models.wall_clock_scale(model),
        "call_timeout_seconds": config["model"]["model_kwargs"].get("timeout"),
        "max_tokens": config["model"]["model_kwargs"].get("max_tokens"),
        "rates": pricing.rates(model),
        "instances": [i["instance_id"] for i in instances],
        "n_instances": len(instances),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_effort(dataset: variants.Dataset, model: str, effort: str, instances: list[dict],
               workers: int, cost_limit: float, wall_time: int, max_tokens: int | None,
               redo: bool) -> None:
    config = build_config(dataset, model, effort, cost_limit, wall_time, max_tokens)
    run_dir = paths.run_dir(dataset.benchmark, models.name_slug(model), effort)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.merged.yaml").write_text(yaml.dump(config, sort_keys=False, allow_unicode=True))
    add_file_handler(run_dir / "minisweagent.log")

    todo = list(instances)
    preds = run_dir / "preds.json"
    if not redo and preds.exists():
        done = set(json.loads(preds.read_text()))
        todo = [i for i in todo if i["instance_id"] not in done]
        logger.info(f"[{effort}] resuming: {len(done)} done, {len(todo)} to go")

    print(f"\n=== {dataset.title}  model={model}  effort={effort}  scaffold={variants.scaffold(model)}  "
          f"instances={len(instances)} (to run: {len(todo)})  -> {run_dir} ===")

    if todo:
        pm = RunBatchProgressManager(len(todo), run_dir / f"exit_statuses_{time.time()}.yaml")

        def drain(futs):
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fut.result()
                except concurrent.futures.CancelledError:
                    pass
                except Exception as e:
                    iid = futs[fut]
                    logger.error(f"Error in future for {iid}: {e}", exc_info=True)
                    pm.on_uncaught_exception(iid, e)

        with Live(pm.render_group, refresh_per_second=4):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(process_instance_with_image, inst, run_dir, config, pm):
                        inst["instance_id"] for inst in todo}
                try:
                    drain(futs)
                except KeyboardInterrupt:
                    logger.info("Cancelling pending jobs (^C again to force).")
                    for f in futs:
                        if not f.running() and not f.done():
                            f.cancel()
                    drain(futs)

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest(dataset, model, effort, config, instances), indent=2))
    a = report.write_cost_summary(run_dir, model)["aggregate"]
    print(f"  [{effort}] submitted {a['n_submitted']}/{a['n_instances']}  "
          f"mean $ from tokens={a['mean_cost_from_tokens']}  mean calls={a['mean_api_calls']}")
    share = a["thinking_call_share"]
    if share is not None:
        print(f"    {a['calls_with_reasoning']}/{a['n_lm_calls']} LM calls produced reasoning "
              f"({share * 100:.1f}%)")
    if effort == "none" and share:
        print("    WARNING: a `none` run should have no reasoning calls; the provider did not "
              "honour the thinking-off request on some calls")
    if dataset.git_history_stripped and a["n_env_setup_ok"] != a["n_instances"]:
        print(f"    WARNING: only {a['n_env_setup_ok']}/{a['n_instances']} instances report "
              f"STRIP-OK; the rest failed to sanitize (see minisweagent.log)")
    if dataset.egress_sealed and a["n_egress_sealed"] != a["n_instances"]:
        print(f"    WARNING: only {a['n_egress_sealed']}/{a['n_instances']} instances recorded "
              f"an egress policy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model slug from common/models.py")
    ap.add_argument("--efforts", default="",
                    help="comma list of effort levels (default: every level the model supports)")
    variants.add_argument(ap)
    ap.add_argument("--subset", type=Path, default=PILOT,
                    help="file of instance ids, one per line (default: the 44-instance pilot)")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N instances")
    ap.add_argument("--filter", default="", help="regex on instance_id")
    ap.add_argument("--workers", type=int, default=4, help="instances run in parallel")
    ap.add_argument("--cost-limit", type=float, default=DEFAULT_COST_LIMIT,
                    help=f"per-instance $ cap (default {DEFAULT_COST_LIMIT:g}; 0 disables)")
    ap.add_argument("--wall-time", type=int, default=DEFAULT_WALL_TIME,
                    help=f"per-instance wall-clock cap in seconds, scaled per model by "
                         f"common.models.wall_clock_scale (default {DEFAULT_WALL_TIME}; 0 disables)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help=f"per-call output ceiling (default {MAX_TOKENS} for Anthropic and "
                         f"Together, the model default for OpenAI)")
    ap.add_argument("--redo", action="store_true",
                    help="re-run instances already in preds.json (overwrites their trajectories)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each effort's merged config and run directory, then exit")
    args = ap.parse_args()

    dataset = variants.resolve(args.dataset)
    efforts = ([e.strip() for e in args.efforts.split(",") if e.strip()]
               or models.valid_efforts(args.model))
    models.check_efforts(args.model, efforts)

    if args.dry_run:
        for effort in efforts:
            config = build_config(dataset, args.model, effort, args.cost_limit, args.wall_time,
                                  args.max_tokens)
            print(f"# {dataset.key} / {args.model} / {effort}  scaffold={variants.scaffold(args.model)}"
                  f"  configs={[p.name for p in variants.config_paths(dataset, args.model)]}")
            print(f"# -> {paths.run_dir(dataset.benchmark, models.name_slug(args.model), effort)}")
            print(yaml.dump(config, sort_keys=False, allow_unicode=True))
        return

    paths.load_env(required=[models.resolve(args.model).key_env])
    if pricing.rates(args.model) is None:
        print(f"WARNING: no rates for {args.model} in common/pricing.py; mini's cost tracking "
              "may abort. Token counts are still recorded.")
    else:
        for line in pricing.register_litellm():
            print(f"  rate note (this run uses common/pricing.py): {line}")
    instances = load_instances(args.subset, args.limit, args.filter)
    print(f"selected {len(instances)} instance(s); efforts={efforts}; model={args.model}")
    print(f"dataset: {dataset.title}: {dataset.env_note}")
    if dataset.egress_sealed:
        pol = sealed_docker.policy()
        print(f"  egress allow-list {pol['sha256_16']} ({pol['n_allowed_hosts']} hosts)")
    for effort in efforts:
        run_effort(dataset, args.model, effort, instances, args.workers, args.cost_limit,
                   args.wall_time, args.max_tokens, args.redo)
    print(f"\ndone -> {paths.data_root(dataset.benchmark) / models.name_slug(args.model)}")
    print(f"next: evaluate.py --model {args.model} --dataset {dataset.key} --run-eval, "
          f"then report.py --dataset {dataset.key}")


if __name__ == "__main__":
    main()
