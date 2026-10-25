"""Per-(model, effort, instance) features of the SWE-Bench Pro trajectories.

    python behavior/leakage/extract_traj_features.py                  # both datasets
    python behavior/leakage/extract_traj_features.py --which pro-corrected

Walks the mini-swe-agent `.traj.json` files of the original (`swebench-pro`) and corrected
(`swebench-pro-corrected`) result trees and writes one JSON line per trajectory to
<OUTPUT_ROOT>/leakage/pro_features.jsonl and pro_corrected_features.jsonl. The features are
syntactic (command classes, thought length, token counts, outcome); `traj_path` is relative
to DATA_ROOT and is what oracle_hunt.py and net_fetch_intent.py read the commands from.

`cost` is recomputed at the common/pricing.py rates for OpenAI models (exact from their
token counts, with no cache-write premium); other models keep the harness cost.
`cost_billed` always keeps the harness figure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root, for `common`
from common import models as model_registry, paths, pricing  # noqa: E402

TREES = {
    "pro": ("swebench-pro", "swebench_pro", "pro_features.jsonl"),
    "pro-corrected": ("swebench-pro-corrected", "swebench_pro_corrected",
                      "pro_corrected_features.jsonl"),
}


# ---------------------------------------------------------------- command taxonomy

# Ordered: first match wins, so `python -m pytest` lands in TEST not RUN.
CMD_PATTERNS = [
    ("test", r"\b(pytest|go test|npm (run )?test|jest|mocha|tox|nosetests|unittest|"
             r"npx jest|yarn test|cargo test|ctest|make test|./test|rspec|"
             r"gotestsum|vitest|_test\.go|test_.*\.py)\b"),
    ("git", r"\bgit\s+(diff|add|status|log|show|stash|checkout|commit|apply)\b"),
    ("edit", r"(apply_patch|>\s*/app/|>\s*\S+\.(py|js|ts|go|java|rb|c|h|cpp|rs|json|yaml|yml)\b"
             r"|\bpatch\s+-p|python(3)?\s+-\s*<<|cat\s*>\s*|tee\s+|sed\s+-i|"
             r"perl\s+-[a-z]*i|<<\s*'?(EOF|PATCH|PY)'?)"),
    ("search", r"\b(grep|rg|ag|find|ls|fd|locate|tree)\b"),
    ("read", r"\b(cat|sed\s+-n|head|tail|less|more|awk\s+'NR|wc\s+-l|nl\s)\b"),
    ("build", r"\b(make|cmake|go build|npm (run )?build|tsc|cargo build|gcc|g\+\+|"
              r"pip install|npm install|apt-get|go mod|yarn install|uv pip)\b"),
    ("run", r"\b(python3?|node|go run|ruby|bash|sh|\./)\b"),
]

SUBMIT_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def reprice(row: dict, model: str) -> dict:
    """Recompute an OpenAI row's `cost` at the price table's rates; keep the harness cost."""
    row["cost_billed"] = row.get("cost")
    if model_registry.resolve(model).family != "openai":
        return row
    priced = pricing.cost_from_usage(model, prompt_tokens=row.get("input_tokens"),
                                     output_tokens=row.get("output_tokens"),
                                     cache_read=row.get("cached_tokens"))
    if priced is not None:
        row["cost"] = round(priced, 6)
    return row


def classify(cmd: str) -> str:
    for name, pat in CMD_PATTERNS:
        if re.search(pat, cmd, re.I):
            return name
    return "other"


def cmd_features(commands: list[str]) -> dict:
    kinds = Counter(classify(c) for c in commands)
    return {
        "n_commands": len(commands),
        **{f"n_{k}": kinds.get(k, 0) for k, _ in CMD_PATTERNS},
        "n_other": kinds.get("other", 0),
        # position of the first / last test-like command, normalised 0..1
        "first_test_frac": next(
            (i / max(1, len(commands)) for i, c in enumerate(commands)
             if classify(c) == "test"), None),
        "n_distinct_commands": len(set(commands)),
        "repeat_command_frac": (
            1 - len(set(commands)) / len(commands)) if commands else None,
    }


# ---------------------------------------------------------------- SWE-Bench-Pro

def load_pro_evals(model_dir: Path) -> dict[str, dict[str, bool]]:
    out = {}
    for f in sorted((model_dir / "eval").glob("*/eval_results.json")):
        out[f.parent.name] = json.load(open(f))
    return out


def pro_row(traj_path: Path) -> dict:
    j = json.load(open(traj_path))
    msgs = j.get("messages", [])
    commands, thoughts = [], []
    in_tok = out_tok = reason_tok = cached_tok = 0
    n_calls = 0
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        n_calls += 1
        extra = m.get("extra") or {}
        for a in extra.get("actions") or []:
            if a.get("command"):
                commands.append(a["command"])
        content = m.get("content") or ""
        # THOUGHT text = everything before the first fenced block
        thoughts.append(content.split("```")[0])
        usage = ((extra.get("response") or {}).get("usage") or {})
        in_tok += usage.get("prompt_tokens", 0) or 0
        out_tok += usage.get("completion_tokens", 0) or 0
        ctd = usage.get("completion_tokens_details") or {}
        reason_tok += ctd.get("reasoning_tokens", 0) or 0
        ptd = usage.get("prompt_tokens_details") or {}
        cached_tok += ptd.get("cached_tokens", 0) or 0

    # the final patch, if the agent submitted
    patch = ""
    for m in reversed(msgs):
        if m.get("role") == "user" and SUBMIT_MARK in str(m.get("content", "")):
            patch = str(m["content"])
            break

    non_submit = [c for c in commands if SUBMIT_MARK not in c]
    row = {
        "instance_id": j.get("instance_id") or traj_path.parent.name,
        "api_calls": (j.get("info", {}).get("model_stats") or {}).get("api_calls", n_calls),
        "cost": (j.get("info", {}).get("model_stats") or {}).get("instance_cost"),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "reasoning_tokens": reason_tok,
        "cached_tokens": cached_tok,
        "n_assistant_msgs": n_calls,
        "thought_chars_total": sum(len(t) for t in thoughts),
        "thought_chars_mean": (sum(len(t) for t in thoughts) / len(thoughts)) if thoughts else 0,
        "patch_chars": len(patch),
        **cmd_features(non_submit),
    }
    return row


def collect_pro(root: Path, bench: str) -> list[dict]:
    """One feature row per trajectory under a Pro result tree."""
    if not root.exists():
        return []
    rows = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        evals = load_pro_evals(model_dir)
        for eff_dir in sorted(p for p in model_dir.iterdir()
                              if p.is_dir() and p.name != "eval"):
            resolved_map = evals.get(eff_dir.name, {})
            summary = {}
            cs = eff_dir / "cost_summary.json"
            if cs.exists():
                summary = {r["instance_id"]: r
                           for r in json.load(open(cs)).get("per_instance", [])}
            for inst_dir in sorted(eff_dir.glob("instance_*")):
                tp = next(iter(inst_dir.glob("*.traj.json")), None)
                if tp is None:
                    continue
                r = reprice(pro_row(tp), model_dir.name)
                iid = r["instance_id"]
                r.update(
                    bench=bench,
                    model=model_dir.name,
                    effort=eff_dir.name,
                    resolved=resolved_map.get(iid),
                    exit_status=(summary.get(iid) or {}).get("exit_status"),
                    patch_len=(summary.get(iid) or {}).get("patch_len"),
                    repo=iid.replace("instance_", "").split("__")[0],
                    traj_path=str(tp.relative_to(paths.DATA_ROOT)),
                )
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Extract per-trajectory features of the SWE-Bench "
                                             "Pro runs.")
    ap.add_argument("--which", choices=["pro", "pro-corrected", "both"], default="both",
                    help="original dataset, corrected dataset, or both (default); each is "
                         "written to its own table")
    a = ap.parse_args()
    out = paths.output_dir("leakage", create=True)
    for which, (benchmark, bench, name) in TREES.items():
        if a.which not in (which, "both"):
            continue
        rows = collect_pro(paths.data_root(benchmark), bench)
        with open(out / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{which}: {len(rows)} rows -> {out / name}")


if __name__ == "__main__":
    main()
