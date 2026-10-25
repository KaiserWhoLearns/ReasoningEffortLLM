"""Turn agent trajectories into compact, blinded Act-onomy skeletons. No API calls.

A skeleton is the Act-onomy annotated-JSON schema with every annotation field left as
TODO: one turn per assistant message, holding the observation that preceded it (`obs`),
the agent's prose (`thought`) and the command(s) it issued (`action`). The annotator
(run_batch.py) fills in the rest.

    python behavior/actonomy/extract.py --benchmark tb --sample 30
    python behavior/actonomy/extract.py --benchmark pro-corrected --sample 30

SWE-Bench Pro trajectories are mini-swe-agent `.traj.json` files (parsed here);
Terminal-Bench trials are harbor/Terminus-2 runs (parsed in tb.py).

Conventions that the annotation relies on:
- `obs` is the observation *preceding* the thought, as in Act-onomy's (observation,
  thought, action) turn: the `recover` phase and Reflection both need the failure signal
  of the previous command in view.
- Observations are truncated head-biased: a mini observation opens with
  `<returncode>N</returncode>`, which is the failure signal; `rc` is also kept in `_x`.
- The annotator is blind to the outcome and to which model and effort produced the run;
  both are kept in `skeletons/_manifest.json` for the analysis. The subject's own model
  identifiers are redacted from the visible text, and any remaining model-family tokens
  are counted into the manifest (`leak_hits`).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402
import tb  # noqa: E402

EXTRACTOR_VERSION = "1"
SUBSTRATES = ("content", "both")
OBS_HEAD, OBS_TAIL = 600, 200
THOUGHT_CAP, ISSUE_CAP = 20_000, 4_000

# The `THOUGHT:` header the SWE-Bench prompt asks for; models use it inconsistently, so it
# is stripped rather than relied on.
_THOUGHT_PREFIX = re.compile(r"^\s*THOUGHT\s*:?\s*", re.IGNORECASE)
_RETURNCODE = re.compile(r"^<returncode>(-?\d+)</returncode>")
_OBS_ROLES = ("user", "tool", "observation")


def _strip_trailing_fences(text: str) -> str:
    """Drop fenced blocks that trail the prose, leaving inline examples alone."""
    out = text.rstrip()
    while True:
        close = out.rfind("```")
        if close == -1 or out[close + 3:].strip():
            return out
        open_ = out.rfind("```", 0, close)
        if open_ == -1:
            return out
        out = out[:open_].rstrip()


def narration(content: str | None) -> str:
    """The agent's visible prose: `THOUGHT:` header and trailing command block removed."""
    if not content:
        return ""
    return _strip_trailing_fences(_THOUGHT_PREFIX.sub("", content, count=1)).strip()


def commands(msg: dict) -> list[str]:
    """Bash command(s) of an assistant message: `extra.actions`, else `tool_calls`."""
    out: list[str] = []
    for a in (msg.get("extra") or {}).get("actions") or []:
        if a.get("command"):
            out.append(a["command"])
    if out:
        return out
    for tc in msg.get("tool_calls") or []:
        args = (tc.get("function") or {}).get("arguments")
        if not args:
            continue
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict) and parsed.get("command"):
            out.append(parsed["command"])
    return out


def head_tail(text: str, head: int, tail: int) -> tuple[str, bool]:
    """Head-biased truncation. Returns (text, was_truncated)."""
    if len(text) <= head + tail:
        return text, False
    dropped = len(text) - head - tail
    return f"{text[:head]}\n…[{dropped} chars cut]…\n{text[-tail:] if tail else ''}", True


def returncode(obs: str) -> int | None:
    m = _RETURNCODE.match(obs)
    return int(m.group(1)) if m else None


# Model-family tokens that would tell a blinded annotator whose run it is reading. Word-
# bounded; some hits are false positives (`opus` is also an audio codec), so they are
# recorded rather than treated as failures.
_LEAK_TOKENS = ("claude", "anthropic", "sonnet", "opus", "haiku", "gpt", "openai",
                "deepseek", "glm", "zhipu", "kimi", "moonshot", "inkling", "qwen",
                "llama", "gemini", "mistral")
_LEAK_RE = re.compile(r"\b(" + "|".join(_LEAK_TOKENS) + r")\b", re.IGNORECASE)


def leak_scan(skel: dict[str, Any]) -> dict[str, int]:
    """Count model-family tokens in the text the annotator will read."""
    hits: dict[str, int] = {}
    chunks = [skel.get("issue_text", "")]
    for t in skel.get("turns", []):
        chunks += [t.get("obs", ""), t.get("thought", ""), t.get("action", "")]
    for chunk in chunks:
        for m in _LEAK_RE.finditer(chunk or ""):
            tok = m.group(0).lower()
            hits[tok] = hits.get(tok, 0) + 1
    return hits


def identity_strings(model_dir_name: str, harness_model_name: str | None) -> list[str]:
    """Strings that name this trajectory's own subject model.

    An agent can print its own harness files (e.g. a Terminal-Bench trajectory.json naming
    the model). Only the subject's own identifiers are redacted, never provider names in
    general, since tasks can legitimately mention them in code.
    """
    out = {model_dir_name}
    if harness_model_name:
        out.add(harness_model_name)
        out.add(harness_model_name.replace("/", "_"))
        tail = harness_model_name.rsplit("/", 1)[-1]
        if len(tail) >= 4:
            out.add(tail)
    # Provider-prefixed directory slugs (`openai_gpt-5.6-sol`): the model half alone.
    for sep in ("_", "/"):
        parts = model_dir_name.split(sep)
        if len(parts) > 1 and len(parts[-1]) >= 4:
            out.add(parts[-1])
    return sorted((v for v in out if v), key=len, reverse=True)


def redact_identity(text: str, needles: list[str]) -> tuple[str, int]:
    """Replace the subject's own identifiers. Returns (text, n_replacements)."""
    if not text:
        return text, 0
    n = 0
    for needle in needles:
        pat = re.compile(re.escape(needle), re.IGNORECASE)
        text, k = pat.subn("<subject-model-redacted>", text)
        n += k
    return text, n


def issue_text(msgs: list[dict], cap: int) -> str:
    """Task statement from the first user message, unwrapped from `<pr_description>`."""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            raw = m.get("content") or ""
            i, j = raw.find("<pr_description>"), raw.find("</pr_description>")
            body = raw[i + len("<pr_description>"): j].strip() if 0 <= i < j else raw.strip()
            return head_tail(body, cap, 0)[0]
    return ""


def parse_traj(path: Path, *, substrate: str) -> dict[str, Any]:
    """Read one mini-swe-agent `.traj.json` into turns + task text + extraction stats."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ValueError(f"{path}: not a mini-swe-agent trajectory (no `messages` list)")
    msgs = payload["messages"]
    if any(isinstance(m, dict) and isinstance(m.get("output"), list) for m in msgs):
        raise ValueError(f"{path}: Responses-API serialisation is not supported")

    turns: list[dict[str, Any]] = []
    pending = "Task delivered: PR description + instructions. No command run yet."
    pending_rc: int | None = None
    seen_first_user = False
    submitted = False
    n_trunc_thought = 0

    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "exit":
            submitted = bool((m.get("content") or "").strip())
            continue
        if role == "user" and not seen_first_user:
            seen_first_user = True          # the task itself, not an observation
            continue
        if role != "assistant":
            continue

        content_txt = narration(m.get("content"))
        reasoning_txt = (m.get("reasoning_content") or "").strip()
        if substrate == "content":
            thought = content_txt
        else:
            thought = "\n\n".join(p for p in (reasoning_txt, content_txt) if p)
        thought, cut = head_tail(thought, THOUGHT_CAP, 0)
        n_trunc_thought += cut

        cmds = commands(m)
        # The observations after this turn become the next turn's obs.
        nxt_parts: list[str] = []
        for nxt in msgs[i + 1:]:
            if not isinstance(nxt, dict):
                continue
            r = nxt.get("role")
            if r == "assistant" or r == "exit":
                break
            if r in _OBS_ROLES:
                nxt_parts.append((nxt.get("content") or "").strip())
        nxt_obs = "\n\n".join(p for p in nxt_parts if p)

        obs_txt, obs_cut = head_tail(pending, OBS_HEAD, OBS_TAIL)
        turns.append({
            "n": len(turns) + 1,
            "phase": layout.TODO,
            "phase_label": layout.TODO,
            "headline": layout.TODO,
            "obs": obs_txt,
            "thought": thought,
            "action": "\n".join(cmds),
            "quotes": [],
            "_x": {
                "rc": pending_rc,
                "obs_chars": len(pending),
                "obs_truncated": obs_cut,
                "content_chars": len(content_txt),
                "reasoning_chars": len(reasoning_txt),
                "thought_truncated": cut,
                "n_actions": len(cmds),
            },
        })
        pending, pending_rc = nxt_obs, returncode(nxt_obs)

    return {
        "issue_text": issue_text(msgs, ISSUE_CAP),
        "turns": turns,
        "stats": {
            "n_turns": len(turns),
            "submitted": submitted,
            "empty_thought_turns": sum(1 for t in turns if not t["thought"].strip()),
            "empty_action_turns": sum(1 for t in turns if not t["action"].strip()),
            "truncated_thought_turns": n_trunc_thought,
            "reasoning_turns": sum(1 for t in turns if t["_x"]["reasoning_chars"] > 0),
            "failed_obs_turns": sum(1 for t in turns if (t["_x"]["rc"] or 0) != 0),
            "trajectory_format": payload.get("trajectory_format"),
        },
    }


def build_skeleton(parsed: dict[str, Any], *, instance_id: str, substrate: str,
                   scaffold: str) -> dict[str, Any]:
    """Wrap parsed turns in Act-onomy's annotated-JSON schema, blinded, TODOs unfilled.

    `model` names the (withheld) subject; the annotator records itself in
    `annotator_model`.
    """
    return {
        "instance_id": instance_id,
        "model": f"{scaffold} (subject identity withheld)",
        "outcome": "withheld",
        "issue_text": parsed["issue_text"],
        "summary": layout.TODO,
        "quote_target": "thought",
        "findings": [],
        "turns": parsed["turns"],
        "annotator_model": layout.TODO,
        "_extract": {
            "extractor_version": EXTRACTOR_VERSION,
            "substrate": substrate,
            "obs_convention": "preceding",
            "obs_head": OBS_HEAD,
            "obs_tail": OBS_TAIL,
            "blind": True,
            "anonymized": True,
            **parsed["stats"],
        },
    }


# --- discovery ----------------------------------------------------------------------
def load_evals(model_dir: Path) -> dict[str, dict[str, bool]]:
    """`eval/<effort>/eval_results.json` -> {effort: {instance: resolved}}."""
    out: dict[str, dict[str, bool]] = {}
    for f in sorted((model_dir / "eval").glob("*/eval_results.json")):
        out[f.parent.name] = json.loads(f.read_text())
    return out


def shared_pool(bench: str) -> tuple[list[str], str]:
    """(instances present in every cell of `bench`, one-line description of that basis).

    Not conditioned on outcome: the sample covers solved and failed runs alike.
    """
    root = layout.data_root(bench)
    if bench in layout.TB_BENCHMARKS:
        pool = tb.shared_tasks(root)
        basis = "tasks present in every Terminal-Bench cell"
    else:
        sets = []
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for ev in load_evals(model_dir).values():
                if ev:
                    sets.append(set(ev))
        pool = sorted(set.intersection(*sets)) if sets else []
        basis = "instances scored in every cell (NOT conditioned on outcome)"
    if not pool:
        raise SystemExit(f"no instances found for {bench!r}")
    return pool, basis


def resolve_sample(bench: str, n: int, *, seed: int) -> list[str]:
    """The fixed instance sample: loaded from sample.json, or drawn once and saved there.

    Every cell annotates the same instances, so the comparison is paired by construction.
    """
    path = layout.sample_path(bench)
    if path.exists():
        saved = json.loads(path.read_text())
        if len(saved["instances"]) != n:
            print(f"[warn] {layout.rel(path)} holds {len(saved['instances'])} instances but "
                  f"--sample {n} was requested; keeping the saved set (delete the file to "
                  f"redraw)", file=sys.stderr)
        return saved["instances"]

    pool, basis = shared_pool(bench)
    if n > len(pool):
        print(f"[warn] only {len(pool)} instances are shared across all cells; "
              f"sampling all of them instead of {n}", file=sys.stderr)
        n = len(pool)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(pool, n))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"benchmark": bench, "seed": seed, "n": n, "pool_size": len(pool),
         "basis": basis, "instances": chosen}, indent=2))
    print(f"drew {n} of {len(pool)} shared instances (seed {seed}) -> {layout.rel(path)}")
    return chosen


def iter_work(bench: str, *, models: list[str] | None, efforts: list[str] | None,
              instances: list[str]) -> Iterator[tuple[str, str, str, Path, str]]:
    """Yield (model, effort, instance_id, traj_path, outcome) for SWE-Bench Pro."""
    wanted = set(instances)
    for model_dir in sorted(p for p in layout.data_root(bench).iterdir() if p.is_dir()):
        model = model_dir.name
        if models and model not in models:
            continue
        evals = load_evals(model_dir)
        for effort in sorted(e for e in evals if not efforts or e in efforts):
            for inst, resolved in sorted(evals[effort].items()):
                if inst not in wanted:
                    continue
                traj = model_dir / effort / inst / f"{inst}.traj.json"
                if traj.exists():
                    yield model, effort, inst, traj, "resolved" if resolved else "unresolved"


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract blinded Act-onomy skeletons from "
                                             "agent trajectories.")
    ap.add_argument("--benchmark", default="pro-corrected", choices=sorted(layout.BENCHMARKS))
    ap.add_argument("--model", action="append", dest="models",
                    help="restrict to a result-tree model directory; repeatable")
    ap.add_argument("--effort", action="append", dest="efforts",
                    help="restrict to an effort level; repeatable")
    ap.add_argument("--sample", type=int, default=30,
                    help="size of the instance sample shared by every cell (default 30); "
                         "drawn once into sample.json and reused afterwards")
    ap.add_argument("--seed", type=int, default=42, help="seed for drawing the sample")
    ap.add_argument("--substrate", choices=SUBSTRATES,
                    help="text annotated as `thought`: `content` (narrated prose) or `both` "
                         "(reasoning_content + content). Default: the paper's setting, `both` "
                         "for the corrected-Pro models in layout.BOTH_FIELD_MODELS, else "
                         "`content`. Terminal-Bench has no separate reasoning field")
    ap.add_argument("--limit", type=int, help="stop after N trajectories")
    ap.add_argument("--force", action="store_true", help="overwrite existing skeletons")
    a = ap.parse_args()

    out_root = layout.skeleton_root(a.benchmark)
    manifest_file = layout.manifest_path(a.benchmark)
    manifest: dict[str, Any] = {"benchmark": a.benchmark, "entries": {}}
    if manifest_file.exists():                     # merge: extraction is often per model
        manifest = json.loads(manifest_file.read_text())
        manifest.setdefault("entries", {})
    written = skipped = failed = 0
    turns = chars = 0
    total_redactions = 0
    leak_tally: dict[str, int] = {}
    leak_files: dict[str, set] = {}
    agg = {"empty_thought_turns": 0, "empty_action_turns": 0,
           "truncated_thought_turns": 0, "reasoning_turns": 0}
    per_sub: dict[str, int] = {}

    is_tb = a.benchmark in layout.TB_BENCHMARKS
    sample = resolve_sample(a.benchmark, a.sample, seed=a.seed)
    if is_tb:
        work = tb.iter_work(layout.data_root(a.benchmark), models=a.models,
                            efforts=a.efforts, tasks=sample)
    else:
        work = iter_work(a.benchmark, models=a.models, efforts=a.efforts, instances=sample)

    for model, effort, inst, traj, outcome in work:
        if a.limit and written + skipped >= a.limit:
            break
        sub = layout.submission(model, effort)
        out = out_root / sub / f"{inst}.skeleton.json"
        if out.exists() and not a.force:
            skipped += 1
            continue
        substrate = a.substrate or layout.default_substrate(a.benchmark, model)
        try:
            if is_tb:
                parsed = tb.parse_trial(traj, head_tail=head_tail, obs_head=OBS_HEAD,
                                        obs_tail=OBS_TAIL, thought_cap=THOUGHT_CAP,
                                        issue_cap=ISSUE_CAP)
            else:
                parsed = parse_traj(traj, substrate=substrate)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[err] {traj}: {e}", file=sys.stderr)
            failed += 1
            continue
        if not parsed["turns"]:
            print(f"[err] {traj}: no turns extracted", file=sys.stderr)
            failed += 1
            continue
        needles = identity_strings(model, parsed.get("harness_model_name"))
        n_redacted = 0
        for t in parsed["turns"]:
            for field in ("obs", "thought", "action"):
                t[field], k = redact_identity(t[field], needles)
                n_redacted += k
        parsed["issue_text"], k = redact_identity(parsed["issue_text"], needles)
        n_redacted += k
        parsed["stats"]["identity_redactions"] = n_redacted
        total_redactions += n_redacted

        skel = build_skeleton(parsed, instance_id=inst, substrate=substrate,
                              scaffold="terminus-2" if is_tb else "mini-swe-agent")
        entry = {
            "subject_model": model, "effort": effort, "outcome": outcome,
            "source": layout.rel(traj), "n_turns": parsed["stats"]["n_turns"],
        }
        leaks = leak_scan(skel)
        if leaks:
            entry["leak_hits"] = leaks
            for tok, c in leaks.items():
                leak_tally[tok] = leak_tally.get(tok, 0) + c
                leak_files.setdefault(tok, set()).add(f"{sub}/{inst}")
        manifest["entries"][f"{sub}/{inst}"] = entry
        out.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(skel, indent=2, ensure_ascii=False)
        out.write_text(body)
        written += 1
        turns += parsed["stats"]["n_turns"]
        chars += len(body)
        per_sub[sub] = per_sub.get(sub, 0) + 1
        for k in agg:
            agg[k] += parsed["stats"][k]

    if written:
        manifest["blind"] = True
        manifest["anonymized"] = True
        manifest["extractor_version"] = EXTRACTOR_VERSION
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {written} skeletons ({skipped} already present, {failed} failed) "
          f"-> {out_root}")
    if written:
        print(f"  {len(manifest['entries'])} entries in {layout.rel(manifest_file)} "
              f"(never mount that file into the annotator)")
        print(f"  {turns} turns, {chars/1e6:.1f} MB, ~{chars/4/written:,.0f} tok/trajectory")
        print(f"  {len(per_sub)} submissions: " +
              ", ".join(f"{k}={v}" for k, v in sorted(per_sub.items())[:6]) +
              (" …" if len(per_sub) > 6 else ""))
        print(f"  turns with empty thought={agg['empty_thought_turns']}, "
              f"empty action={agg['empty_action_turns']}, "
              f"thought truncated={agg['truncated_thought_turns']}, "
              f"reasoning_content present={agg['reasoning_turns']}")
        if total_redactions:
            print(f"  redacted {total_redactions} occurrence(s) of a trajectory's own model "
                  f"identifier from annotator-visible text")
        if leak_tally:
            print(f"  leak scan: {sum(leak_tally.values())} model-family token(s) remain in "
                  f"annotator-visible text across {len(set().union(*leak_files.values()))} "
                  f"file(s); some are false positives:")
            for tok, c in sorted(leak_tally.items(), key=lambda kv: -kv[1]):
                print(f"    {tok:10s} {c:4d}x in {len(leak_files[tok])} file(s)")
        else:
            print("  leak scan: no model-family tokens in annotator-visible text")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
