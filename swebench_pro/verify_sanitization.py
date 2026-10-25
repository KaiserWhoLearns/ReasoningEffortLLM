#!/usr/bin/env python3
"""SWE-Bench Pro-Corrected: check the git-history strip in each container, without running an agent.

Spends no tokens. For each selected instance it starts the instance's image with the sweep's
container flags, probes `/app`, runs the same sanitizer the run uses (environment.setup_command
from configs/strip_history.yaml), probes again, and checks:

The fix is gone:
  * the instance's gold fix commit (the sha in its id) is not a known object afterwards, in /app or
    in any nested repository, and does not appear in `git log --all`;
  * one commit, one ref, no remotes, tags or stash; no unreachable objects; `git fsck` is clean.

The task is unchanged:
  * HEAD^{tree}, the tracked-file count and the untracked-file count are unchanged;
  * `git status --porcelain` and the body of `git diff HEAD` hash the same before and after (some
    images ship pre-existing changes, which are part of the task);
  * /app is still the work-tree root.

The egress seal is checked separately by verify_egress.py. The sanitizer enforces the same
invariants at run time (a failing instance is aborted), so this is the cheap way to find a failing
instance before a sweep.

Usage (needs Docker; images are pulled on demand):
    python verify_sanitization.py --limit 2
    python verify_sanitization.py --workers 6          # the 44-instance pilot

Writes `<output>/swebench_pro_corrected/sanitization_check.json` and prints a table.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for `common`
from common import paths

import variants

HERE = Path(__file__).resolve().parent
INSTANCES = paths.DATASETS_ROOT / "swebench_pro" / "mini_instances.jsonl"
PILOT = HERE / "subsets" / "pilot.txt"
DATASET = variants.resolve("pro-corrected")
GOLD_SHA = re.compile(r"-([0-9a-f]{40})")

# The whole repository state in one `docker exec`, parsed back into a dict. Keys must not contain '='.
PROBE = r"""
cd /app 2>/dev/null || { echo "toplevel="; exit 0; }
echo "toplevel=$(git rev-parse --show-toplevel 2>/dev/null)"
echo "head=$(git rev-parse HEAD 2>/dev/null)"
echo "tree=$(git rev-parse "HEAD^{tree}" 2>/dev/null)"
echo "commits=$(git rev-list --count HEAD 2>/dev/null)"
echo "tracked=$(git ls-files 2>/dev/null | wc -l)"
echo "dirty=$(git status --porcelain 2>/dev/null | wc -l)"
echo "diff_head_lines=$(git diff HEAD 2>/dev/null | wc -l)"
echo "status_sha=$(git status --porcelain 2>/dev/null | sha256sum | cut -d' ' -f1)"
echo "diff_sha=$(git diff HEAD 2>/dev/null | sha256sum | cut -d' ' -f1)"
echo "diff_body_sha=$(git diff HEAD 2>/dev/null | grep -v '^index ' | sha256sum | cut -d' ' -f1)"
echo "untracked=$(git ls-files --others --exclude-standard 2>/dev/null | wc -l)"
echo "refs=$(git for-each-ref 2>/dev/null | wc -l)"
echo "remotes=$(git remote 2>/dev/null | wc -l)"
echo "tags=$(git tag 2>/dev/null | wc -l)"
echo "stash=$(git stash list 2>/dev/null | wc -l)"
echo "objects=$(git cat-file --batch-all-objects --batch-check 2>/dev/null | wc -l)"
echo "reachable=$(git rev-list --objects --all 2>/dev/null | wc -l)"
echo "nested=$(find /app -mindepth 2 -name .git -prune 2>/dev/null | wc -l)"
echo "nested_dirs=$(find /app -mindepth 2 -name .git -prune 2>/dev/null | tr '\n' ' ')"
echo "gold_in_nested=$(for g in $(find /app -mindepth 2 -name .git -prune 2>/dev/null); do git -C "$(dirname "$g")" cat-file -t GOLDSHA 2>/dev/null; done | grep -c commit)"
echo "author=$(git log -1 --format='%an <%ae> %ad' 2>/dev/null)"
echo "gold_type=$(git cat-file -t GOLDSHA 2>/dev/null)"
echo "gold_in_log=$(git log --all --oneline 2>/dev/null | grep -c GOLDSHORT)"
"""


def load_env_config() -> tuple[str, str, list[str], str]:
    """(sanitizer script, its sha256[:16], container run_args, cwd) from the merged run config.

    The container is started on the default network: only the strip is checked here.
    """
    env = variants.merged_config(DATASET).get("environment", {})
    script = env.get("setup_command", "")
    if not script.strip():
        sys.exit(f"no environment.setup_command in {[p.name for p in variants.config_paths(DATASET)]}")
    return (script, hashlib.sha256(script.encode()).hexdigest()[:16],
            list(env.get("run_args", ["--rm"])), env.get("cwd", "/app"))


def load_instances(subset: Path | None, limit: int | None, filt: str) -> list[dict]:
    if not INSTANCES.exists():
        sys.exit(f"missing {INSTANCES}; run prepare_dataset.py first")
    rows = [json.loads(l) for l in INSTANCES.open()]
    if subset:
        wanted = {x.strip() for x in subset.read_text().splitlines() if x.strip()}
        rows = [r for r in rows if r["instance_id"] in wanted]
    if filt:
        rows = [r for r in rows if re.search(filt, r["instance_id"])]
    rows.sort(key=lambda r: r["instance_id"])
    if limit:
        rows = rows[:limit]
    if not rows:
        sys.exit("no instances selected")
    return rows


def parse_probe(out: str) -> dict:
    state: dict = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            state[key.strip()] = value.strip()
    for key in ("commits", "tracked", "dirty", "diff_head_lines", "refs", "remotes", "tags",
                "stash", "objects", "reachable", "nested", "gold_in_log", "untracked",
                "gold_in_nested"):
        try:
            state[key] = int(state.get(key, ""))
        except ValueError:
            state[key] = None
    return state


def docker(*args: str, timeout: int = 120, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], input=stdin, capture_output=True, text=True,
                          timeout=timeout)


def check(instance: dict, script: str, run_args: list[str], cwd: str,
          pull_timeout: int, setup_timeout: int) -> dict:
    iid = instance["instance_id"]
    gold = (GOLD_SHA.findall(iid) or [""])[0]
    probe = PROBE.replace("GOLDSHA", gold or "HEAD").replace("GOLDSHORT", (gold or "HEAD")[:12])
    row: dict = {"instance_id": iid, "gold_sha": gold, "base_commit": instance.get("base_commit"),
                 "repo": instance.get("repo"), "ok": False, "failures": [], "notes": []}
    started = time.time()
    container = None
    try:
        # As mini's DockerEnvironment: detached, cwd set, the config's run_args, kept alive by sleep.
        created = docker("run", "-d", "-w", cwd, *run_args, instance["image_name"],
                         "sleep", "1h", timeout=pull_timeout)
        if created.returncode != 0:
            row["failures"].append(f"docker run failed: {created.stderr.strip()[:300]}")
            return row
        container = created.stdout.strip()

        before = parse_probe(docker("exec", "-i", container, "bash", "-s",
                                    stdin=probe, timeout=300).stdout)
        row["before"] = before

        t0 = time.time()
        result = docker("exec", "-i", container, "bash", "-s", stdin=script,
                        timeout=setup_timeout)
        row["strip_seconds"] = round(time.time() - t0, 1)
        row["strip_returncode"] = result.returncode
        combined = (result.stdout + result.stderr).strip()
        # The script's verdict lines, kept separately from the (possibly long) raw output.
        row["strip_message"] = " | ".join(l.strip() for l in combined.splitlines()
                                         if l.startswith(("STRIP-OK", "STRIP-FAIL", "STRIP-NOTE")))
        row["strip_output"] = combined[-2000:]

        after = parse_probe(docker("exec", "-i", container, "bash", "-s",
                                   stdin=probe, timeout=300).stdout)
        row["after"] = after
        row["fsck"] = docker("exec", container, "bash", "-lc",
                             "cd /app && git fsck --lost-found --no-progress 2>&1 | head -5",
                             timeout=300).stdout.strip()

        fail = row["failures"].append
        note = row["notes"].append

        # --- did the sanitizer itself accept the instance? ---
        if result.returncode != 0:
            fail(f"sanitizer exited {result.returncode}: "
                 f"{row['strip_message'] or row['strip_output'][-300:]}")
            return row
        # STRIP-OK with no STRIP-FAIL is a pass; STRIP-NOTE lines and git's stderr may precede it.
        if "STRIP-OK" not in row["strip_message"] or "STRIP-FAIL" in row["strip_message"]:
            fail(f"sanitizer did not report STRIP-OK (said: {row['strip_message'][:200]!r})")

        # --- the task must be unchanged ---
        if before.get("tree") != after.get("tree"):
            fail(f"HEAD tree changed {before.get('tree')} -> {after.get('tree')}: the agent would "
                 f"not see the same files as on the original dataset")
        if before.get("tracked") != after.get("tracked"):
            fail(f"tracked files {before.get('tracked')} -> {after.get('tracked')}")
        if after.get("toplevel") != "/app":
            fail(f"work-tree root is {after.get('toplevel')!r}, not /app")
        # Not "must be clean": pre-existing changes are part of the task, so both observable
        # states must be identical to the original container's.
        if before.get("status_sha") != after.get("status_sha"):
            fail(f"`git status` changed (before {before.get('dirty')} entries, after "
                 f"{after.get('dirty')}): the agent would start from a different state")
        if before.get("diff_body_sha") != after.get("diff_body_sha"):
            fail(f"`git diff HEAD` content changed ({before.get('diff_head_lines')} -> "
                 f"{after.get('diff_head_lines')} lines): the submitted patch would differ")
        elif before.get("diff_sha") != after.get("diff_sha"):
            # Only the `index <abbrev>..<abbrev>` header differs; the content is identical.
            note("diff index-header abbreviation differs (content identical)")
        if before.get("untracked") != after.get("untracked"):
            fail(f"untracked files {before.get('untracked')} -> {after.get('untracked')}")
        if before.get("dirty"):
            note(f"the original container is dirty ({before.get('dirty')} status entries, "
                 f"{before.get('untracked')} untracked) — preserved as-is")
        if before.get("head") != instance.get("base_commit"):
            note(f"original HEAD {before.get('head')} != dataset base_commit "
                 f"{instance.get('base_commit')}")

        # --- the leak must be gone ---
        if before.get("gold_type") != "commit":
            note("gold commit was NOT reachable in the original container either "
                 "(nothing to leak for this instance)")
        if after.get("gold_type"):
            fail(f"gold commit {gold[:12]} is still an object after the strip "
                 f"(type={after.get('gold_type')})")
        if after.get("gold_in_log"):
            fail("gold sha still appears in `git log --all`")
        for key, want, label in (("commits", 1, "commit"), ("refs", 1, "ref"),
                                 ("remotes", 0, "remote"), ("tags", 0, "tag"),
                                 ("stash", 0, "stash entry")):
            if after.get(key) != want:
                fail(f"expected {want} {label}(s) after the strip, found {after.get(key)}")
        # Nested repositories of other projects may stay; the gold sha must not be in any of them.
        if after.get("gold_in_nested"):
            fail(f"gold commit {gold[:12]} is reachable inside a nested repo "
                 f"({after.get('nested_dirs')})")
        if after.get("nested"):
            note(f"nested repo(s) left in place, gold sha absent from them: "
                 f"{after.get('nested_dirs')}")
        if after.get("objects") != after.get("reachable"):
            fail(f"{(after.get('objects') or 0) - (after.get('reachable') or 0)} unreachable "
                 f"object(s) left in .git — recoverable history")
        if row["fsck"]:
            fail(f"git fsck reports leftovers: {row['fsck'][:200]}")

        row["ok"] = not row["failures"]
        return row
    except subprocess.TimeoutExpired as e:
        row["failures"].append(f"timeout: {e}")
        return row
    finally:
        row["seconds"] = round(time.time() - started, 1)
        if container:
            subprocess.Popen(["docker", "rm", "-f", container],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=Path, default=PILOT,
                    help="file of instance ids, one per line (default: the 44-instance pilot)")
    ap.add_argument("--limit", type=int, default=None, help="cap instances")
    ap.add_argument("--filter", default="", help="regex on instance_id")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel containers (each holds a multi-GB image's layers open)")
    ap.add_argument("--pull-timeout", type=int, default=1800,
                    help="seconds for `docker run` including a first-time image pull")
    ap.add_argument("--setup-timeout", type=int, default=900,
                    help="seconds for the sanitizer itself (as configs/strip_history.yaml)")
    ap.add_argument("--output", type=Path, default=None,
                    help="where to write the JSON (default "
                         "<output>/swebench_pro_corrected/sanitization_check.json)")
    args = ap.parse_args()

    script, script_sha, run_args, cwd = load_env_config()
    instances = load_instances(args.subset, args.limit, args.filter)
    print(f"verifying {len(instances)} instance(s) with sanitizer {script_sha} "
          f"from {'+'.join(p.name for p in variants.config_paths(DATASET))}; "
          f"container flags {' '.join(run_args)}; cwd {cwd}")

    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(check, inst, script, run_args, cwd,
                             args.pull_timeout, args.setup_timeout): inst["instance_id"]
                   for inst in instances}
        for future in concurrent.futures.as_completed(futures):
            iid = futures[future]
            try:
                row = future.result()
            except Exception as e:      # report the instance, keep checking the others
                row = {"instance_id": iid, "ok": False, "failures": [f"{type(e).__name__}: {e}"]}
            rows.append(row)
            mark = "ok  " if row["ok"] else "FAIL"
            print(f"  [{mark}] {iid[:70]:72s} {row.get('strip_seconds', '?')}s strip, "
                  f"{row.get('seconds', '?')}s total"
                  + ("" if row["ok"] else "\n         " + "\n         ".join(row["failures"])))

    rows.sort(key=lambda r: r["instance_id"])
    n_ok = sum(1 for r in rows if r["ok"])
    payload = {
        "dataset": DATASET.key,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "sanitizer_sha256_16": script_sha,
        "n_instances": len(rows),
        "n_ok": n_ok,
        "container_run_args": run_args,
        "rows": rows,
    }
    out = args.output or (paths.output_dir(paths.BENCHMARK_DATA_SUBDIR[DATASET.benchmark])
                          / "sanitization_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    leaked_before = sum(1 for r in rows if (r.get("before") or {}).get("gold_type") == "commit")
    strip_times = [r["strip_seconds"] for r in rows if r.get("strip_seconds")]
    print(f"\n{n_ok}/{len(rows)} instances sanitized correctly")
    print(f"  gold commit was reachable before the strip in {leaked_before}/{len(rows)}")
    if strip_times:
        print(f"  sanitizer wall time: min {min(strip_times)}s / median "
              f"{sorted(strip_times)[len(strip_times) // 2]}s / max {max(strip_times)}s")
    print(f"  wrote {out}")
    if n_ok != len(rows):
        print("\nFailing instances abort with a RuntimeError at run time instead of running "
              "unsanitized.")
        sys.exit(1)


if __name__ == "__main__":
    main()
