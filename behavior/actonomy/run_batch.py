"""Annotate skeletons with Claude Code and the Act-onomy skill, one container per trajectory.

    docker build -t actonomy-annotator:pinned behavior/actonomy
    python behavior/actonomy/run_batch.py --benchmark tb --dry-run
    python behavior/actonomy/run_batch.py --benchmark tb --concurrency 3

Each trajectory runs in a fresh container of the pinned image (see Dockerfile) so no
context carries over between trajectories. The container sees only its own skeleton
(never `_manifest.json`) and the task prompt in PROMPT.md, and writes the annotated JSON to
`annotations/<model>__<effort>/`. The raw Claude Code stream and run metadata are kept under
`runs/<model>__<effort>/<instance>/`, and every attempt is appended to
`runs/_batch_log.jsonl`.

Authentication: `--auth subscription` (default) passes CLAUDE_CODE_OAUTH_TOKEN, created
with `claude setup-token` from your own Claude login; `--auth api` passes
ANTHROPIC_API_KEY. Only the selected credential is forwarded, since an API key would
otherwise take precedence over the subscription token. Either is read from the
environment or the repository .env.

Resumable: a trajectory with an existing annotation is skipped unless `--force`. Run
validate.py on the output before analysing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402

HERE = Path(__file__).resolve().parent
PROMPT_TEMPLATE = HERE / "PROMPT.md"
DEFAULT_IMAGE = "actonomy-annotator:pinned"
DEFAULT_MODEL = "claude-sonnet-5"


def cred_env(auth: str) -> list[str]:
    name = "CLAUDE_CODE_OAUTH_TOKEN" if auth == "subscription" else "ANTHROPIC_API_KEY"
    layout.paths.load_env()
    value = os.environ.get(name)
    if not value:
        hint = (" (run `claude setup-token` and export the token)" if auth == "subscription"
                else "")
        sys.exit(f"{name} is not set{hint}")
    return ["-e", f"{name}={value}"]


def load_selection(bench: str) -> set[str] | None:
    """Instances to annotate, from selection.json (all skeletons if it does not exist)."""
    p = layout.selection_path(bench)
    if not p.exists():
        return None
    return set(json.loads(p.read_text())["instances"])


def discover(bench: str, submissions: list[str] | None, instances: list[str] | None,
             limit: int | None, force: bool,
             selection: set[str] | None = None) -> list[tuple[str, Path, Path]]:
    """[(submission, skeleton_path, annotation_path)] still to do."""
    skel_root, ann_root = layout.skeleton_root(bench), layout.annotation_root(bench)
    work = []
    for skel in sorted(skel_root.glob("*/*.skeleton.json")):
        sub = skel.parent.name
        inst = skel.name[: -len(".skeleton.json")]
        if submissions and sub not in set(submissions):
            continue
        if instances and inst not in set(instances):
            continue
        if selection is not None and inst not in selection:
            continue
        out = ann_root / sub / f"{inst}.annotated.json"
        if out.exists() and not force:
            continue
        work.append((sub, skel, out))
        if limit and len(work) >= limit:
            break
    return work


def docker_base(args) -> list[str]:
    cmd = ["docker", "run", "--rm"]
    if args.docker_runtime:
        cmd += ["--runtime", args.docker_runtime]
    return cmd


def run_one(sub: str, skel: Path, out: Path, args, creds: list[str],
            image_id: str, cc_version: str, prompt_sha: str) -> dict:
    """Annotate one trajectory. Returns a record; never raises."""
    inst = skel.name[: -len(".skeleton.json")]
    run_dir = layout.runs_root(args.benchmark) / sub / inst
    started = time.time()

    prompt = (PROMPT_TEMPLATE.read_text()
              .replace("{skeleton}", skel.name)
              .replace("{output}", out.name)
              .replace("{annotator_model}", args.model))

    # Stage the single skeleton in its own directory so the container sees neither sibling
    # trajectories nor the de-blinding manifest.
    with tempfile.TemporaryDirectory(prefix="actonomy-in-") as tmp_in, \
         tempfile.TemporaryDirectory(prefix="actonomy-out-") as tmp_out:
        shutil.copy(skel, Path(tmp_in) / skel.name)
        os.chmod(tmp_out, 0o777)
        cmd = docker_base(args)
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        cmd += creds + [
            "-e", "DISABLE_AUTOUPDATER=1",
            # Claude Code's default 32000-token response cap truncates long annotations;
            # PROMPT.md also tells the annotator to write the file incrementally.
            "-e", f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={args.max_output_tokens}",
            "-v", f"{tmp_in}:/inputs:ro",
            "-v", f"{tmp_out}:/out",
            "-w", "/work",
            args.image,
            "claude", "-p", prompt,
            "--model", args.model,
            "--strict-mcp-config",
            "--permission-mode", args.permission_mode,
            "--max-turns", str(args.max_turns),
            "--output-format", "stream-json", "--verbose",
        ]
        if args.dry_run:
            print(f"[dry-run] {sub}/{inst}: {' '.join(cmd[:4])} … {args.image} claude -p "
                  f"<{len(prompt)} char prompt> --model {args.model}")
            return {"submission": sub, "instance": inst, "status": "dry-run"}

        run_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        (run_dir / "trajectory.jsonl").write_text(proc.stdout)
        if proc.stderr:
            (run_dir / "stderr.txt").write_text(proc.stderr)

        # Cost and usage are in the final `result` event of the stream (Claude Code's
        # API-equivalent cost, not a subscription bill).
        envelope: dict[str, Any] = {}
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "result":
                envelope = ev

        produced = Path(tmp_out) / out.name
        stray = [p.name for p in Path(tmp_out).iterdir() if p.name != out.name]
        status = "ok"
        if produced.exists():
            try:
                doc = json.loads(produced.read_text())
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
            except json.JSONDecodeError as e:
                status = f"invalid-json: {e}"
                (run_dir / "raw_output.json").write_text(produced.read_text())
        else:
            status = "no-output"

    rec = {
        "submission": sub, "instance": inst, "status": status,
        # Hash of the prompt TEMPLATE (not the rendered prompt, which embeds file names),
        # so runs under the same instructions can be identified.
        "prompt_template_sha256": prompt_sha,
        "max_output_tokens": args.max_output_tokens,
        "returncode": proc.returncode, "seconds": round(time.time() - started, 1),
        "annotator_model": args.model, "cc_version": cc_version, "image_id": image_id,
        "skeleton": layout.rel(skel), "annotation": layout.rel(out) if status == "ok" else None,
        "stray_out_files": stray,
        "notional_cost_usd": envelope.get("total_cost_usd"),
        "cc_num_turns": envelope.get("num_turns"),
        "cc_is_error": envelope.get("is_error"),
        "cc_result": (str(envelope.get("result"))[:300] if envelope.get("is_error") else None),
        "usage": envelope.get("usage"),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(rec, indent=2))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the pinned Act-onomy annotator over skeletons.")
    ap.add_argument("--benchmark", default="tb", choices=sorted(layout.BENCHMARKS))
    ap.add_argument("--submission", action="append", dest="submissions",
                    help="restrict to a <model>__<effort> cell; repeatable")
    ap.add_argument("--instance", action="append", dest="instances", help="repeatable")
    ap.add_argument("--limit", type=int, help="stop after N trajectories")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="annotator model; must match the image's ANTHROPIC_MODEL pin")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--auth", default="subscription", choices=["subscription", "api"])
    ap.add_argument("--concurrency", type=int, default=2,
                    help="parallel containers (default 2; subscription runs are rate limited)")
    ap.add_argument("--max-turns", type=int, default=120,
                    help="Claude Code turn cap; long trajectories need many tool calls")
    ap.add_argument("--max-output-tokens", type=int, default=64_000,
                    help="CLAUDE_CODE_MAX_OUTPUT_TOKENS inside the container")
    ap.add_argument("--timeout", type=int, default=3600, help="per-container seconds")
    ap.add_argument("--permission-mode", default="bypassPermissions")
    ap.add_argument("--docker-runtime", default=None,
                    help="docker --runtime (e.g. runc where the default runtime cannot run "
                         "CPU-only containers)")
    ap.add_argument("--force", action="store_true", help="re-annotate even if output exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sel = load_selection(args.benchmark)
    work = discover(args.benchmark, args.submissions, args.instances, args.limit, args.force,
                    selection=sel)
    if sel is not None:
        print(f"selection: {len(sel)} instance(s) per cell "
              f"({layout.rel(layout.selection_path(args.benchmark))})", flush=True)
    if not work:
        print("nothing to do (every requested skeleton already has an annotation)")
        return 0
    creds = [] if args.dry_run else cred_env(args.auth)

    image_id = cc_version = "unknown"
    if not args.dry_run:
        def probe(cmd):
            r = subprocess.run(cmd, capture_output=True, text=True)
            return (r.stdout or r.stderr).strip()
        image_id = probe(["docker", "inspect", "-f", "{{.Id}}", args.image])
        if not image_id.startswith("sha256:"):
            sys.exit(f"image {args.image!r} not found; build it first:\n"
                     f"  docker build -t {args.image} {HERE}")
        cc_version = probe(docker_base(args) + [args.image, "claude", "--version"])

    cells = sorted({s for s, _, _ in work})
    print(f"{len(work)} trajectory/ies over {len(cells)} cell(s): {', '.join(cells)}", flush=True)
    prompt_sha = hashlib.sha256(PROMPT_TEMPLATE.read_bytes()).hexdigest()[:16]
    print(f"annotator={args.model}  cc={cc_version}  auth={args.auth}  "
          f"concurrency={args.concurrency}  prompt={prompt_sha}", flush=True)

    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(run_one, sub, skel, out, args, creds, image_id, cc_version,
                               prompt_sha)
                   for sub, skel, out in work]
        for i, fut in enumerate(futures, 1):
            try:
                rec = fut.result()
            except subprocess.TimeoutExpired:
                rec = {"submission": "?", "instance": "?", "status": "timeout"}
            except Exception as e:                       # keep the batch alive
                rec = {"submission": "?", "instance": "?", "status": f"error: {e}"}
            records.append(rec)
            print(f"[{i}/{len(work)}] {rec.get('submission')}/{rec.get('instance')}: "
                  f"{rec['status']} ({rec.get('seconds', '?')}s)", flush=True)

    ok = sum(1 for r in records if r["status"] in ("ok", "dry-run"))
    spend = sum(r.get("notional_cost_usd") or 0 for r in records)
    log = layout.runs_root(args.benchmark) / "_batch_log.jsonl"
    if not args.dry_run:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    print(f"\n{ok}/{len(records)} annotated -> {layout.annotation_root(args.benchmark)}")
    if spend:
        print(f"  notional cost ${spend:.2f} (API-equivalent) — "
              f"${spend/max(1,ok):.2f}/annotation")
    for r in records:
        if r["status"] not in ("ok", "dry-run"):
            print(f"  FAILED {r.get('submission')}/{r.get('instance')}: {r['status']}")
    if not args.dry_run:
        print(f"log appended -> {log}")
    print(f"\nNext: python {HERE / 'validate.py'} --benchmark {args.benchmark}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
