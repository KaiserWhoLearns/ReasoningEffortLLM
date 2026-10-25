#!/usr/bin/env python3
"""SWE-Bench Pro, stage 2 of 3: score the patches with the official evaluator.

Two steps:

  convert     (always, no Docker): turn each effort's preds.json into the evaluator's patch list
              (`eval/<effort>_patches.json`, [{instance_id, patch, prefix}]) and write
              `eval/raw_sample.jsonl` (the test lists and image tags) from mini_instances.jsonl.

  --run-eval  (needs Docker and the `docker` Python SDK): run the official
              scaleapi/SWE-bench_Pro-os evaluator with --use_local_docker, writing
              `eval/<effort>/eval_results.json` = {instance_id: resolved}.

The evaluator is cloned into `vendor/SWE-bench_Pro-os` on first use, pinned to EVAL_REPO_COMMIT.
It applies each patch in a fresh container from the instance's original image, runs the selected
tests, and marks an instance resolved iff fail_to_pass and pass_to_pass all pass. Scoring is the
same for both datasets: the original image still has the git history the evaluator needs.

Usage:
    python evaluate.py --model gpt-5.6-sol                                  # convert only
    python evaluate.py --model gpt-5.6-sol --run-eval --workers 16
    python evaluate.py --model gpt-5.6-sol --dataset pro-corrected --run-eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for `common`
from common import models, paths

import variants

INSTANCES = paths.DATASETS_ROOT / "swebench_pro" / "mini_instances.jsonl"
EVAL_REPO = paths.VENDOR_DIR / "SWE-bench_Pro-os"
EVAL_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
EVAL_REPO_COMMIT = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
DOCKERHUB_USER = "jefzda"
RAW_COLS = ["instance_id", "fail_to_pass", "pass_to_pass", "repo", "base_commit",
            "before_repo_set_cmd", "selected_test_files_to_run", "dockerhub_tag"]


def effort_dirs(model_root: Path) -> list[Path]:
    if not model_root.exists():
        sys.exit(f"no runs at {model_root}; run run_sweep.py first")
    return sorted(d for d in model_root.iterdir() if d.is_dir() and (d / "preds.json").exists())


def build_raw_sample(eval_dir: Path) -> Path:
    """The evaluator's --raw_sample_path: one row per instance with its test lists."""
    if not INSTANCES.exists():
        sys.exit(f"missing {INSTANCES}; run prepare_dataset.py first")
    out = eval_dir / "raw_sample.jsonl"
    with INSTANCES.open() as src, out.open("w") as f:
        for line in src:
            r = json.loads(line)
            f.write(json.dumps({k: r.get(k) for k in RAW_COLS}) + "\n")
    return out


def convert(model_root: Path) -> Path:
    dirs = effort_dirs(model_root)
    eval_dir = model_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    build_raw_sample(eval_dir)
    for d in dirs:
        preds = json.loads((d / "preds.json").read_text())
        patches = [{"instance_id": iid, "patch": rec.get("model_patch") or "", "prefix": d.name}
                   for iid, rec in preds.items()]
        (eval_dir / f"{d.name}_patches.json").write_text(json.dumps(patches, indent=2))
        n_empty = sum(1 for p in patches if not p["patch"].strip())
        print(f"  [{d.name}] {len(patches)} patches ({n_empty} empty) -> {d.name}_patches.json")
    print(f"converted -> {eval_dir}")
    return eval_dir


def ensure_eval_repo(eval_repo: Path) -> Path:
    """Fetch the evaluator at EVAL_REPO_COMMIT if it is not checked out yet."""
    if not eval_repo.exists():
        print(f"fetching {EVAL_REPO_URL}@{EVAL_REPO_COMMIT[:12]} -> {eval_repo}")
        eval_repo.mkdir(parents=True)
        for cmd in (["init", "-q"], ["remote", "add", "origin", EVAL_REPO_URL],
                    ["fetch", "-q", "--depth", "1", "origin", EVAL_REPO_COMMIT],
                    ["checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(["git", "-C", str(eval_repo), *cmd], check=True)
    _patch_eval_runtime(eval_repo)
    return eval_repo


def _patch_eval_runtime(eval_repo: Path) -> None:
    """Start the evaluator's containers with runtime=$SWEAP_DOCKER_RUNTIME (default runc).

    swe_bench_pro_eval.py creates its containers with the daemon's default runtime; on hosts where
    that is nvidia, CPU-only containers fail to start and every instance scores False. Idempotent.
    """
    fp = eval_repo / "swe_bench_pro_eval.py"
    src = fp.read_text()
    if "SWEAP_DOCKER_RUNTIME" in src:
        return
    anchor = '            "command": ["-c", "bash /workspace/entryscript.sh"],\n'
    if anchor not in src:
        print("  [warn] could not find the run_kwargs anchor to set the Docker runtime",
              file=sys.stderr)
        return
    inject = anchor + '            "runtime": os.environ.get("SWEAP_DOCKER_RUNTIME", "runc"),\n'
    fp.write_text(src.replace(anchor, inject, 1))
    print(f"  [patch] set Docker runtime=runc in {fp.name}")


def run_eval(model_root: Path, eval_repo: Path, workers: int, efforts: list[str]) -> None:
    try:
        import docker  # noqa: F401
    except ImportError:
        sys.exit(f"--run-eval needs the `docker` Python SDK: {sys.executable} -m pip install docker")
    eval_dir = model_root / "eval"
    raw = eval_dir / "raw_sample.jsonl"
    for pf in sorted(eval_dir.glob("*_patches.json")):
        effort = pf.stem.replace("_patches", "")
        if efforts and effort not in efforts:
            continue
        out = eval_dir / effort
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "swe_bench_pro_eval.py",
               f"--raw_sample_path={raw}", f"--patch_path={pf}",
               f"--output_dir={out}", "--scripts_dir=run_scripts",
               f"--num_workers={workers}", f"--dockerhub_username={DOCKERHUB_USER}",
               "--use_local_docker"]
        print(f"\n[eval {effort}] {' '.join(cmd)}  (cwd={eval_repo})")
        subprocess.run(cmd, check=False, cwd=str(eval_repo))
        res = out / "eval_results.json"
        if res.exists():
            d = json.loads(res.read_text())
            print(f"[eval {effort}] resolved {sum(1 for v in d.values() if v)}/{len(d)} -> {res}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model slug from common/models.py")
    variants.add_argument(ap)
    ap.add_argument("--run-eval", action="store_true", help="run the evaluator (needs Docker)")
    ap.add_argument("--eval-repo", type=Path, default=EVAL_REPO,
                    help=f"evaluator checkout (default {EVAL_REPO}, fetched if absent)")
    ap.add_argument("--workers", type=int, default=16, help="evaluator parallelism")
    ap.add_argument("--efforts", default="",
                    help="comma list of effort directories to score (default: all); "
                         "conversion always covers every effort")
    args = ap.parse_args()

    dataset = variants.resolve(args.dataset)
    model_root = paths.data_root(dataset.benchmark) / models.name_slug(args.model)
    print(f"scoring {dataset.title} runs under {model_root}")
    convert(model_root)
    if args.run_eval:
        repo = ensure_eval_repo(args.eval_repo)
        efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
        run_eval(model_root, repo, args.workers, efforts)
        print(f"\nnext: report.py --dataset {dataset.key}")


if __name__ == "__main__":
    main()
