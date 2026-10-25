#!/usr/bin/env python3
"""Check that a SWE-Bench Pro result tree is complete and that its scores are real results.

For each (model, effort) directory:

  1. coverage: preds.json holds every instance of --subset (default: the pilot), every row has its
     trajectory, and no id is outside the dataset;
  2. consistency: cost_summary.json describes the same instances as preds.json;
  3. scoring coverage: every submitted instance has an entry in eval/<effort>/eval_results.json;
  4. evaluator gaps: the evaluator records False both when the tests fail and when its own
     machinery failed (image not pulled, missing run script, worker exception). A real evaluation
     leaves `eval/<effort>/<iid>/<effort>_output.json`, so a False without it is a gap, not a
     failed patch. Re-running `evaluate.py --run-eval` retries exactly those instances.

Exits 0 only when every directory passes every check.

Usage:
    python verify_run_integrity.py                                   # every model, original dataset
    python verify_run_integrity.py --dataset pro-corrected --model gpt-5.6-sol
    python verify_run_integrity.py --retry-list gaps.txt             # ids needing re-scoring
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for `common`
from common import paths

import variants

HERE = Path(__file__).resolve().parent
INSTANCES = paths.DATASETS_ROOT / "swebench_pro" / "mini_instances.jsonl"
PILOT = HERE / "subsets" / "pilot.txt"


def dataset_ids() -> set[str]:
    if not INSTANCES.exists():
        raise SystemExit(f"missing {INSTANCES}; run prepare_dataset.py first")
    return {json.loads(l)["instance_id"] for l in INSTANCES.open()}


def cells(model_root: Path) -> list[Path]:
    return sorted(d for d in model_root.iterdir()
                  if d.is_dir() and d.name != "eval" and (d / "preds.json").exists())


def check_cell(cell: Path, eval_root: Path, known: set[str],
               want: set[str] | None) -> tuple[dict, list[str]]:
    """One (model, effort) directory. Returns (row, ids whose score is a gap, not a result)."""
    effort = cell.name
    preds = json.loads((cell / "preds.json").read_text())
    ids = set(preds)

    no_traj = sorted(i for i in ids if not (cell / i / f"{i}.traj.json").exists())
    alien = sorted(ids - known)
    missing = sorted(want - ids) if want is not None else []

    cs = cell / "cost_summary.json"
    cs_n, cs_mismatch = None, []
    if cs.exists():
        per = {r["instance_id"] for r in json.loads(cs.read_text()).get("per_instance", [])}
        cs_n = len(per)
        cs_mismatch = sorted(ids ^ per)

    submitted = {i for i, r in preds.items() if (r.get("model_patch") or "").strip()}
    res_p = eval_root / effort / "eval_results.json"
    scored, unscored, gaps = None, [], []
    if res_p.exists():
        res = json.loads(res_p.read_text())
        scored = len(res)
        unscored = sorted(submitted - set(res))
        # A False with no cached output.json is machinery failure, not a failed patch.
        gaps = sorted(i for i, ok in res.items()
                      if not ok and not (eval_root / effort / i / f"{effort}_output.json").exists())
    return ({
        "cell": effort, "preds": len(ids), "no_traj": len(no_traj), "alien": len(alien),
        "missing": len(missing), "cost_n": cs_n, "cs_mismatch": len(cs_mismatch),
        "submitted": len(submitted), "scored": scored, "unscored": len(unscored),
        "gaps": len(gaps),
        "detail": {"no_traj": no_traj[:5], "alien": alien[:5], "missing": missing[:5],
                   "cs_mismatch": cs_mismatch[:5], "unscored": unscored[:5], "gaps": gaps[:5]},
    }, gaps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", dest="models_",
                    help="repeatable; default = every model directory in the tree")
    ap.add_argument("--subset", type=Path, default=PILOT,
                    help="file of instance ids every directory must contain (default: the pilot)")
    ap.add_argument("--retry-list", type=Path, default=None,
                    help="write the union of unverified-failure ids here, one per line")
    variants.add_argument(ap)
    args = ap.parse_args()

    dataset = variants.resolve(args.dataset)
    root = paths.data_root(dataset.benchmark)
    known = dataset_ids()
    want = None
    if args.subset:
        want = {x.strip() for x in args.subset.read_text().split() if x.strip()}
        stray = want - known
        if stray:
            raise SystemExit(f"--subset has {len(stray)} id(s) not in the dataset, e.g. "
                             f"{sorted(stray)[:2]}")

    names = args.models_ or sorted(d.name for d in root.iterdir() if d.is_dir())

    print(f"verifying {dataset.title} under {root}")
    if want is not None:
        print(f"required coverage: {len(want)} ids from {args.subset}")

    all_gaps: set[str] = set()
    problems = 0
    hdr = (f"{'model':16s} {'cell':8s} {'preds':>6s} {'traj?':>6s} {'alien':>6s} {'miss':>5s} "
           f"{'cost=':>6s} {'subm':>5s} {'scored':>7s} {'unsc':>5s} {'GAPS':>5s}")
    for name in names:
        mroot = root / name
        if not mroot.is_dir():
            print(f"{name}: no such directory under {root}")
            problems += 1
            continue
        print(f"\n{hdr}\n{'-' * len(hdr)}")
        for cell in cells(mroot):
            row, gaps = check_cell(cell, mroot / "eval", known, want)
            all_gaps |= set(gaps)
            bad = (row["no_traj"] or row["alien"] or row["missing"] or row["cs_mismatch"]
                   or row["unscored"] or row["gaps"])
            problems += bool(bad)
            print(f"{name:16s} {row['cell']:8s} {row['preds']:6d} {row['no_traj']:6d} "
                  f"{row['alien']:6d} {row['missing']:5d} "
                  f"{'ok' if row['cs_mismatch'] == 0 else str(row['cs_mismatch']):>6s} "
                  f"{row['submitted']:5d} "
                  f"{('-' if row['scored'] is None else row['scored']):>7} "
                  f"{row['unscored']:5d} {row['gaps']:5d}{'   <-- CHECK' if bad else ''}")
            if bad:
                for k, v in row["detail"].items():
                    if v:
                        print(f"    {k}: {v}{' …' if len(v) == 5 else ''}")

    print(f"\ncolumns: traj?=preds rows with no trajectory | alien=ids not in the dataset | "
          f"miss=required ids absent | cost==cost_summary vs preds | unsc=submitted but unscored")
    print("GAPS = scored False with no cached <effort>_output.json: evaluator machinery failed, "
          "NOT a failed patch. Re-run evaluate.py with the images present before reporting.")
    if all_gaps:
        print(f"\n{len(all_gaps)} distinct instance(s) need re-scoring")
        if args.retry_list:
            args.retry_list.write_text("\n".join(sorted(all_gaps)) + "\n")
            print(f"  -> {args.retry_list}")
    print(f"\n{'PASS: every directory complete and scored' if not problems else f'{problems} directory(ies) need attention'}")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
