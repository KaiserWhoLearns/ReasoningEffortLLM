"""Restore verbatim text into annotations whose copied-through text drifted.

    python behavior/actonomy/repair.py --benchmark tb --dry-run

The annotator rewrites the whole skeleton, and on long turns it occasionally mis-copies
`obs`/`thought`/`action` (e.g. a newline lost at a truncation marker), which validate.py
rejects. This restores those fields from the skeleton and keeps the annotator's judgments
(phases, headlines, quotes, summary, findings). A quote that no longer matches the restored
text is dropped. Each repair is recorded in the file under `_repair`. A file whose turn
count differs from the skeleton is left alone and must be re-annotated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402

FIELDS = ("obs", "thought", "action")


def repair_file(ann: Path, skel: Path, *, dry_run: bool) -> dict | None:
    """Restore verbatim fields. Returns a record, or None if nothing needed repair."""
    a = json.loads(ann.read_text())
    s = json.loads(skel.read_text())
    at, st = a.get("turns") or [], s.get("turns") or []
    if len(at) != len(st):
        return {"file": layout.rel(ann), "status": "SKIPPED: turn count differs "
                f"({len(at)} vs {len(st)}) — needs re-annotation, not repair"}

    restored = 0
    dropped: list[str] = []
    for ta, ts in zip(at, st):
        changed = [f for f in FIELDS if (ta.get(f) or "") != (ts.get(f) or "")]
        if not changed:
            continue
        for f in changed:
            ta[f] = ts.get(f) or ""
        restored += len(changed)
        keep = []
        for q in ta.get("quotes") or []:
            target = q.get("target") or (a.get("quote_target") or "thought")
            host = ta["action"] if target == "action" else ta["thought"]
            if (q.get("quote") or "") in host:
                keep.append(q)
            else:
                dropped.append(f"T{ta.get('n')}: {(q.get('quote') or '')[:60]}")
        ta["quotes"] = keep

    if not restored:
        return None
    a["_repair"] = {"fields_restored": restored, "quotes_dropped": len(dropped),
                    "dropped": dropped, "tool": "repair.py"}
    if not dry_run:
        ann.write_text(json.dumps(a, indent=2, ensure_ascii=False))
    return {"file": layout.rel(ann), "status": "repaired", "fields_restored": restored,
            "quotes_dropped": len(dropped), "dropped": dropped}


def main() -> int:
    ap = argparse.ArgumentParser(description="Restore verbatim text into drifted annotations.")
    ap.add_argument("--benchmark", default="tb", choices=sorted(layout.BENCHMARKS))
    ap.add_argument("--annotation-root", type=Path)
    ap.add_argument("--skeleton-root", type=Path)
    ap.add_argument("--submission", action="append", dest="submissions")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    a = ap.parse_args()

    ann_root = a.annotation_root or layout.annotation_root(a.benchmark)
    skel_root = a.skeleton_root or layout.skeleton_root(a.benchmark)
    files = sorted(ann_root.glob("*/*.annotated.json"))
    if a.submissions:
        files = [f for f in files if f.parent.name in set(a.submissions)]

    recs = []
    for f in files:
        skel = skel_root / f.parent.name / f.name.replace(".annotated.json", ".skeleton.json")
        if not skel.exists():
            continue
        r = repair_file(f, skel, dry_run=a.dry_run)
        if r:
            recs.append(r)
            print(f"{'[dry] ' if a.dry_run else ''}{r['status']}: {Path(r['file']).name}"
                  + (f" — {r.get('fields_restored')} field(s) restored, "
                     f"{r.get('quotes_dropped')} quote(s) dropped"
                     if r["status"] == "repaired" else ""))
            for d in r.get("dropped", []):
                print(f"      dropped quote {d}")

    ok = [r for r in recs if r["status"] == "repaired"]
    print(f"\n{len(files)} file(s) checked, {len(ok)} repaired, "
          f"{sum(r['quotes_dropped'] for r in ok)} quote(s) dropped"
          + (" (dry run, nothing written)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
