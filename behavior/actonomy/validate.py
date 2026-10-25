"""Check annotated Act-onomy JSON against the codebook and its skeleton. No API calls.

    python behavior/actonomy/validate.py --benchmark tb

Nothing constrains the JSON the annotator writes, and the analysis counts
`turns[*].quotes[*].grp` directly, so an off-codebook tag would silently become a share of a
distribution. Errors (the file must be redone): missing keys, TODO fields, a phase outside
the seven, non-contiguous turns, a quote that is not an exact substring of its turn, a group
outside the ten, a non-conformant triple with no unambiguous remap, a turn count that
differs from the skeleton, edited `obs`/`thought`/`action` text, and a filled-in withheld
outcome. Warnings: zero or more than five quotes on a turn, overlapping quotes, a
remappable triple, a short summary. The codebook is a closed vocabulary here, so a new leaf
is an error.

Writes a per-file JSONL report; exits non-zero if any file has an error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402

REQUIRED_TOP = ("instance_id", "outcome", "issue_text", "summary", "turns")
REQUIRED_TURN = ("n", "phase", "phase_label", "headline", "obs", "thought", "action", "quotes")
VERBATIM_FIELDS = ("obs", "thought", "action")
MAX_QUOTES = 5
MIN_SUMMARY_SENTENCES = 4
_SENTENCE = re.compile(r"[.!?](?:\s|$)")
_TAG = re.compile(r"<[^>]+>")


def _sentences(text: str) -> int:
    return len(_SENTENCE.findall(_TAG.sub("", text or "").strip()))


def _spans(indexed: list[tuple[int, dict]], host: str) -> list[tuple[int, int, int]]:
    """(start, end, quote_index) of each quote's first occurrence in its host text.

    `quote_index` is the quote's position in its turn's own `quotes` list, not in the
    per-host sublist — an overlap warning has to name something the annotator can find.
    """
    out = []
    for i, q in indexed:
        text = q.get("quote") or ""
        at = host.find(text) if text else -1
        if at >= 0:
            out.append((at, at + len(text), i))
    return sorted(out)


def validate_file(
    path: Path,
    *,
    skeleton: Path | None,
    index,
) -> dict[str, Any]:
    """Validate one `.annotated.json` (read-only). Returns a record."""
    errors: list[str] = []
    warnings: list[str] = []
    new_leaves: Counter = Counter()
    groups: Counter = Counter()

    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return {"file": str(path), "errors": [f"unparseable JSON: {e}"], "warnings": [],
                "n_turns": 0, "n_quotes": 0, "groups": {}, "new_leaves": {}}

    for k in REQUIRED_TOP:
        if k not in doc:
            errors.append(f"missing top-level key '{k}'")
    if doc.get("summary", layout.TODO) == layout.TODO or not str(doc.get("summary", "")).strip():
        errors.append("summary not written")
    elif _sentences(doc["summary"]) < MIN_SUMMARY_SENTENCES:
        warnings.append(f"summary is {_sentences(doc['summary'])} sentences, "
                        f"guidance is {MIN_SUMMARY_SENTENCES}-8")
    if doc.get("annotator_model", layout.TODO) == layout.TODO:
        errors.append("annotator_model still TODO — the rater must be recorded")
    if doc.get("outcome") not in ("resolved", "unresolved", "withheld"):
        errors.append(f"outcome is {doc.get('outcome')!r}, "
                      f"expected resolved/unresolved/withheld")
    if not doc.get("findings"):
        warnings.append("no findings written")

    turns = doc.get("turns") or []
    quote_target = doc.get("quote_target", "thought")
    if quote_target not in ("thought", "action", "both"):
        errors.append(f"quote_target {quote_target!r} is not thought/action/both")

    # --- compare against the extractor's skeleton ---
    skel: dict[str, Any] = {}
    skel_turns: list[dict] = []
    if skeleton and skeleton.exists():
        try:
            skel = json.loads(skeleton.read_text())
            skel_turns = skel.get("turns") or []
        except json.JSONDecodeError:
            warnings.append(f"skeleton unreadable: {skeleton}")
        if skel_turns and len(skel_turns) != len(turns):
            errors.append(f"{len(turns)} turns annotated but the skeleton has "
                          f"{len(skel_turns)} — turns were dropped or added")
        # An outcome the annotator filled in would be indistinguishable from a given one.
        if skel.get("outcome") == "withheld" and doc.get("outcome") != "withheld":
            errors.append(f"blinding broken: skeleton withheld the outcome but the "
                          f"annotation says {doc.get('outcome')!r}")
        for field in ("subject_model", "effort", "submission"):
            if field not in skel and field in doc:
                warnings.append(f"annotator supplied '{field}' ({doc[field]!r}) though it was "
                                f"withheld — a guess, not evidence; do not join on it")
    elif skeleton:
        warnings.append("no skeleton found; verbatim and turn-count checks skipped")

    n_quotes = 0
    for pos, turn in enumerate(turns):
        tag = f"T{turn.get('n', pos + 1)}"
        for k in REQUIRED_TURN:
            if k not in turn:
                errors.append(f"{tag}: missing '{k}'")
        if turn.get("n") != pos + 1:
            errors.append(f"{tag}: turn numbering is not contiguous from 1 (position {pos + 1})")
        for k in ("phase", "phase_label", "headline"):
            if turn.get(k, layout.TODO) == layout.TODO:
                errors.append(f"{tag}: '{k}' still TODO")
        if turn.get("phase") not in layout.PHASES and turn.get("phase") != layout.TODO:
            errors.append(f"{tag}: phase {turn.get('phase')!r} not in {list(layout.PHASES)}")

        thought, action = turn.get("thought") or "", turn.get("action") or ""
        if pos < len(skel_turns):
            for field in VERBATIM_FIELDS:
                if (turn.get(field) or "") != (skel_turns[pos].get(field) or ""):
                    errors.append(f"{tag}: '{field}' was modified from the skeleton — quotes are "
                                  f"matched by substring, so edited text breaks grounding")

        quotes = turn.get("quotes") or []
        n_quotes += len(quotes)
        if not quotes:
            warnings.append(f"{tag}: no quotes (valid for a bare-command turn, "
                            f"suspicious otherwise)")
        elif len(quotes) > MAX_QUOTES:
            warnings.append(f"{tag}: {len(quotes)} quotes, guidance is 1-{MAX_QUOTES}")

        by_host: dict[str, list[tuple[int, dict]]] = {}
        for qi, q in enumerate(quotes):
            text = q.get("quote") or ""
            grp, sub, leaf = q.get("grp"), q.get("sub"), q.get("leaf")
            if not text:
                errors.append(f"{tag}: quote with empty text")
                continue
            target = q.get("target") or ("action" if quote_target == "both" else quote_target)
            host = action if target == "action" else thought
            if text not in host:
                where = "action" if target == "action" else "thought"
                errors.append(f"{tag}: quote not a substring of {where}: {text[:60]!r}")
            else:
                by_host.setdefault(target, []).append((qi, q))
            if grp:
                groups[grp] += 1
            if grp not in layout.GROUPS:
                errors.append(f"{tag}: group {grp!r} is not one of the 10 Act-onomy groups")
                continue
            reason = index.validate_triple(grp, sub or "", leaf or "")
            if reason is None:
                continue
            suggestion = index.suggest_remap(grp, sub or "", leaf or "")
            # `suggest_remap` searches only within `grp`; a leaf filed under the wrong group
            # is found by looking it up across the whole codebook.
            if suggestion is None and leaf:
                parents = sorted(index.leaf_parents.get(leaf, set()))
                if len(parents) == 1:
                    suggestion = (parents[0][0], parents[0][1], leaf)
                elif parents:
                    warnings.append(f"{tag}: leaf {leaf!r} is filed under '{grp} / {sub}' but "
                                    f"the codebook lists it under {parents} — ambiguous, "
                                    f"resolve by hand")
            if suggestion:
                warnings.append(f"{tag}: {reason}; remappable to {suggestion}")
            elif grp in index.hierarchy and sub in index.hierarchy[grp]:
                # A leaf that exists nowhere in the codebook (PROMPT.md forbids coining one).
                new_leaves[f"{grp} / {sub} / {leaf}"] += 1
                errors.append(f"{tag}: leaf {leaf!r} is not in the codebook under '{grp} / {sub}'")
            else:
                errors.append(f"{tag}: {reason}")

        for host_name, hqs in by_host.items():
            host = action if host_name == "action" else thought
            spans = _spans(hqs, host)
            for (s1, e1, i1), (s2, _, i2) in zip(spans, spans[1:]):
                if s2 < e1:
                    warnings.append(f"{tag}: quotes {i1} and {i2} overlap in {host_name} "
                                    f"(codebook rule: quotes must not overlap)")

    return {
        "file": layout.rel(path),
        # Anonymized annotations carry no `submission`; the directory is the cell key.
        "submission": doc.get("submission") or path.parent.name,
        "instance_id": doc.get("instance_id"),
        "annotator_model": doc.get("annotator_model"),
        "n_turns": len(turns),
        "n_quotes": n_quotes,
        "groups": dict(groups),
        "new_leaves": dict(new_leaves),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate annotated Act-onomy JSON against the codebook and the skeletons.")
    ap.add_argument("--benchmark", default="pro-corrected", choices=sorted(layout.BENCHMARKS))
    ap.add_argument("--annotation-root", type=Path,
                    help="default: <ANNOTATION_ROOT>/<benchmark>/annotations")
    ap.add_argument("--skeleton-root", type=Path,
                    help="default: <ANNOTATION_ROOT>/<benchmark>/skeletons")
    ap.add_argument("--no-skeleton-check", action="store_true",
                    help="skip turn-count and verbatim-text comparison")
    ap.add_argument("--submission", action="append", dest="submissions",
                    help="restrict to a <model>__<effort> directory; repeatable")
    ap.add_argument("--report", type=Path,
                    help="per-file JSONL report (default: "
                         "<OUTPUT_ROOT>/behavior/validation_<benchmark>.jsonl)")
    ap.add_argument("--requeue", type=Path,
                    help="write the list of files with errors, for the driver to redo")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    a = ap.parse_args()

    ann_root = a.annotation_root or layout.annotation_root(a.benchmark)
    skel_root = None if a.no_skeleton_check else (
        a.skeleton_root or layout.skeleton_root(a.benchmark))
    if not ann_root.exists():
        print(f"nothing to validate: {ann_root} does not exist", file=sys.stderr)
        return 0

    index = layout.codebook_index()
    files = sorted(ann_root.glob("*/*.annotated.json"))
    if a.submissions:
        files = [f for f in files if f.parent.name in set(a.submissions)]

    records, bad = [], []
    n_err = n_warn = n_quotes = n_turns = 0
    groups: Counter = Counter()
    new_leaves: Counter = Counter()
    raters: Counter = Counter()

    for f in files:
        skel = (skel_root / f.parent.name / f.name.replace(".annotated.json", ".skeleton.json")
                if skel_root else None)
        rec = validate_file(f, skeleton=skel, index=index)
        records.append(rec)
        n_err += len(rec["errors"])
        n_warn += len(rec["warnings"])
        n_turns += rec["n_turns"]
        n_quotes += rec["n_quotes"]
        groups.update(rec["groups"])
        new_leaves.update(rec["new_leaves"])
        if rec.get("annotator_model") and rec["annotator_model"] != layout.TODO:
            raters[rec["annotator_model"]] += 1
        if rec["errors"]:
            bad.append(rec["file"])
            if not a.quiet:
                print(f"\n[FAIL] {rec['file']}")
                for e in rec["errors"][:10]:
                    print(f"   error: {e}")
                if len(rec["errors"]) > 10:
                    print(f"   … {len(rec['errors']) - 10} more errors")
        elif rec["warnings"] and not a.quiet:
            print(f"[warn] {rec['file']}: {len(rec['warnings'])} warning(s)")

    report = a.report or (layout.paths.output_dir("behavior") / f"validation_{a.benchmark}.jsonl")
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{len(files)} file(s): {len(files) - len(bad)} clean, {len(bad)} with errors "
          f"({n_err} errors, {n_warn} warnings)")
    print(f"  {n_turns} turns, {n_quotes} quotes "
          f"({n_quotes / n_turns:.2f}/turn)" if n_turns else "  no turns")
    if raters:
        print(f"  annotator(s): {', '.join(f'{k}={v}' for k, v in raters.most_common())}"
              + ("   (more than one rater)" if len(raters) > 1 else ""))
    if groups:
        tot = sum(groups.values())
        print("  groups: " + ", ".join(f"{g}={100 * c / tot:.0f}%"
                                      for g, c in groups.most_common()))
    if new_leaves:
        print(f"  {len(new_leaves)} off-codebook leaf label(s):")
        for leaf, c in new_leaves.most_common(10):
            print(f"    {c:4d}x  {leaf}")
    print(f"  report -> {report}")
    if a.requeue:
        a.requeue.parent.mkdir(parents=True, exist_ok=True)
        a.requeue.write_text(json.dumps(bad, indent=2))
        print(f"  requeue ({len(bad)}) -> {a.requeue}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
