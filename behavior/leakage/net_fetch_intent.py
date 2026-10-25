"""Classify what network-reaching commands in SWE-Bench Pro trajectories were aimed at.

    python behavior/leakage/net_fetch_intent.py                    # corrected dataset
    python behavior/leakage/net_fetch_intent.py --features output/leakage/pro_features.jsonl
    python behavior/leakage/net_fetch_intent.py --model gpt-5.6-sol --quotes 12

A network fetch is either ordinary dependency resolution or retrieval of the answer. Every
command that can leave the container is assigned one target:

  own_repo_source   the repository under test over HTTP (raw.githubusercontent.com,
                    api.github.com, github.com, a jsDelivr /gh/ mirror)
  own_repo_registry the repository under test through a package registry (e.g.
                    `proxy.golang.org/github.com/<org>/<repo>/@v/<version>.zip` serves the
                    full source of a public Go module)
  own_repo_git      the repository under test over git (fetch / clone / ls-remote / remote add)
  code_search       a code-search engine (grep.app, ...)
  dependency        a package registry or package-manager invocation
  third_party_src   source of a different project hosted on a code host
  other             reaches the network but matches none of the above

The first four are answer-seeking (ANSWER_SEEKING). Two further signals are recorded:
`post_fix_state` (the request names master/main/HEAD, a commit listing or a compare view,
which on these instances all denote the state after the fix) and `hits_own_patch` (the
requested path is a file the agent's own final patch modifies). With `--quotes` the
agents' `THOUGHT:` text for answer-seeking fetches is printed.

Reads the feature table written by extract_traj_features.py and the trajectories it points
to; writes every event to <OUTPUT_ROOT>/leakage/net_fetch_intent.json (see --json).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root, for `common`
from common import paths  # noqa: E402

EFF_ORDER = ["none", "low", "medium", "high", "xhigh"]

# A command that can leave the container at all; deliberately wide, so that every outbound
# command is classified.
NET = re.compile(r"\b(curl|wget)\b|urllib|requests\.(get|post)|http\.client"
                 r"|git\s+(fetch|clone|ls-remote|pull)\b|git\s+remote\s+add"
                 r"|pip\s+(install|download)|npm\s+(install|view|pack)"
                 r"|go\s+(get|mod\s+download)|ansible-galaxy\s+\w+\s+(install|download)", re.I)
URL = re.compile(r"https?://([^/\s'\"]+)([^\s'\"`)]*)")
CODE_HOST = re.compile(r"github\.com|githubusercontent\.com|gitlab\.com|bitbucket\.org"
                       r"|jsdelivr\.net/gh/|codeload\.github", re.I)
REGISTRY = re.compile(r"proxy\.golang\.org|goproxy\.io|sum\.golang\.org|pypi\.org|pythonhosted"
                      r"|registry\.npmjs|yarnpkg|crates\.io|rubygems|maven|nuget"
                      r"|galaxy\.ansible|nodejs\.org|deb\.debian|ubuntu\.com|alpinelinux", re.I)
PKG_TOOL = re.compile(r"pip\s+(install|download)|npm\s+(install|view|pack)"
                      r"|go\s+(get|mod\s+download)|ansible-galaxy\s+\w+\s+(install|download)", re.I)
GIT_REMOTE = re.compile(r"git\s+(fetch|clone|ls-remote|pull)\b|git\s+remote\s+add", re.I)
CODE_SEARCH = re.compile(r"grep\.app|searchcode|sourcegraph", re.I)
# On these instances the base commit precedes the fix, so master/main/HEAD and any commit listing
# denote the post-fix state.
POST_FIX = re.compile(r"/(master|main|develop|HEAD)/|@(master|main|develop)\b|/commits|/compare/"
                      r"|refs/heads|:master|:main", re.I)
THOUGHT = re.compile(r"THOUGHT:\s*(.+?)(?:```|$)", re.S)


def org_repo(instance_id: str) -> tuple[str, str]:
    """`instance_<org>__<repo>-<40 hex sha>[-v<sha>]` -> (org, repo), lowercased.

    Split on the sha, not on a hyphen: org and repo names can contain hyphens
    (`element-hq__element-web-53a9…`).
    """
    core = re.sub(r"-[0-9a-f]{40}.*$", "", instance_id.replace("instance_", ""))
    org, _, repo = core.partition("__")
    return org.lower(), repo.lower()


def patch_files(submission: str) -> set[str]:
    """Paths touched by the agent's own final diff."""
    return {m for m in re.findall(r"^diff --git a/(\S+)", submission or "", re.M)}


def classify(cmd: str, org: str, repo: str, patched: set[str]) -> dict:
    urls = URL.findall(cmd)
    joined = " ".join(h + p for h, p in urls).lower()
    own_slug = f"{org}/{repo}"
    # `origin` counts as the repo under test: in the original containers it points there, and
    # in the corrected ones re-adding it reconstructs what was removed.
    own_git = bool(GIT_REMOTE.search(cmd)) and (own_slug in cmd.lower() or "origin" in cmd.lower())
    # `repo:<org>/<repo>` as a code-search qualifier, in any case, literal or URL-encoded
    # (repo%3A…%2F…).
    qualifier = re.search(rf"repo(?::|%3a)\s*{re.escape(org)}(?:/|%2f){re.escape(repo)}",
                          cmd, re.I) is not None
    own_named = own_slug in joined or qualifier
    # A file the agent's own patch modifies is own-repo content whatever host or project name
    # it is served under (e.g. a renamed predecessor repository).
    own_by_path = bool(urls) and any(f.lower() in joined for f in patched)
    registry_host = REGISTRY.search(joined) is not None
    if own_named and registry_host:
        kind = "own_repo_registry"
    elif own_named or (own_by_path and CODE_HOST.search(joined)):
        kind = "own_repo_source"
    elif own_git:
        kind = "own_repo_git"
    elif CODE_SEARCH.search(cmd):
        kind = "code_search"
    elif REGISTRY.search(joined) or PKG_TOOL.search(cmd):
        kind = "dependency"
    elif urls and CODE_HOST.search(joined):
        kind = "third_party_src"
    else:
        kind = "other"
    # Which of the agent's own patched files did it ask upstream about?
    asked_about = sorted(f for f in patched if f.lower() in joined)
    return {"kind": kind,
            "post_fix_state": bool(POST_FIX.search(cmd)),
            "hits_own_patch": asked_about,
            "hosts": sorted({h.lower() for h, _ in urls})}


def collect(features: Path) -> list[dict]:
    rows = [json.loads(line) for line in features.open()]
    events = []
    for r in rows:
        traj = paths.DATA_ROOT / r["traj_path"] if r.get("traj_path") else None
        if not traj or not traj.exists():
            continue
        data = json.loads(traj.read_text())
        org, repo = org_repo(r["instance_id"])
        patched = patch_files((data.get("info") or {}).get("submission", ""))
        msgs = data["messages"]
        for idx, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            for action in ((m.get("extra") or {}).get("actions") or []):
                cmd = action.get("command") or ""
                if not NET.search(cmd):
                    continue
                thought = THOUGHT.search(m.get("content") or "")
                observation = next((n.get("content") or "" for n in msgs[idx + 1:idx + 3]
                                    if n.get("role") == "user"), "")
                events.append({"instance_id": r["instance_id"], "effort": r["effort"],
                               "model": r["model"], "resolved": r.get("resolved"),
                               "command": " ".join(cmd.split())[:600],
                               "thought": " ".join(thought.group(1).split())[:400] if thought else "",
                               "blocked": bool(re.search(r"403 Filtered|Could not resolve proxy"
                                                         r"|CONNECT tunnel failed|Tunnel connection"
                                                         r" failed|returned error: 403", observation)),
                               **classify(cmd, org, repo, patched)})
    return events


ANSWER_SEEKING = {"own_repo_source", "own_repo_registry", "own_repo_git", "code_search"}
LEGITIMATE = {"dependency", "third_party_src", "other"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    out = paths.output_dir("leakage")
    ap.add_argument("--features", type=Path, default=out / "pro_corrected_features.jsonl",
                    help="feature table from extract_traj_features.py (default: the corrected "
                         "dataset's)")
    ap.add_argument("--model", default="", help="restrict to one model slug")
    ap.add_argument("--quotes", type=int, default=6, help="how many THOUGHT quotes to print")
    ap.add_argument("--json", type=Path, default=out / "net_fetch_intent.json",
                    help="where to write every event (default: <OUTPUT_ROOT>/leakage/"
                         "net_fetch_intent.json)")
    args = ap.parse_args()
    if not args.features.exists():
        raise SystemExit(f"missing {args.features}; run extract_traj_features.py first")

    events = collect(args.features)
    if args.model:
        events = [e for e in events if e["model"] == args.model]
    print(f"# {args.features.name}{' (' + args.model + ')' if args.model else ''}: "
          f"{len(events)} network-reaching commands in "
          f"{len({(e['instance_id'], e['effort']) for e in events})} runs\n")

    by_kind = Counter(e["kind"] for e in events)
    print(f"{'target':18}{'cmds':>6}{'runs':>6}{'post-fix state':>16}{'asked about a file it patched':>32}")
    for kind in ("own_repo_source", "own_repo_registry", "own_repo_git", "code_search",
                 "dependency", "third_party_src", "other"):
        sel = [e for e in events if e["kind"] == kind]
        if not sel:
            continue
        runs = len({(e["instance_id"], e["effort"]) for e in sel})
        print(f"{kind:18}{len(sel):>6}{runs:>6}{sum(e['post_fix_state'] for e in sel):>16}"
              f"{sum(1 for e in sel if e['hits_own_patch']):>32}")
    leak = [e for e in events if e["kind"] == "own_repo_registry"]
    if leak:
        runs = sorted({(e["instance_id"].replace("instance_", "")[:38], e["effort"]) for e in leak})
        print(f"\n{len(leak)} commands in {len(runs)} runs reached the repo under test "
              f"through a package registry (own_repo_registry):")
        for iid, eff in runs:
            print(f"     {iid}  [{eff}]")
    seek = [e for e in events if e["kind"] in ANSWER_SEEKING]
    legit = [e for e in events if e["kind"] in LEGITIMATE]
    print(f"\nANSWER-SEEKING {len(seek)}/{len(events)} commands "
          f"({len(seek)/max(1,len(events))*100:.0f}%), in "
          f"{len({(e['instance_id'], e['effort']) for e in seek})} runs")
    print(f"LEGITIMATE     {len(legit)}/{len(events)} commands "
          f"({len(legit)/max(1,len(events))*100:.0f}%), in "
          f"{len({(e['instance_id'], e['effort']) for e in legit})} runs")
    print(f"  of the answer-seeking ones, {sum(e['post_fix_state'] for e in seek)} name the post-fix "
          f"state and {sum(1 for e in seek if e['hits_own_patch'])} name a file the agent's own "
          f"patch modifies")
    print(f"  blocked by the egress proxy: {sum(e['blocked'] for e in events)}/{len(events)}")

    print(f"\n{'effort':8}{'runs w/ any fetch':>18}{'answer-seeking runs':>21}{'legitimate runs':>17}")
    for eff in EFF_ORDER:
        sel = [e for e in events if e["effort"] == eff]
        if not sel:
            continue
        allr = len({e["instance_id"] for e in sel})
        seekr = len({e["instance_id"] for e in sel if e["kind"] in ANSWER_SEEKING})
        legr = len({e["instance_id"] for e in sel if e["kind"] in LEGITIMATE})
        print(f"{eff:8}{allr:>18}{seekr:>21}{legr:>17}")

    if args.quotes:
        print(f"\n{'=' * 100}\nWhat the agents say they are doing (their own THOUGHT preambles)\n"
              f"{'=' * 100}")
        seen: set[str] = set()
        shown = 0
        for e in sorted(seek, key=lambda x: -len(x["thought"])):
            if not e["thought"] or e["instance_id"] in seen:
                continue
            seen.add(e["instance_id"])
            shown += 1
            if shown > args.quotes:
                break
            tag = "post-fix" if e["post_fix_state"] else "unversioned"
            patch = f" | asked about {', '.join(e['hits_own_patch'])}" if e["hits_own_patch"] else ""
            print(f"\n[{e['effort']}] {e['instance_id'].replace('instance_', '')[:44]} "
                  f"({e['kind']}, {tag}{patch})")
            print(f'  "{e["thought"][:340]}"')

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(events, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
