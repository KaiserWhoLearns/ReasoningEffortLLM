#!/usr/bin/env python3
"""
parse_trajectory.py
===================

Stage-1 preprocessor for SWE-agent trajectory analysis. Takes a raw trajectory
JSON file (the message-list format), extracts (observation, thought, action)
triples per turn, and emits a *partial* annotated JSON skeleton that the user
fills in with phase labels, codebook annotations, a summary, and findings
before running render_artifact.py.

USAGE
-----

    python parse_trajectory.py <trajectory.json> [options]

Options:
    --output PATH        Where to write the skeleton (default: <stem>.annotated.json)
    --instance-id ID     Pre-fill instance_id (e.g. psf/requests-2317)
    --model NAME         Pre-fill model (default: SWE-agent w/ GPT-4 Turbo)
    --outcome STATE      Pre-fill outcome: resolved | unresolved

The skeleton uses the literal token "TODO" for fields the user must fill in.
The render script validates phase values against a fixed set, so any unfilled
phase will surface as a warning at render time.

INPUT FORMATS
-------------

Two trajectory layouts are supported, both as JSON arrays of message dicts:

1. Native trajectory.json (`role` ∈ {"system", "user", "ai"}):
   - system message has a `system_prompt` field; `text` is null
   - user/ai messages have a `text` field
   - ai text shape: "DISCUSSION\\n<prose>\\n```\\n<command>\\n```"
   - user text shape: "<observation>\\n(Open file: <path>)\\n(Current directory: ...)\\nbash-$"

2. SWE-agent `.traj` format (`role` ∈ {"system", "user", "assistant"}):
   - same conceptual shape, just the role label differs

The first user message is treated specially: it contains the issue text plus
the SETTING/INSTRUCTIONS block. The script splits on the literal string
"INSTRUCTIONS:" to recover just the issue body.

WHAT THE SCRIPT DOES NOT DO
---------------------------

This script handles stage 1 (parse) only. It does NOT:
- assign phase labels (stage 2 — judgment call per turn)
- pick quote-grounded codebook tags (stage 3 — the actual cognitive work)
- write a summary or findings (still requires reading the trajectory)

Those remain Claude's job. The skeleton just removes the mechanical extraction
overhead and guarantees the verbatim thought text is preserved exactly (which
matters because the renderer matches quotes by `String.indexOf`).
"""

import argparse
import json
import re
import sys
from pathlib import Path


AI_ROLES = {"ai", "assistant"}
USER_ROLES = {"user"}
SYSTEM_ROLES = {"system"}

# Trailing prompt that the SWE-agent shell tacks onto every observation.
# Examples:
#   (Open file: /foo/bar.py)\n(Current directory: /foo)\nbash-$
#   (Open file: n/a)\n(Current directory: /foo)\nbash-$
PROMPT_SUFFIX_RE = re.compile(
    r"\n*\(Open file: [^)]*\)\s*\n\(Current directory: [^)]*\)\s*\nbash-\$\s*$",
    re.MULTILINE,
)


def split_thought_action(ai_text: str):
    """
    Split an AI message into (thought, action).

    AI messages produced by SWE-agent have the shape:

        DISCUSSION
        <prose paragraphs explaining what to do>
        ```
        <one shell command>
        ```

    The fenced code block at the end is the action; the prose above (with
    the leading 'DISCUSSION' header stripped) is the thought.

    Returns (thought, action) as strings. If the layout doesn't match (e.g.
    no fenced block), the whole text becomes the thought and action is "".
    """
    text = ai_text.strip()

    # Strip the leading "DISCUSSION" header line if present.
    if text.startswith("DISCUSSION"):
        # Remove "DISCUSSION" plus the newline after it.
        text = text[len("DISCUSSION"):].lstrip("\n")

    # Find the LAST fenced code block — that's the action. Earlier fenced
    # blocks (if any) are part of the thought (e.g. inline examples).
    # Pattern: ```\n<command>\n```  OR  ```<lang>\n<command>\n```
    action = ""
    thought = text

    # Search backwards for the closing fence.
    fence_close = text.rfind("```")
    if fence_close != -1:
        # Find the matching opening fence before it.
        fence_open = text.rfind("```", 0, fence_close)
        if fence_open != -1 and fence_open != fence_close:
            # Extract the command between the fences.
            block = text[fence_open + 3:fence_close]
            # The block may start with a language tag on its own line; drop it.
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
                # If the first "line" is a short token (no spaces) it's a lang tag.
                if first_line and " " not in first_line and len(first_line) <= 16:
                    block = rest
            action = block.strip()
            # The thought is everything before the opening fence.
            thought = text[:fence_open].rstrip()

    return thought, action


def strip_prompt_suffix(obs_text: str) -> str:
    """Strip the trailing `(Open file: ...) ... bash-$` block from an observation."""
    return PROMPT_SUFFIX_RE.sub("", obs_text).rstrip()


def extract_issue_text(first_user_text: str) -> str:
    """
    The first user message contains:
        We're currently solving the following issue within our repository. Here's the issue text:
        ISSUE:
        <issue body>

        INSTRUCTIONS:
        <31 lines of agent instructions>
        ...
        bash-$

    Pull out just the issue body — between 'ISSUE:' and 'INSTRUCTIONS:'.
    Falls back to returning the whole first user message (minus prompt suffix)
    if the markers aren't found.
    """
    text = strip_prompt_suffix(first_user_text)
    issue_marker = "ISSUE:"
    instr_marker = "INSTRUCTIONS:"
    i = text.find(issue_marker)
    j = text.find(instr_marker)
    if i != -1 and j != -1 and j > i:
        return text[i + len(issue_marker):j].strip()
    return text.strip()


def parse_trajectory(path: Path):
    """
    Read a trajectory JSON file and return (issue_text, turns_list).

    Each entry in turns_list is a dict {"n", "obs", "thought", "action"}
    using the verbatim text from the trajectory (no paraphrasing — quote
    substring matching at render time depends on this).
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array of message dicts, got {type(data).__name__}. "
            f"This script handles the message-list trajectory format; for paper-text "
            f"trajectories you'll need to convert them to JSON first."
        )

    # Skip leading system message(s).
    i = 0
    while i < len(data) and data[i].get("role") in SYSTEM_ROLES:
        i += 1

    if i >= len(data) or data[i].get("role") not in USER_ROLES:
        raise ValueError(
            "Expected the first non-system message to be a user message containing "
            "the issue text. Got: " + str(data[i].get("role") if i < len(data) else "EOF")
        )

    # First user message: extract issue text, treat as initial observation.
    issue_text = extract_issue_text(data[i].get("text", ""))
    initial_obs = (
        "Initial state: issue text + INSTRUCTIONS delivered. "
        "No file open; agent is in the repo root."
    )
    pending_obs = initial_obs
    i += 1

    turns = []
    n = 1
    while i < len(data):
        msg = data[i]
        role = msg.get("role")

        if role in AI_ROLES:
            thought, action = split_thought_action(msg.get("text", ""))
            # Look ahead for the next user observation.
            obs_for_next = None
            if i + 1 < len(data) and data[i + 1].get("role") in USER_ROLES:
                obs_for_next = strip_prompt_suffix(data[i + 1].get("text", ""))

            turns.append({
                "n": n,
                "obs": pending_obs,
                "thought": thought,
                "action": action,
            })
            n += 1

            if obs_for_next is not None:
                pending_obs = obs_for_next
                i += 2
            else:
                # Final AI turn (e.g. submit) with no trailing user message.
                i += 1
        else:
            # Stray user message without a preceding ai message — unusual but skip.
            i += 1

    return issue_text, turns


def build_skeleton(issue_text: str, turns: list, *, instance_id: str, model: str, outcome: str):
    """Wrap the parsed turns into the partial annotated-JSON skeleton."""
    return {
        "instance_id": instance_id,
        "model": model,
        "outcome": outcome,
        "issue_text": issue_text,
        "summary": (
            "TODO: 4-8 sentence narrative summary of the run. Write this LAST, "
            "after annotating turns. Cover what the agent was trying to do, the "
            "shape of the run (T-ranges per phase), the key turn(s), and a quality "
            "judgment (real fix vs. symptom; verification real or circular)."
        ),
        "findings": [],
        "turns": [
            {
                "n": t["n"],
                "phase": "TODO",
                "phase_label": "TODO",
                "headline": "TODO",
                "obs": t["obs"],
                "thought": t["thought"],
                "action": t["action"],
                "quotes": [],
            }
            for t in turns
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parse a raw SWE-agent trajectory JSON into a partial annotated skeleton.",
    )
    parser.add_argument("input_json", type=Path, help="Path to the raw trajectory JSON")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for the skeleton (default: <stem>.annotated.json)")
    parser.add_argument("--instance-id", default="TODO",
                        help="Pre-fill instance_id (e.g. psf/requests-2317)")
    parser.add_argument("--model", default="SWE-agent w/ GPT-4 Turbo",
                        help="Pre-fill model name")
    parser.add_argument("--outcome", default="TODO",
                        choices=["resolved", "unresolved", "TODO"],
                        help="Pre-fill outcome")
    args = parser.parse_args()

    if not args.input_json.exists():
        print(f"ERROR: input file not found: {args.input_json}", file=sys.stderr)
        sys.exit(1)

    issue_text, turns = parse_trajectory(args.input_json)

    skeleton = build_skeleton(
        issue_text, turns,
        instance_id=args.instance_id,
        model=args.model,
        outcome=args.outcome,
    )

    output_path = args.output or args.input_json.with_suffix(".annotated.json")
    with open(output_path, "w") as f:
        json.dump(skeleton, f, indent=2, ensure_ascii=False)

    print(f"✓ Wrote skeleton: {output_path}")
    print(f"  · {len(turns)} turns extracted")
    print(f"  · issue_text: {len(issue_text)} chars")
    print(f"  · TODO fields to fill in: instance_id={'TODO' if args.instance_id == 'TODO' else 'set'}, "
          f"outcome={'TODO' if args.outcome == 'TODO' else 'set'}, "
          f"summary, findings, and per-turn (phase, phase_label, headline, quotes)")


if __name__ == "__main__":
    main()
