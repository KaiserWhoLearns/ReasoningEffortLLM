---
name: trace-qualitative-analyst
description: Analyze SWE-agent trajectories with quote-level codebook annotations and produce an interactive HTML report with a pie chart of action types. Use this skill whenever the user wants to analyze, annotate, visualize, or break down an agent trajectory — especially SWE-agent or SWE-bench trajectories — with phrase-level grounding to an action codebook. Trigger this for requests like "analyze this trajectory", "annotate these turns", "make a codebook breakdown", "generate a trajectory report", or whenever the user provides raw `(observation, thought, action)` data and wants structured analysis. Also trigger when the user mentions terms like "trajectory analysis", "action space codebook", "thought-action grounding", "qualitative trajectory analysis", or asks to ground codebook tags to specific quotes from agent reasoning.
---

# SWE-agent Trajectory Analyzer

Produce an interactive HTML report from an agent trajectory. The report shows each turn as a `(observation, thought, action)` triple, with codebook tags **grounded to specific quoted phrases** in the thought text, plus an SVG horizontal bar chart of action-group distribution and a per-turn **codebook-group signature** (right panel, above the bar chart).

## When to use this skill

Use this skill when the user gives you any of the following:

- A raw SWE-agent trajectory (`.traj` JSON file, pasted text from the paper, or a copy-paste from logs)
- A description of agent turns asking for codebook analysis
- A request to "annotate", "tag", "ground", "break down", or "visualize" agent reasoning
- A request to build a qualitative-analysis artifact for one or more agent runs

The deliverable is always an interactive HTML file with a pie chart, written in English.

## High-level workflow

There are four stages plus a required summarization step. **Stages 1 and 4 are now fully deterministic scripts**; stage 2 is a quick judgment per turn; **stage 3 and the summary are the cognitive work** — that's where you, Claude, do the reasoning.

```
1. Parse trajectory     → run scripts/parse_trajectory.py on a raw trajectory JSON;
                          it emits a partial annotated skeleton with TODOs
2. Phase-label turns    → assign reproduce/localize/edit/recover/verify/submit
                          (fill in `phase`, `phase_label`, `headline` per turn)
3. ANNOTATE             → for each thought, pick 2-5 verbatim quotes and
                          map each to a codebook (group, subgroup, leaf) triple
3b. SUMMARIZE           → write a 4–8 sentence narrative summary of the run
                          (REQUIRED — the renderer rejects JSON without it)
4. Render               → run scripts/render_artifact.py on the annotated JSON
```

**Read `references/codebook.md` before stage 3.** It defines the 10 codebook groups (a three-level hierarchy: action → sub-action → leaf, aligned with arXiv:2605.13625v1), their subgroups and leaf actions, plus annotation guidelines. Stage 3 is impossible to do well without it.

---

## Stage 1: Parse trajectory

**For the message-list trajectory format (a JSON array of `{role, text, ...}` dicts), use the preprocessor — don't extract by hand.** Manual extraction is slow and a single off-by-one character break the substring matching downstream.

```bash
python scripts/parse_trajectory.py <trajectory.json> \
    --instance-id <owner/repo-N> \
    --outcome <resolved|unresolved> \
    --output <stem>.annotated.json
```

What the preprocessor does:
- Strips the leading system message
- Pulls the issue body out of the first user message (between `ISSUE:` and `INSTRUCTIONS:`)
- Walks each subsequent ai/user pair and emits one turn per ai message:
  - `thought`: ai message text with the `DISCUSSION` header and the trailing fenced code block stripped
  - `action`: the contents of that fenced code block (multi-line edits preserved verbatim)
  - `obs`: the previous user message with the trailing `(Open file: …) bash-$` prompt stripped
- Wraps the result in the full annotated-JSON schema with `TODO` placeholders for everything cognitive (`phase`, `phase_label`, `headline`, `quotes`, `summary`, `findings`)

Both role conventions are accepted: the native trajectory format uses `role: "ai"`; SWE-agent's `.traj` files use `role: "assistant"`. The preprocessor handles either.

**After running the preprocessor**, open the resulting `*.annotated.json` and walk the turns to fill in stages 2 and 3. The verbatim thought text is already exact — quote substrings will match at render time as long as you copy from the file.

### When the preprocessor doesn't apply

- **Paper-formatted trajectories** (e.g., arXiv Figure 33–36 transcripts pasted as text). The marker shape is:
  ```
  SWE-agent <instance>     ← assistant turn (thought + action follow)
  Observation <instance>   ← environment response (the observation for the NEXT turn)
  ```
  Convert these to the JSON message-list format first (see `assets/example_*.json` for the layout), then run the preprocessor. Or extract turns manually if the trajectory is short.
- **Raw paste / mixed format**: ask the user one clarifying question before proceeding (e.g., "Is this from the SWE-agent paper appendix, or from a `.traj` file you ran?").

### What you produce in stage 1

The preprocessor's output JSON has the full schema. Each turn entry looks like:
```json
{
  "n": 1,
  "phase": "TODO",
  "phase_label": "TODO",
  "headline": "TODO",
  "obs": "...verbatim observation...",
  "thought": "...verbatim thought, no DISCUSSION prefix, no code block...",
  "action": "...the command...",
  "quotes": []
}
```

Verbatim text is preserved — do not paraphrase the thought when filling in quotes. The renderer matches quoted substrings by `String.indexOf`.

---

## Stage 2: Phase-label turns

Assign one of these phases to each turn (used for the colored pill in the UI):

| Phase | When to use |
|---|---|
| `reproduce` | Creating reproduction script, running it to confirm bug |
| `localize` | Searching for files, navigating to relevant code, scrolling/goto |
| `recover` | Recovering from a failed command or lint error |
| `edit` | Applying the actual fix patch |
| `verify` | Re-running tests/repro after the fix |
| `submit` | Cleanup, `submit` command, exit |
| `other` | Anything that doesn't fit |

Also write a short `phase_label` (display string, e.g., "Reproduction", "File-level localize", "Pinpoint") and a one-line `headline` (~6-10 words capturing what this turn accomplishes).

---

## Stage 3: Quote-grounded annotation (the cognitive step)

**This is the part you must do manually for each turn.** The script cannot do it.

For each turn's thought, pick **2–5 short verbatim quotes** that each carry a distinct cognitive move, and map each quote to a `(grp, sub, leaf)` codebook triple from `references/codebook.md`.

### Hard rules

1. **Quotes must be exact substrings.** The renderer matches by `String.indexOf`. If the thought says "the bug" but you write "this bug", the highlight will silently fail. Copy-paste, don't paraphrase.
2. **Quotes must not overlap.** If two cognitive moves are in the same phrase, pick the dominant one or split.
3. **Use a leaf from the codebook when possible.** If genuinely none fits, write a new leaf in the same naming style ("verb-phrase capturing the cognitive move") and the user can accept or revise it.
4. **The `grp` field uses these exact values (10-category v4.2 codebook):** `Retrieval`, `Memory`, `Planning`, `Reasoning`, `Evaluate`, `Deciding`, `Grounding`, `Executing`, `Learning`, `Reflection`. The keys are also their display names — no rewriting is performed by the template. The triple has no super-class layer; the paper's 3 clusters (External Interaction / Cognition & Execution / Learning & Adaptation) are presentation only.

### Heuristics

- **Short procedural turns** ("Now let's add the comments") may have only **1 annotation**. Don't pad.
- **Phase-transition turns** (turns that wrap up one phase and pivot to the next) often have **4-5 annotations**. Don't compress.
- **Tag the cognitive move, not the surface verb.** "Let's open the file" is *Grounding › Invoke structured API*, not just "Executing".
- **Reflection requires a failure signal.** Words like "did not work", "issue persists", "errored" are the trigger. Without one, prefer Reasoning or Deciding.
- **Reflection vs. Learning.** Reflection steers within the current episode; Learning persistently changes the agent's prompt, code, weights, or stored knowledge. If the move evaporates at episode end, it's Reflection.
- **Reasoning › Generating vs. Reasoning › Analysing.** Generating produces *new* artifacts (code, requirements, candidate moves) *de novo*. Analysing transforms information that is already present. Drafting a fresh patch is Generating; tracing dataflow through an existing patch is Analysing.
- **Diagnose vs. Infer**: Diagnose names a *cause* ("the regex doesn't match punctuation"). Infer chains *consequences* ("if X then Y"). Same thought can have both.

### Output of stage 3

For each turn, attach a `quotes` list:
```json
"quotes": [
  {"quote": "first try to replicate the bug as described",
   "grp": "Planning",
   "sub": "Decompose task",
   "leaf": "Decompose into subgoals with success conditions"},
  ...
]
```

---

## Stage 4: Render

Once you have a fully annotated JSON, run the renderer:

```bash
python scripts/render_artifact.py <annotated.json> --output <output.html>
```

The script:
- Validates every quote is a substring of its turn's thought (warnings, not errors)
- Builds an SVG donut pie chart of group counts
- Fills the HTML template
- Writes a self-contained HTML file

If there are warnings, **read them and fix the JSON** — usually the quote has a typo or smart-quote difference. Don't ship a render with warnings.

### Input JSON schema (full reference)

```json
{
  "instance_id": "pylint-dev/pylint-5859",
  "model": "SWE-agent w/ GPT-4 Turbo",
  "outcome": "resolved",
  "issue_text": "...full issue body, plain text...",
  "summary": "REQUIRED. A 4–8 sentence narrative summary of the entire trajectory...",
  "quote_target": "thought",   // OPTIONAL. "thought" (default), "action", or "both".
                               // See "Quote target modes" below.
  "findings": [
    {"title": "Reasoning is layered", "body": "Turn 9 alone has 5 distinct phrases..."},
    ...
  ],
  "segments": [                // OPTIONAL. See "Segments (optional)" below.
    {"start": 1, "end": 5, "title": "First investigation cycle",
     "summary": "Localize → reproduce → draft a first patch; T5 fails to dispatch."},
    ...
  ],
  "turns": [
    {
      "n": 1,
      "phase": "reproduce",
      "phase_label": "Reproduction",
      "headline": "Create the empty test file.",
      "obs": "...",
      "thought": "...verbatim...",
      "action": "create test.py",
      "quotes": [
        {"quote": "...", "grp": "Planning", "sub": "...", "leaf": "..."},
        ...
      ],
      "insight": "Optional callout shown in yellow — paper cross-reference, key observation, etc."
    },
    ...
  ]
}
```

### Quote target modes

The renderer supports three top-level modes for where quotes are anchored:

| `quote_target` | When to use | Quote substrings come from | Highlighted block |
|---|---|---|---|
| `"thought"` (default) | SWE-agent reproduce/localize/edit loops where each turn has a clear `(thought, action)` split — thought is multi-paragraph reasoning, action is a single shell command. | `t.thought` | Thought block (italic Fraunces serif) |
| `"action"` | Multi-agent dialog trajectories (AutoGen, MAST, etc.) where the agent's *message itself is the action* — code blocks, stdout tokens, peer directives. The "thought" surrounding the action is mostly explanatory prose; the user wants codebook tags grounded in the executable surface. | `t.action` | Action block (monospace dark `<code>`) |
| `"both"` | Hybrid runs (HyperAgent, Planner-Executor loops) where each turn carries *both* a multi-paragraph Planner thought AND a multi-block sub-agent action surface, and the cognitive interest lives on both. Each quote in `turns[*].quotes[]` then needs an extra `"target": "thought"` or `"target": "action"` field naming its host (defaults to `"action"`). | `t.thought` for thought-targeted quotes, `t.action` for action-targeted quotes. | Both blocks render with their respective highlights and a separate annotation table under each. |

**In `"action"` mode**, the schema interpretation flips:

- `t.action` should hold the **verbatim action surface** the agent emitted — multi-line code blocks, stdout strings, the full delegation/peer message — not a one-line summary like `edit foo.py`.
- `t.thought` becomes **context only** (rendered without highlights). Use it for a brief framing note such as `"(executor reports stdout)"` or `"(assistant emits Python; prose explanation around code blocks not annotated)"`.
- Quotes must be substrings of `t.action`, not `t.thought`. The validator will warn if they aren't.
- Pure-thought turns (e.g., an agent emitting only a plan outline with no code, no peer directive) are valid with `quotes: []`. The renderer handles 0-quote turns gracefully.

**Annotation rule for action mode**: do *not* pull quotes from explanatory prose around the code. If an agent emits `# This computes the daily usage\nx = 5 * 0.2`, quote `x = 5 * 0.2`, not the comment-style preamble. The whole point of action mode is that the cognitive interest lies in the executable surface — that is exactly where bugs hide and where the multi-agent loop's failures are most diagnostic.

### Summary section (required)

The `summary` field is **required** for every render. It populates a "Trajectory summary" panel that sits between the issue statement and the turn-by-turn timeline — the first thing a reader sees that has been written *by you* about *this specific run*.

Aim for **4–8 sentences** of narrative prose. A good summary covers:

1. **What the agent was trying to do** — one-line restatement of the bug/task in the agent's words, not the issue's.
2. **The shape of the run** — how many phases the trajectory falls into and what each phase accomplished. Use turn ranges (`T1–T9`, `T10–T11`) so the reader can jump in.
3. **The key turn(s)** — which one or two turns carry the most weight (densest reasoning, phase pivot, the patch itself).
4. **A judgment about quality** — did the agent solve the *bug* or just the *symptom*? Was the verification real or circular? Were there wasted turns and what pattern caused them?

Inline HTML is allowed (`<strong>`, `<em>`, `<code>`) so you can highlight turn numbers and key terms. Keep the tone descriptive, not boosterish — say what happened, including what went sideways.

The summary is the lens through which the rest of the artifact is read — write it last, after annotation, when you actually understand the run.

### Findings section

The `findings` array (0–4 entries) populates a "Summary findings" footer. Each finding is a `{title, body}` pair. Good findings are observations that would only become visible **because** of the quote-level granularity — for example:
- Turns where one paragraph fires 4+ different codebook leaves
- A single short phrase that carries a phase transition
- The contrast between annotation-dense reasoning turns and bare procedural turns

If you have nothing meaningful to say, omit `findings` entirely. The script will leave a polite placeholder.

**Summary vs. Findings:** the **summary** is a holistic narrative of the whole run (what happened, in what order, with what quality). The **findings** are punchy 1–2 sentence observations about *patterns the quote-level granularity reveals*. Don't duplicate; they read as complementary layers.

### Trajectory signature (auto-generated)

The right panel automatically renders a **codebook signature** above the bar chart — no schema fields required. For each turn it lists, in firing order, the full `Group › Sub › Leaf` of that turn's quotes. The block is shown twice:

- A **flat copy-pasteable text block** (one line per turn, `T01 = [Group › Sub › Leaf, ...]` form), useful for diffing two runs side by side. Pre-element is `user-select: all` so a single click selects the whole block.
- A **clickable per-turn row list**, where each row shows colored chips — one chip per quote — with a small-caps group prefix, the subgroup as the dominant text, and the leaf trailing in muted italic after a `›` separator. Colored by the group palette. Hovering any chip reveals the full triple as a tooltip. Clicking a row scrolls the trajectory to that turn.

**Session-restart detection** is built in: if any turn's verbatim `thought` text equals an earlier turn's `thought`, a `↺` divider is inserted at that boundary in both views. This catches multi-agent runs (e.g. HyperAgent) that re-issue an initial planner thought after an internal reset, and is silent on normal single-session SWE-agent runs.

The signature is purely a JS-side projection of the existing `turns[*].quotes[*].{grp,sub,leaf}` data — no extra annotation work is required to populate it.

### Segments (optional)

The optional `segments[]` field lets you carve the run into named sub-sequences (e.g., one per session-restart cycle, or one per phase block) and attach a 1–3 sentence narrative to each. Each entry has the shape:

```json
{
  "start": 1,                                  // first turn n in the segment, inclusive
  "end":   5,                                  // last turn n in the segment, inclusive
  "title": "First investigation cycle",        // OPTIONAL short label
  "summary": "Localize → reproduce → draft a first patch (T4); T5 fails to dispatch."  // OPTIONAL prose
}
```

Segments are surfaced **only inside the right-panel Trajectory signature** (not in the main trajectory column), in two places:

1. **Inline in the flat copy-paste `<pre>`** as plain-text headers (HTML stripped, summary auto-wrapped to ~78 cols), so analyses copy-paste cleanly into chat / notebooks:
   ```
   === SEQ 01 · T01–T05 · First investigation cycle ===
   Localize → reproduce → draft a first patch (T4); T5 fails to dispatch.

   T01 = [Reasoning › Inferring › Initial diagnostic framing, ...]
   T02 = [...]
   ```
2. **As compact yellow cards** above the first clickable row of each segment (same `--highlight` background + `--accent` left border as the `insight` callouts; smaller font for the narrow panel).

The summary may contain inline HTML (`<strong>`, `<em>`, `<code>`) — it renders as-is in the cards and is stripped (with entity unescape) for the flat block. Segments must be sorted by `start` and non-overlapping; the renderer doesn't enforce coverage so partial segmentation is fine.

When to add segments: anytime the run has natural narrative chapters that are hard to spot from chips alone — session restarts, phase shifts, before/after a key edit, or stretches of stuck-loop behavior worth flagging. Skip for short trajectories (< ~10 turns) where the timeline is already self-evident.

### Resizable right panel

The right panel has a **drag-to-resize handle** on its left edge (default width 360px, min 280, capped at 70% of viewport). Drag to widen / narrow; double-click the handle to reset; the chosen width is persisted in `localStorage` so it sticks across reloads. Useful when the segment summaries or long leaf names spill across multiple lines and you want more breathing room.

---

## Working with the user

Before stage 3, **show the user the parsed turns** (stage 1 + 2 output) and ask if anything looks off. Trajectory parsing can be ambiguous, especially around phase labels for borderline turns.

After stage 3 but before stage 4, **show the user a sample of your annotations** (3–5 turns' worth) and ask if the granularity / tag choices feel right. Annotation style is partly a matter of taste; calibrating early saves rework.

After rendering, **always run `present_files`** so the user can open and review the artifact directly.

If the user asks for multiple trajectories (e.g., a successful one and a failed one for comparison), produce one HTML per trajectory. The codebook is shared across runs but each render is self-contained.

---

## Worked example

The skill ships with `assets/example_pylint_5859.json` — a complete annotation of the `pylint-dev/pylint-5859` trajectory from arXiv:2405.15793 Figure 34. Read it as a reference for granularity, quote-selection style, finding-writing style, and overall calibration.

To see the rendered output:

```bash
python scripts/render_artifact.py assets/example_pylint_5859.json --output /tmp/example.html
```

This is your gold-standard reference for what "good" looks like.

---

## File layout

```
trace-qualitative-analyst/
├── SKILL.md                              ← this file
├── references/
│   └── codebook.md                       ← the 10-group v4.2 action codebook (read in stage 3)
├── scripts/
│   ├── parse_trajectory.py               ← stage 1 preprocessor (raw JSON → skeleton)
│   └── render_artifact.py                ← stage 4 renderer (annotated JSON → HTML)
└── assets/
    ├── template.html                     ← HTML template with placeholders
    └── example_pylint_5859.json          ← worked example / reference
```
