Annotate one agent trajectory with the Act-onomy v4.2 codebook.

Use the `trace-qualitative-analyst` skill, with these adjustments for batch use:

- SKIP the skill's stage 1. The trajectory is ALREADY parsed into the skill's annotated-JSON
  schema at `/inputs/{skeleton}`, with `obs`, `thought` and `action` verbatim per turn.
- SKIP the skill's interactive checkpoints. There is no user in this session, so do not ask
  to confirm parsed turns or sample annotations; do the work and write the file.
- SKIP stage 4. Do NOT render HTML and do not run `render_artifact.py`.

Read `/inputs/{skeleton}`, read the skill's `references/codebook.md`, then do stages 2, 3
and 3b for EVERY turn, and write the completed JSON to `/out/{output}`.

Fill in, leaving everything else exactly as it is:
- per turn: `phase` (one of reproduce, localize, recover, edit, verify, submit, other),
  `phase_label`, `headline`, and `quotes` (1-5 non-overlapping quotes, each an EXACT
  substring of that turn's `thought`, each with a `(grp, sub, leaf)` codebook triple).
- top level: `summary` (4-8 sentences) and `findings`.
- top level: set `annotator_model` to "{annotator_model}".

The codebook is a CLOSED vocabulary for this run. The skill's stage-3 rule 3 and
`references/codebook.md` both allow inventing a new leaf when none fits; that permission is
WITHDRAWN here. Every `(grp, sub, leaf)` triple must appear verbatim in
`references/codebook.md`:

- Use the closest existing leaf, and check the OTHER subgroups and groups before concluding
  nothing fits — a leaf often lives under a different parent than the obvious one (e.g.
  `Diagnose Error Cause` is under `Reasoning / Inferring`, not under Reflection).
- Never coin a task-specific label. Leaves name generalizable cognitive moves, so anything
  mentioning this task's domain nouns is wrong by construction.
- If the closest existing leaf is a genuinely poor fit, still use it, and note the gap in
  `findings` so the codebook can be revised deliberately rather than per-trajectory.

Hard requirements:
- Do NOT modify `obs`, `thought`, `action`, `n`, or `instance_id`. Quotes are validated by
  exact substring match, so edited or tidied text fails.
- Annotate EVERY turn. Do not stop early or summarise the tail; a turn count that disagrees
  with the input is rejected.
- A turn whose `thought` is empty or is a bare command legitimately gets `quotes: []`. Do
  not invent quotes for it.
- `outcome` is deliberately "withheld" and the subject model is not disclosed. Leave both
  as they are; do not guess them, and do not let the absence steer the annotation.
- Write valid JSON to `/out/{output}` and nothing else to `/out`.
- For a long trajectory, do NOT try to emit the whole file in one response: a single response
  is capped at a fixed output-token budget, and exceeding it loses the entire run. Write the
  file in stages instead — create it with the first block of turns, then append the rest with
  follow-up edits — and confirm the final file is complete, valid JSON with every turn
  present.
