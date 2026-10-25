# Vendored Act-onomy skill

`trace-qualitative-analyst/` is copied verbatim from the Act-onomy repository:

- Source: https://github.com/gaojie058/Act-onomy
- Path: `2_tools/trace-qualitative-analyst/`
- Commit: `0527a10dbd7363d80dce56a8b558f34946d2ddfd` (2026-05-30)
- Licences: source code MIT; codebook content (`references/codebook.md`) CC BY 4.0
- Reference: Act-onomy v4.2, arXiv:2605.13625v1

It is vendored so that the skill text steering the annotator is pinned together with the
annotator model. The batch driver (`../run_batch.py`) installs it into the annotation image
(`../Dockerfile`) and supplies its own task prompt (`../PROMPT.md`), which skips the skill's
interactive checkpoints and HTML rendering and makes the codebook a closed vocabulary.
`references/codebook.md` is also the codebook that `../validate.py` checks annotations against.
