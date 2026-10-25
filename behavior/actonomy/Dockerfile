# Pinned Claude Code image for Act-onomy annotation (used by run_batch.py).
#
# Contains Claude Code at a fixed version and only the vendored Act-onomy skill: no
# memory files, no inherited ~/.claude, no plugins, no MCP servers, autoupdater off.
#
#   docker build -t actonomy-annotator:pinned behavior/actonomy
FROM node:22-bookworm-slim

ARG CC_VERSION=2.1.193
ENV DISABLE_AUTOUPDATER=1

RUN npm install -g @anthropic-ai/claude-code@${CC_VERSION}

# `--model` (passed by run_batch.py) outranks ANTHROPIC_MODEL, so both name the same model.
# The haiku pin sets Claude Code's background model, which `--model` does not change.
ARG CC_MODEL=claude-sonnet-5
ARG CC_HAIKU_MODEL=claude-haiku-4-5-20251001
ENV ANTHROPIC_MODEL=${CC_MODEL} \
    ANTHROPIC_DEFAULT_HAIKU_MODEL=${CC_HAIKU_MODEL}
LABEL cc_model="${CC_MODEL}" cc_haiku_model="${CC_HAIKU_MODEL}" cc_version="${CC_VERSION}"

# Clean slate, then install only the vendored Act-onomy skill.
RUN rm -rf /root/.claude /root/.claude.json /home/node/.claude /home/node/.claude.json 2>/dev/null || true
COPY skill/trace-qualitative-analyst /home/node/.claude/skills/trace-qualitative-analyst
COPY skill/UPSTREAM.md /home/node/.claude/skills/UPSTREAM.md
RUN chown -R node:node /home/node/.claude

# run_batch.py runs the container as the host uid (so outputs are owned by the caller);
# that uid has no passwd entry, so HOME must be writable by anyone. Claude Code refuses
# bypassPermissions as uid 0, so the container never runs as root.
RUN chmod -R a+rwX /home/node

ENV HOME=/home/node
USER node
WORKDIR /work
ENTRYPOINT []
