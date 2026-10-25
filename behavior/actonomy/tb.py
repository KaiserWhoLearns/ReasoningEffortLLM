"""Terminal-Bench 2.1 (harbor / Terminus-2) trial -> Act-onomy turns.

harbor's `agent/trajectory.json` keeps only the agent's `Analysis:`/`Plan:` prose: one
`user` step with the system prompt and task, then the `agent` steps, with no interleaved
observations or commands. The (observation, thought, action) triple is therefore rebuilt
from `agent/recording.cast` (asciinema): `i` events are the keystrokes the agent sent, `o`
events the terminal output it saw. The cast header carries an absolute epoch and each event
a relative offset, while steps carry ISO timestamps, so events are bucketed into step
intervals: agent step k gets the `i` events in [t_k, t_k+1) as its action and the `o`
output in [t_k-1, t_k) as its observation (the obs-precedes-thought convention of the Pro
reader). The alignment is approximate at interval boundaries and is recorded as
`aligned="cast-timestamps"`.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# Boundaries in the single `user` step: the Terminus-2 prompt is
# <format instructions> "Task Description:" <task> "Current terminal state:" <screen>.
_TASK_MARK = "Task Description:"
_TERM_MARK = "Current terminal state:"
_SCREEN_MARK = "Current Terminal Screen:"

# CSI / OSC / single-char escapes emitted by tmux redraws.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_CR_RUN = re.compile(r"(?:\r\n|\r|\n){3,}")


def strip_ansi(text: str) -> str:
    return _CR_RUN.sub("\n\n", _ANSI.sub("", text).replace("\r\n", "\n")).strip()


def _iso(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def load_cast(path: Path) -> list[tuple[float, str, str]]:
    """[(absolute_time, 'i'|'o', payload)] from an asciinema v2 recording."""
    if not path.exists():
        return []
    out: list[tuple[float, str, str]] = []
    with open(path, errors="replace") as f:
        try:
            base = float(json.loads(f.readline()).get("timestamp") or 0)
        except (ValueError, json.JSONDecodeError):
            return []
        for line in f:
            line = line.strip()
            if not line.startswith("["):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(ev) >= 3 and ev[1] in ("i", "o"):
                out.append((base + float(ev[0]), ev[1], str(ev[2])))
    return out


def split_prompt(message: str) -> tuple[str, str]:
    """(task_description, initial_terminal_screen) from the first user step."""
    task, screen = message, ""
    i, j = message.find(_TASK_MARK), message.find(_TERM_MARK)
    if 0 <= i < j:
        task = message[i + len(_TASK_MARK):j].strip()
    elif i >= 0:
        task = message[i + len(_TASK_MARK):].strip()
    k = message.find(_SCREEN_MARK)
    if k >= 0:
        screen = message[k + len(_SCREEN_MARK):].strip()
    return task, screen


def reward_of(trial_dir: Path) -> float | None:
    """Verifier reward: 1.0 == resolved. `reward.txt` first, `result.json` as fallback."""
    rp = trial_dir / "verifier" / "reward.txt"
    if rp.exists():
        try:
            return float(rp.read_text().strip())
        except ValueError:
            pass
    res = trial_dir / "result.json"
    if res.exists():
        try:
            vr = json.loads(res.read_text()).get("verifier_result") or {}
            r = (vr.get("rewards") or {}).get("reward")
            return float(r) if r is not None else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def parse_trial(
    trial_dir: Path,
    *,
    head_tail,                     # extract.head_tail, so truncation is identical
    obs_head: int = 600,
    obs_tail: int = 200,
    thought_cap: int = 20_000,
    issue_cap: int = 4_000,
) -> dict[str, Any]:
    """Read one harbor trial into turns + task text + extraction stats."""
    traj_p = trial_dir / "agent" / "trajectory.json"
    if not traj_p.exists():
        raise ValueError(f"{trial_dir}: no agent/trajectory.json")
    traj = json.loads(traj_p.read_text())
    steps = traj.get("steps") or []
    if not steps:
        raise ValueError(f"{trial_dir}: trajectory has no steps")

    task_text, screen = "", ""
    for st in steps:
        if st.get("source") == "user":
            task_text, screen = split_prompt(str(st.get("message") or ""))
            break

    agent_steps = [st for st in steps if st.get("source") == "agent"]
    if not agent_steps:
        raise ValueError(f"{trial_dir}: no agent steps")

    cast = load_cast(trial_dir / "agent" / "recording.cast")
    times: list[float] = []
    for st in agent_steps:
        try:
            times.append(_iso(str(st.get("timestamp"))))
        except (TypeError, ValueError):
            times.append(float("nan"))
    usable = cast and all(t == t for t in times)          # NaN check

    def window(lo: float, hi: float, kind: str) -> str:
        return "".join(p for (t, k, p) in cast if k == kind and lo <= t < hi)

    turns: list[dict[str, Any]] = []
    n_trunc = 0
    for i, st in enumerate(agent_steps):
        thought = str(st.get("message") or "").strip()
        thought, cut = head_tail(thought, thought_cap, 0)
        n_trunc += cut

        if usable:
            nxt = times[i + 1] if i + 1 < len(times) else float("inf")
            action = strip_ansi(window(times[i], nxt, "i"))
            prev = times[i - 1] if i else float("-inf")
            obs_full = screen if i == 0 else strip_ansi(window(prev, times[i], "o"))
        else:
            action, obs_full = "", (screen if i == 0 else "")
        obs, obs_cut = head_tail(obs_full, obs_head, obs_tail)

        turns.append({
            "n": i + 1,
            "phase": "TODO",
            "phase_label": "TODO",
            "headline": "TODO",
            "obs": obs,
            "thought": thought,
            "action": action,
            "quotes": [],
            "_x": {
                "rc": None,                              # TB has no returncode envelope
                "obs_chars": len(obs_full),
                "obs_truncated": obs_cut,
                "content_chars": len(thought),
                "reasoning_chars": 0,                    # harbor exposes no separate CoT
                "thought_truncated": cut,
                "n_actions": 1 if action else 0,
            },
        })

    reward = reward_of(trial_dir)
    return {
        # Used by extract.py to redact the model's name; never written into a skeleton.
        "harness_model_name": (traj.get("agent") or {}).get("model_name"),
        "issue_text": head_tail(task_text, issue_cap, 0)[0],
        "turns": turns,
        "stats": {
            "n_turns": len(turns),
            "submitted": reward is not None,
            "empty_thought_turns": sum(1 for t in turns if not t["thought"].strip()),
            "empty_action_turns": sum(1 for t in turns if not t["action"].strip()),
            "truncated_thought_turns": n_trunc,
            "reasoning_turns": 0,
            "failed_obs_turns": 0,                       # no returncode to key on
            "trajectory_format": f"harbor-terminus-2/{traj.get('schema_version')}",
            "aligned": "cast-timestamps" if usable else "steps-only",
            "reward": reward,
        },
    }


def cell_tasks(effort_dir: Path) -> dict[str, Path]:
    """{task_name: trial_dir} for one `<model>/<effort>`; the task name is the join key.

    Trial directories carry a random suffix (`sam-cell-seg__ccRCrXj`); if a task has
    several trials, the first in sorted order is used.
    """
    out: dict[str, Path] = {}
    for run_dir in sorted(p for p in effort_dir.iterdir() if p.is_dir()):
        for trial in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if (trial / "result.json").exists():
                out.setdefault(trial.name.rsplit("__", 1)[0], trial)
    return out


def all_cells(root: Path) -> dict[tuple[str, str], dict[str, Path]]:
    """{(model, effort): {task: trial_dir}} over the whole Terminal-Bench tree."""
    cells: dict[tuple[str, str], dict[str, Path]] = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for eff_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            tasks = cell_tasks(eff_dir)
            if tasks:
                cells[(model_dir.name, eff_dir.name)] = tasks
    return cells


def shared_tasks(root: Path) -> list[str]:
    """Tasks present in every cell, the pool a paired sample is drawn from."""
    cells = all_cells(root)
    if not cells:
        return []
    return sorted(set.intersection(*(set(t) for t in cells.values())))


def iter_work(
    root: Path,
    *,
    models: list[str] | None,
    efforts: list[str] | None,
    tasks: list[str] | None,
) -> Iterator[tuple[str, str, str, Path, str]]:
    """Yield (model, effort, task, trial_dir, outcome)."""
    keep = set(tasks) if tasks else None
    for (model, effort), by_task in sorted(all_cells(root).items()):
        if models and model not in models:
            continue
        if efforts and effort not in efforts:
            continue
        for task, trial in sorted(by_task.items()):
            if keep is not None and task not in keep:
                continue
            r = reward_of(trial)
            yield model, effort, task, trial, ("resolved" if r == 1.0 else "unresolved")
