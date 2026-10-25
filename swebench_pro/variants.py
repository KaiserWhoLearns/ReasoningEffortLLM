"""The two SWE-Bench Pro environments, and the config stack each (dataset, model) runs with.

| `--dataset`     | benchmark id             | environment                                          |
|-----------------|--------------------------|------------------------------------------------------|
| `pro`           | `swebench-pro`           | the official images, unmodified                      |
| `pro-corrected` | `swebench-pro-corrected` | git history stripped + egress limited to registries  |

The official images ship the repository's full clone, so the instance's own fix commit is
reachable from `/app`. SWE-Bench Pro-Corrected removes it in both directions:

* `configs/strip_history.yaml` replaces `/app/.git` with a single-commit repository at the base
  commit before the agent's first step, keeping the starting file tree and any pre-existing
  working-tree changes (`common/sanitized_docker.py` runs it);
* `configs/seal_network.yaml` puts the container on an `--internal` Docker network whose only exit
  is a proxy that allows the package registries in `common/egress_proxy/allowlist.txt`
  (`common/sealed_docker.py`).

Both datasets use the same tasks (`mini_instances.jsonl`, `subsets/pilot.txt`) and the same
evaluator: `evaluate.py` scores every patch in a fresh container from the original image, which
still has the history the evaluator needs to check out the test files.

The scaffold is a property of the model, not of the dataset: models in
`common.models.REQUIRES_TOOLCALL` get `configs/toolcall.yaml` appended to the stack (mini's native
bash tool); every other model uses the text-based scaffold (exactly one ```bash block per reply).
All models of one dataset write to the same tree, and the run manifest records the scaffold.

(The module is `variants.py`, not `datasets.py`, so it does not shadow HuggingFace's `datasets`
package when this directory is on sys.path.)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from common import models

CONFIGS = Path(__file__).resolve().parent / "configs"
DEFAULT = "pro"
TOOLCALL_CONFIG = "toolcall.yaml"
SCAFFOLD_TEXT = "text-based"
SCAFFOLD_TOOLCALL = "native-tool-calling"


@dataclass(frozen=True)
class Dataset:
    key: str
    """The `--dataset` value."""
    benchmark: str
    """Benchmark id in common/paths.py; decides the result tree."""
    title: str
    configs: tuple[str, ...]
    """Config files under configs/, deep-merged left to right."""
    env_note: str = ""
    """One line for reports and manifests describing the environment."""
    tags: dict = field(default_factory=dict)
    """Facts about the environment recorded in every run manifest."""

    @property
    def git_history_stripped(self) -> bool:
        return bool(self.tags.get("git_history_stripped"))

    @property
    def egress_sealed(self) -> bool:
        return self.tags.get("egress") == "allowlist-proxy"


DATASETS: dict[str, Dataset] = {
    "pro": Dataset(
        key="pro",
        benchmark="swebench-pro",
        title="SWE-Bench Pro",
        configs=("base.yaml",),
        env_note="official images, unmodified: the container keeps the repository's full git "
                 "history and has unrestricted network access",
        tags={"git_history_stripped": False, "egress": "unrestricted"},
    ),
    "pro-corrected": Dataset(
        key="pro-corrected",
        benchmark="swebench-pro-corrected",
        title="SWE-Bench Pro-Corrected",
        # seal_network.yaml replaces strip_history.yaml's environment class with a subclass that
        # also seals egress, so it must come last.
        configs=("base.yaml", "strip_history.yaml", "seal_network.yaml"),
        env_note="/app/.git is replaced by a single-commit repo at the base commit before the "
                 "agent's first step, and the container's only network egress is a proxy that "
                 "allows package registries and nothing else",
        tags={"git_history_stripped": True, "egress": "allowlist-proxy"},
    ),
}


def add_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT,
                        help=f"`pro` (official environment, default) or `pro-corrected` "
                             f"(git history stripped, egress limited to package registries)")


def resolve(key: str) -> Dataset:
    if key not in DATASETS:
        raise SystemExit(f"unknown dataset {key!r}; choose from {sorted(DATASETS)}")
    return DATASETS[key]


def scaffold(model: str) -> str:
    """`native-tool-calling` for models in models.REQUIRES_TOOLCALL, else `text-based`."""
    return SCAFFOLD_TOOLCALL if models.requires_toolcall(model) else SCAFFOLD_TEXT


def config_paths(dataset: Dataset, model: str | None = None) -> list[Path]:
    """The config files merged for this dataset, plus toolcall.yaml when the model needs it."""
    names = list(dataset.configs)
    if model is not None and scaffold(model) == SCAFFOLD_TOOLCALL:
        names.append(TOOLCALL_CONFIG)
    return [CONFIGS / name for name in names]


def merged_config(dataset: Dataset, model: str | None = None) -> dict:
    """Deep-merge the config stack with mini-swe-agent's own loader and merge rule."""
    from minisweagent.config import get_config_from_spec
    from minisweagent.utils.serialize import recursive_merge

    return recursive_merge(*(get_config_from_spec(str(p)) for p in config_paths(dataset, model)))
