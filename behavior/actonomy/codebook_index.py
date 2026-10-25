"""Parse `codebook.md` into a queryable hierarchy.

The codebook is the source of truth for valid `(grp, sub, leaf)` triples.
The annotator regularly emits triples that don't conform — most commonly,
the LLM puts a leaf name in the `sub` slot (collapsing two levels). This
module gives us a fast lookup so we can detect and fix those.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodebookIndex:
    # grp -> sub -> set[leaf]
    hierarchy: dict[str, dict[str, set[str]]]
    # leaf -> set[(grp, sub)]  (used to remap leaf-in-sub mistakes)
    leaf_parents: dict[str, set[tuple[str, str]]]

    @classmethod
    def from_file(cls, path: Path) -> "CodebookIndex":
        text = path.read_text()
        hierarchy: dict[str, dict[str, set[str]]] = {}
        leaf_parents: dict[str, set[tuple[str, str]]] = defaultdict(set)

        # Split by "## Group: NAME"
        groups = re.split(r"^## Group: (.+)$", text, flags=re.M)[1:]
        for i in range(0, len(groups), 2):
            grp = groups[i].strip()
            body = groups[i + 1]
            subs = re.split(r"^### Subgroup: (.+)$", body, flags=re.M)[1:]
            smap: dict[str, set[str]] = {}
            for j in range(0, len(subs), 2):
                sub = subs[j].strip()
                sb = subs[j + 1]
                # Leaves are bullet items "- **leaf name** ..."
                leaves = re.findall(r"^- \*\*(.+?)\*\*", sb, flags=re.M)
                leaves = {leaf.strip() for leaf in leaves}
                smap[sub] = leaves
                for leaf in leaves:
                    leaf_parents[leaf].add((grp, sub))
            hierarchy[grp] = smap

        return cls(hierarchy=hierarchy, leaf_parents=dict(leaf_parents))

    def validate_triple(self, grp: str, sub: str, leaf: str) -> str | None:
        """Return None if the triple is conformant; otherwise a short reason."""
        if grp not in self.hierarchy:
            return f"unknown group '{grp}'"
        if sub not in self.hierarchy[grp]:
            return f"'{sub}' is not a subgroup of '{grp}'"
        if leaf not in self.hierarchy[grp][sub]:
            return f"'{leaf}' is not a leaf under '{grp} / {sub}'"
        return None

    def suggest_remap(self, grp: str, sub: str, leaf: str) -> tuple[str, str, str] | None:
        """If the triple is non-conformant but obviously recoverable, suggest a
        replacement. Returns the suggested triple or None.

        Cases handled:
          - leaf-in-sub: `sub` is actually a leaf under some `(grp, sub')` — return
            `(grp, sub', sub)` if exactly one parent exists in the same grp.
          - leaf-under-wrong-sub: `leaf` exists under a sibling subgroup — return
            `(grp, sub_of_leaf, leaf)`.
        """
        # leaf-in-sub
        if grp in self.hierarchy and sub not in self.hierarchy[grp]:
            parents = self.leaf_parents.get(sub, set())
            in_grp = [(g, s) for (g, s) in parents if g == grp]
            if len(in_grp) == 1:
                return (grp, in_grp[0][1], sub)
        # leaf-under-wrong-sub
        if (
            grp in self.hierarchy
            and sub in self.hierarchy[grp]
            and leaf not in self.hierarchy[grp][sub]
        ):
            in_grp = [(g, s) for (g, s) in self.leaf_parents.get(leaf, set()) if g == grp]
            if len(in_grp) == 1:
                return (grp, in_grp[0][1], leaf)
        return None
