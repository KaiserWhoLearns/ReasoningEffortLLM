"""Behavior-overview heatmap: change in Act-onomy group shares between effort endpoints.

    python behavior/actonomy/make_behavior_overview.py

Reads the validated annotations, skeletons, manifest and selection.json of both annotated
benchmarks (no model calls). For every model, the same tasks are paired at the lowest and
highest annotated reasoning-enabled setting (`none` excluded) regardless of outcome; each
trajectory contributes its macro share of quotes per group, and the heatmap shows the mean
paired change in percentage points, marked where the 90% paired-bootstrap interval
excludes zero. The verification count is the number of quotes in three Evaluate/Executing
leaves (LEAVES), reported per outcome cohort.

Writes <OUTPUT_ROOT>/figures/behavior_overview.pdf and, under <OUTPUT_ROOT>/behavior/,
behavior_overview_{trajectories,coverage,pairs,effects}.csv.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parents[1] / "figures")]
import layout  # noqa: E402
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from validate import validate_file  # noqa: E402
from common import models  # noqa: E402

STEM = "behavior_overview"
BENCHES = ("pro-corrected", "tb")
TITLES = {"pro-corrected": "SWE-Bench-Pro-Corrected", "tb": "Terminal-Bench"}
GROUPS = ("Evaluate", "Reasoning", "Planning", "Retrieval", "Executing",
          "Reflection", "Deciding", "Memory", "Grounding", "Learning")
# Verification-related leaves counted per trajectory.
LEAVES = (
    ("Evaluate", "Evaluating with gold", "Plan verification step"),
    ("Executing", "Executing plan", "Run verification command"),
    ("Evaluate", "Evaluating with gold", "Confirm actual == expected"),
)
METRICS = ("turns", "quotes", *GROUPS, "checking", *(leaf[-1] for leaf in LEAVES))


def canonical_model(name: str) -> str:
    """Model slug of a cell directory name (model slug or LiteLLM slug)."""
    for spec in models.MODELS.values():
        if name in (spec.slug, spec.litellm_id.replace("/", "_")):
            return spec.slug
    raise ValueError(f"Unregistered model: {name}")


def levels(frame: pd.DataFrame) -> list[str]:
    return sorted(frame.effort.unique(), key=style.effort_key)


def normalize(q, index):
    """The quote's (grp, sub, leaf), with an unambiguous codebook misfiling corrected."""
    triple = (q["grp"], q["sub"], q["leaf"])
    if index.validate_triple(*triple) is None:
        return triple
    replacement = index.suggest_remap(*triple)
    if replacement is None:
        parents = sorted(index.leaf_parents.get(triple[2], set()))
        if len(parents) == 1:
            replacement = (*parents[0], triple[2])
    if replacement is None or index.validate_triple(*replacement):
        raise ValueError(f"Ambiguous or invalid triple: {triple}")
    return replacement


def collect() -> tuple[pd.DataFrame, int]:
    """One row per validated annotation; also returns the number of normalized quotes."""
    index = layout.codebook_index()
    records, n_remapped = [], 0
    for bench in BENCHES:
        root = layout.bench_root(bench)
        manifest = json.loads(layout.manifest_path(bench).read_text())["entries"]
        selected = set(json.loads(layout.selection_path(bench).read_text())["instances"])
        for path in sorted((root / "annotations").glob("*/*.annotated.json")):
            skeleton = root / "skeletons" / path.parent.name / path.name.replace(".annotated.", ".skeleton.")
            validation = validate_file(path, skeleton=skeleton, index=index)
            if validation["errors"]:
                raise ValueError(f"Invalid annotation {path}: {validation['errors']}")
            doc, skel = json.loads(path.read_text()), json.loads(skeleton.read_text())
            raw_model, effort = layout.split_submission(path.parent.name)
            model = canonical_model(raw_model)
            substrate = skel["_extract"]["substrate"]
            if substrate != layout.default_substrate(bench, model):
                raise ValueError(f"Unexpected substrate {substrate}: {path}")
            meta = manifest[f"{path.parent.name}/{doc['instance_id']}"]
            if meta["outcome"] not in ("resolved", "unresolved"):
                raise ValueError(f"Missing outcome: {path}")
            counts = Counter()
            for turn in doc["turns"]:
                for quote in turn["quotes"]:
                    triple = normalize(quote, index)
                    counts[triple] += 1
                    n_remapped += triple != (quote["grp"], quote["sub"], quote["leaf"])
            total = sum(counts.values())
            record = dict(benchmark=bench, model=model, effort=effort, instance=doc["instance_id"],
                          selected=doc["instance_id"] in selected, substrate=substrate,
                          rater=doc["annotator_model"], outcome=meta["outcome"], source=meta["source"],
                          annotation=layout.rel(path), turns=len(doc["turns"]), quotes=total,
                          empty_turns=sum(not t["thought"].strip() for t in doc["turns"]),
                          reasoning_turns=sum(t.get("_x", {}).get("reasoning_chars", 0) > 0 for t in skel["turns"]))
            record.update({g: 100 * sum(n for t, n in counts.items() if t[0] == g) / total if total else np.nan
                           for g in GROUPS})
            record.update({leaf[-1]: counts[leaf] for leaf in LEAVES})
            record["checking"] = sum(counts[leaf] for leaf in LEAVES)
            records.append(record)
    frame = pd.DataFrame(records)
    keys = ["benchmark", "model", "effort", "instance"]
    assert not frame.duplicated(keys).any() and frame.rater.nunique() == 1
    # A zero-quote trajectory has undefined shares; fail rather than drop it silently.
    assert (frame.quotes > 0).all(), frame.loc[frame.quotes == 0, "annotation"].tolist()
    assert np.allclose(frame[list(GROUPS)].sum(axis=1), 100)
    return frame, n_remapped


def summarize(frame, reps, seed):
    """Endpoint coverage, matched pairs, and paired-bootstrap effects per cohort and metric."""
    rng = np.random.default_rng(seed)
    coverage, pairs, effects = [], [], []
    # Kimi K3 cells draw from the generator after the other models', which fixes the order
    # of the random draws behind the published intervals.
    cells = sorted(frame.groupby(["benchmark", "model"], sort=True),
                   key=lambda item: (item[0][1] == "kimi-k3", *item[0]))
    for (bench, model), cell in cells:
        ladder = levels(cell)
        substrates = cell.substrate.unique()
        if len(substrates) != 1:
            raise ValueError(f"Mixed annotation inputs within {bench}/{model}")
        meta = dict(benchmark=bench, model=model, low=ladder[0], high=ladder[-1],
                    substrate=substrates[0], annotated_runs=len(cell))
        if len(ladder) == 1:
            coverage.append(dict(**meta, n_pairs=0, status="one setting"))
            continue
        low = cell[cell.effort == ladder[0]].set_index("instance")
        high = cell[cell.effort == ladder[-1]].set_index("instance")
        ids = sorted(set(low.index) & set(high.index))
        if not ids:
            raise ValueError(f"No paired endpoint tasks for {bench}/{model}")
        low, high = low.loc[ids], high.loc[ids]
        coverage.append(dict(**meta, n_pairs=len(ids), status="paired"))
        local = []
        for instance in ids:
            lp, hp = low.loc[instance, "outcome"] == "resolved", high.loc[instance, "outcome"] == "resolved"
            cohort = {(True, True): "both_pass", (False, False): "both_fail",
                      (False, True): "gain", (True, False): "loss"}[lp, hp]
            rec = dict(**meta, instance=instance, cohort=cohort)
            for metric in METRICS:
                rec["low_" + metric] = float(low.loc[instance, metric])
                rec["high_" + metric] = float(high.loc[instance, metric])
            local.append(rec)
        pairs.extend(local)
        local = pd.DataFrame(local)
        for cohort in ("all", "both_pass", "both_fail", "gain", "loss"):
            subset = local if cohort == "all" else local[local.cohort == cohort]
            if subset.empty:
                continue
            for metric in METRICS:
                lv, hv = subset["low_" + metric].to_numpy(), subset["high_" + metric].to_numpy()
                delta = hv - lv
                draws = rng.integers(0, len(delta), size=(reps, len(delta)))
                ci_low, ci90_low, ci90_high, ci_high = np.quantile(
                    delta[draws].mean(axis=1), [.025, .05, .95, .975])
                effects.append(dict(**meta, cohort=cohort, metric=metric, n_pairs=len(subset),
                                    low_mean=lv.mean(), high_mean=hv.mean(), delta=delta.mean(),
                                    ci_low=ci_low, ci_high=ci_high,
                                    ci90_low=ci90_low, ci90_high=ci90_high))
    return pd.DataFrame(coverage), pd.DataFrame(pairs), pd.DataFrame(effects)


def render(coverage, effects, out: Path):
    style.apply_rc()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7.5,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "savefig.transparent": False})
    # Models with a single annotated setting have no paired contrast and are not drawn.
    coverage = coverage[coverage.n_pairs > 0]
    fig = plt.figure(figsize=(9.2, 2.65))
    grid = fig.add_gridspec(1, 2, left=.14, right=.995, top=.99, bottom=.33,
                            wspace=.38)
    shares = effects[(effects.cohort == "all") & effects.metric.isin(GROUPS)]
    vmax = max(5, np.ceil(shares.delta.abs().max() / 5) * 5)
    cmap = plt.get_cmap("PuOr_r").copy()
    cmap.set_bad("#eeeeee")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for b, bench in enumerate(BENCHES):
        cov = coverage[coverage.benchmark == bench].set_index("model")
        names = sorted(cov.index, key=style.model_sort_key)
        ax_mix = fig.add_subplot(grid[0, b])
        ax_mix.set_ylim(len(names) - .5, -.5)
        ax_mix.set_yticks([])
        for s in ax_mix.spines.values():
            s.set_visible(False)
        matrix = np.full((len(names), len(GROUPS)), np.nan)
        significant = np.zeros(matrix.shape, dtype=bool)
        for y, model in enumerate(names):
            meta = cov.loc[model]
            sub = effects[(effects.benchmark == bench) & (effects.model == model)]
            ax_mix.text(-.65, y - .17, style.label_of(model), ha="right", va="center",
                        weight="medium", fontsize=7, color=style.style_of(model).color)
            ax_mix.text(-.65, y + .2, f"{meta.low} → {meta.high}", ha="right", va="center",
                        fontsize=6, color="#666666")
            intervals = sub[sub.cohort == "all"].set_index("metric").loc[list(GROUPS)]
            matrix[y] = intervals["delta"]
            # Pointwise, unadjusted 90% paired-bootstrap intervals on the unrounded bounds;
            # a singleton or undefined interval is never marked.
            significant[y] = ((intervals.n_pairs >= 2)
                              & np.isfinite(intervals.ci90_low)
                              & np.isfinite(intervals.ci90_high)
                              & ((intervals.ci90_low > 0) | (intervals.ci90_high < 0)))
        img = ax_mix.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", zorder=-1)
        for y in range(len(names)):
            for x in range(len(GROUPS)):
                if np.isfinite(matrix[y, x]):
                    v = matrix[y, x]
                    label = f"{v:+.0f}" if abs(v) >= .5 else "0"
                    if significant[y, x]:
                        label += r"$^{\dagger}$"
                    ax_mix.text(x, y, label, ha="center", va="center", fontsize=6.5,
                                color="white" if abs(v) > .65 * vmax else "#333333")
        ax_mix.set_xticks(range(len(GROUPS)), GROUPS, rotation=50, ha="right", rotation_mode="anchor")
        ax_mix.tick_params(axis="x", length=0, pad=3)
        ax_mix.set_ylabel(TITLES[bench], fontsize=8.5, weight="bold")
        # Raise the Pro title above the long label in its last row.
        ax_mix.yaxis.set_label_coords(-.275, .56 if bench == "pro-corrected" else .5)
    cbax = fig.add_axes([.39, .115, .30, .022])
    cb = fig.colorbar(img, cax=cbax, orientation="horizontal", ticks=[-vmax, 0, vmax])
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("Change in share (percentage points)", fontsize=7, labelpad=2)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=.025,
                facecolor="white", transparent=False)
    plt.close(fig)


def print_checking(effects):
    """Mean verification count at the low and high endpoint, by outcome cohort."""
    rows = effects[(effects.metric == "checking") & effects.cohort.isin(["both_pass", "both_fail"])]
    print("\nVerification count (mean per trajectory, low -> high endpoint):")
    for (bench, model), g in rows.groupby(["benchmark", "model"], sort=False):
        cells = "  ".join(f"{r.cohort}: {r.low_mean:.1f} -> {r.high_mean:.1f} (n={r.n_pairs})"
                          for r in g.itertuples())
        print(f"  {bench:14} {model:18} {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--bootstrap", type=int, default=10000, help="bootstrap replicates")
    args = parser.parse_args()
    tables_dir = layout.paths.output_dir("behavior", create=True)
    figure = layout.paths.output_dir("figures", create=True) / f"{STEM}.pdf"

    frame, n_remapped = collect()
    coverage, pairs, effects = summarize(frame[frame.selected & (frame.effort != "none")],
                                         args.bootstrap, args.seed)
    for name, table in (("trajectories", frame), ("coverage", coverage),
                        ("pairs", pairs), ("effects", effects)):
        table.to_csv(tables_dir / f"{STEM}_{name}.csv", index=False)
    render(coverage, effects, figure)
    print(coverage.to_string(index=False))
    print(f"Validated {len(frame)} annotations ({n_remapped} quotes normalized to their "
          f"codebook parent); {len(pairs)} endpoint pairs.")
    print_checking(effects)
    print(f"\nwrote {figure} and {STEM}_*.csv in {tables_dir}")


if __name__ == "__main__":
    main()
