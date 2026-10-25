"""Shared visual language for the figures.

A colour is a model and nothing else: each model has one colour (Okabe-Ito where possible, so
the set is colour-blind safe) and one marker, so identity survives greyscale printing. Reasoning
effort is always position along an ordered axis plus a text label, sorted by
`common.models.EFFORT_ORDER`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

from common.models import EFFORT_ORDER  # noqa: E402

# Off-white surface and recessive axes.
INK, MUTED, GRID, SURFACE = "#1a1a1a", "#6b6b6b", "#e4e2dd", "#fcfcfb"

# Liberation Sans (Arial metrics); `use_font` falls back to Arial, then DejaVu Sans.
FONT = "Liberation Sans"


@dataclass(frozen=True)
class ModelStyle:
    label: str
    color: str
    marker: str


# Display order is the order of this dict: GPT tiers, Claude, then the Together-hosted models.
MODELS: dict[str, ModelStyle] = {
    # Two blues for two tiers of one model family; the markers keep them apart in greyscale.
    "gpt-5.6-luna":    ModelStyle("GPT-5.6-luna",    "#56B4E9", "D"),   # sky blue
    "gpt-5.6-terra":   ModelStyle("GPT-5.6-terra",   "#0072B2", "s"),   # blue
    "gpt-5.6-sol":     ModelStyle("GPT-5.6-sol",     "#009E73", "^"),   # bluish green
    "claude-sonnet-5": ModelStyle("Claude Sonnet 5", "#D55E00", "o"),   # vermillion
    "glm-5.2":         ModelStyle("GLM-5.2",         "#E69F00", "P"),   # orange
    "inkling":         ModelStyle("Inkling",         "#000000", "X"),   # black
    # Okabe-Ito's remaining yellow vanishes on this surface, so the last two models borrow Paul
    # Tol's muted wine and teal.
    "kimi-k3":         ModelStyle("Kimi K3",         "#882255", "*"),   # wine
    "deepseek-v4-flash": ModelStyle("DeepSeek V4 Flash", "#44AA99", "v"),  # teal
}
FALLBACK = ModelStyle("", "#6b6b6b", "o")


def style_of(model: str) -> ModelStyle:
    """Style for a canonical model slug; an unregistered slug gets grey and its own name."""
    if model in MODELS:
        return MODELS[model]
    return ModelStyle(model, FALLBACK.color, FALLBACK.marker)


def label_of(model: str) -> str:
    return style_of(model).label or model


def model_sort_key(model: str) -> int:
    keys = list(MODELS)
    return keys.index(model) if model in keys else len(keys)


def effort_key(effort: str) -> float:
    """Position on the reasoning ladder; unknown levels sort last but never raise."""
    return EFFORT_ORDER.get(effort, 99.0)


def use_font(family: str = FONT) -> str:
    """Register the first of `family`, Arial, DejaVu Sans installed on this machine.

    The system font files are registered in-process, since matplotlib's own font cache may hold
    only its bundled faces. Arial is the fallback because Liberation Sans shares its metrics.
    """
    global _FONT_RESOLVED
    if _FONT_RESOLVED is None:
        installed = fm.findSystemFonts()
        for candidate in (family, "Arial", "DejaVu Sans"):
            files = [p for p in installed
                     if Path(p).stem.split("-")[0] == candidate.replace(" ", "")]
            for path in files:
                fm.fontManager.addfont(path)
            if files:
                if candidate != family:
                    print(f"note: {family} not installed — figures fall back to {candidate}")
                _FONT_RESOLVED = candidate
                break
        else:
            print(f"note: {family} not installed — figures fall back to DejaVu Sans")
            _FONT_RESOLVED = "DejaVu Sans"
    return _FONT_RESOLVED


_FONT_RESOLVED: str | None = None


def apply_rc() -> None:
    plt.rcParams.update({
        "font.family": use_font(), "font.size": 11,
        "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": MUTED,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })


def despine(ax) -> None:
    """Strip the frame to a single hairline baseline and drop the tick marks (minor ones too)."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="y", which="both", length=0, colors=MUTED)
    ax.tick_params(axis="x", which="both", length=0, pad=6)


# `*` and `+`-based markers carry less ink than a square or a circle at the same nominal size.
MARKER_SCALE = {"*": 1.5, "P": 1.15, "X": 1.15, "^": 1.05}


def marker_size(marker: str, base: float) -> float:
    """Point size for `marker` at a nominal `base` (matplotlib `ms`/`markersize`)."""
    return base * MARKER_SCALE.get(marker, 1.0)


def money_tick(v: float, _pos=None) -> str:
    """A tick label on a dollar axis, at the shortest precision that separates two ticks."""
    if v == 0:
        return "$0"
    if v >= 1:
        return f"${v:,.0f}" if abs(v - round(v)) < 1e-9 else f"${v:,.2f}"
    return f"${v:.2f}" if v >= 0.01 else f"${v:.3f}"


# Ticks for rungs that hold more than one effort name: `max` and `xhigh` are each vendor's top
# level and share one slot (`datasets.RUNG_ALIAS`).
MERGED_RUNG_TICK = {"xhigh": ("xhigh/max", "xhigh/\nmax")}


def rung_tick(rung: str, wrap: bool = False) -> str:
    """Tick text for a rung on a SHARED effort axis; `wrap` breaks it over two lines.

    Wrapped on axes narrow enough that "xhigh/max" runs into the `high` label next to it.
    """
    return MERGED_RUNG_TICK.get(rung, (rung, rung))[1 if wrap else 0]


def fmt_money(v: float) -> str:
    """Dollars at a precision scaled to magnitude, so adjacent effort levels stay distinct."""
    if v >= 100:
        return f"${v:.0f}"
    if v >= 1:
        return f"${v:.2f}"
    if v >= 0.1:
        return f"${v:.3f}"
    return f"${v:.4f}"


def use_log_x(values: list[float], ratio: float = 15.0) -> bool:
    """Log the cost axis only when the cheapest and priciest run differ by >= `ratio`."""
    vals = [v for v in values if v and v > 0]
    return bool(vals) and max(vals) / min(vals) >= ratio
