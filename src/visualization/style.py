"""Shared matplotlib styling: a fixed brand palette (navy / gray / silver
blue / brick red accent) and a few paper-figure conventions (one axis
only, historical-episode shading, recessive gridlines, serif type) used
by every chart this project produces.

Palette: navy is primary (used first, and for the "positive/lead" series
in any two-series chart), gray is secondary, silver blue is tertiary, and
a muted brick red is reserved as the accent -- used sparingly, for the
single most important highlighted element in a chart (e.g. a
statistically significant bar, the second line in a two-line
comparison), never as just "the next color in rotation." Additional
slots extend each hue's own family (a darker/lighter step) so an 8-10
series chart stays visually coherent instead of reaching for unrelated
hues.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

NAVY = "#1B2A4A"
GRAY = "#75787B"
SILVER_BLUE = "#8FA6BF"
ACCENT_RED = "#8C3B34"

CATEGORICAL_PALETTE = [
    NAVY,        # 1 primary
    GRAY,        # 2 secondary
    SILVER_BLUE, # 3 tertiary
    ACCENT_RED,  # 4 accent
    "#3D5473",   # 5 lighter navy
    "#A8ABAE",   # 6 lighter gray
    "#5C7690",   # 7 darker silver blue
    "#B06B62",   # 8 lighter red
]

# Fixed source-family -> color assignment, used everywhere a family is plotted.
FAMILY_COLORS = {
    "market": CATEGORICAL_PALETTE[0],
    "model": CATEGORICAL_PALETTE[1],
    "household": CATEGORICAL_PALETTE[2],
    "professional": CATEGORICAL_PALETTE[3],
    "business": CATEGORICAL_PALETTE[4],
    "survey_generic": CATEGORICAL_PALETTE[5],
    "horizon_generic": CATEGORICAL_PALETTE[6],
    "communications": CATEGORICAL_PALETTE[7],
}

TEXT_PRIMARY = "#000000"
TEXT_SECONDARY = "#4A4D50"
GRID_COLOR = "#D9DBDC"
SURFACE = "#FFFFFF"
NEUTRAL_MARK = "#B7B9BB"  # non-significant / de-emphasized marks

# Navy-white-red diverging colormap for correlation-style heatmaps, built
# from the same brand colors rather than a generic default (matplotlib's
# RdBu_r): negative = accent red, zero = white, positive = navy.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "navy_red_diverging", [ACCENT_RED, "#FFFFFF", NAVY]
)

# Serif stack for a more editorial, social-share-ready look. Georgia is
# widely available and reads well at small preview sizes; the rest are
# fallbacks in descending preference.
SERIF_STACK = ["Georgia", "Palatino", "Times New Roman", "DejaVu Serif", "serif"]

# Episodes worth flagging when they fall within a figure's date range;
# rendered as light vertical spans with a small top label, never forced
# onto a chart where they would clutter it (spec section 12).
HISTORICAL_EPISODES = [
    ("2007-12-01", "2009-06-01", "GFC"),
    ("2013-05-01", "2013-09-01", "Taper tantrum"),
    ("2020-02-01", "2020-06-01", "Pandemic onset"),
    ("2021-03-01", "2021-12-01", "2021 inflation accel."),
    ("2022-03-01", "2023-07-01", "2022-23 tightening"),
]


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF_STACK,
            "mathtext.fontset": "dejavuserif",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": TEXT_PRIMARY,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.5,
            "legend.fontsize": 10.5,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.facecolor": SURFACE,
        }
    )


def add_episode_shading(ax, date_min: pd.Timestamp, date_max: pd.Timestamp) -> None:
    """Light vertical spans for historical episodes overlapping the
    figure's date range. Skips entirely if it would add more than 3
    labels (avoids clutter on already-busy figures).
    """
    visible = [
        (s, e, label)
        for s, e, label in HISTORICAL_EPISODES
        if pd.Timestamp(e) >= date_min and pd.Timestamp(s) <= date_max
    ]
    if len(visible) > 3:
        return
    ylim = ax.get_ylim()
    for start, end, label in visible:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=GRID_COLOR, alpha=0.6, zorder=0)
        ax.text(
            pd.Timestamp(start), ylim[1], label, fontsize=7, color=TEXT_SECONDARY,
            rotation=90, va="top", ha="right", alpha=0.8,
        )
