"""Shared matplotlib styling: a fixed, validated categorical palette and a
few paper-figure conventions (one axis only, historical-episode shading,
recessive gridlines) used by every chart this project produces.

Palette is the dataviz-skill reference instance's light-mode categorical
ramp (8 hues, CVD-safe in the documented adjacent-pair ordering) -- see
the skill's references/palette.md. Series/family colors are assigned in
this FIXED order project-wide so a given source family always gets the
same color across every figure, never re-cycled per chart.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
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

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dd"
SURFACE = "#fcfcfb"

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
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID_COLOR,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "figure.dpi": 150,
            "savefig.dpi": 150,
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
