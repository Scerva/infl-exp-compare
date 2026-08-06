"""Figures for Fed hawkish/dovish stance (spec section 12, Figures 3-4, 9)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.style import CATEGORICAL_PALETTE, GRID_COLOR, add_episode_shading, apply_paper_style


def plot_stance_over_time(
    documents_df: pd.DataFrame,
    out_path: Path,
    rolling_window: int = 10,
    fed_funds_df: pd.DataFrame | None = None,
) -> None:
    """fed_funds_df, if given: columns [date, value] (effective federal
    funds rate, percent), overlaid on a secondary y-axis so the stance
    score can be read directly against the actual policy rate path.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    df = documents_df.sort_values("event_date")
    relevant = df[df["monetary_policy_relevance"] >= 0.10]

    ax.scatter(
        relevant["event_date"], relevant["stance_score"],
        s=8, color=CATEGORICAL_PALETTE[2], alpha=0.45, label="Document stance (dictionary model)",
        zorder=3,
    )
    rolling = relevant.set_index("event_date")["stance_score"].rolling(rolling_window, min_periods=3).mean()
    ax.plot(rolling.index, rolling.values, color=CATEGORICAL_PALETTE[3], linewidth=2.2,
            label=f"{rolling_window}-document rolling mean", zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=1.0, zorder=0)

    if len(relevant):
        add_episode_shading(ax, relevant["event_date"].min(), relevant["event_date"].max())

    ax.set_title("Federal Reserve Hawkish/Dovish Stance Through Time")
    ax.set_ylabel("Stance score (+ hawkish / - dovish)")

    handles, labels = ax.get_legend_handles_labels()
    if fed_funds_df is not None and not fed_funds_df.empty:
        ax2 = ax.twinx()
        ax2.plot(
            fed_funds_df["date"], fed_funds_df["value"],
            color=CATEGORICAL_PALETTE[0], linewidth=1.6, linestyle="--", alpha=0.85,
            label="Effective federal funds rate (%)", zorder=2,
        )
        ax2.set_ylabel("Effective federal funds rate (%)")
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    ax.legend(handles, labels, loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_stance_over_time_rescaled(
    documents_df: pd.DataFrame,
    out_path: Path,
    rolling_window: int = 10,
    fed_funds_df: pd.DataFrame | None = None,
    stance_ylim: tuple[float, float] = (-0.2, 0.2),
    start_date: str | pd.Timestamp | None = "2004-01-01",
) -> None:
    """Figure 3b: same chart as plot_stance_over_time, with the stance
    axis clipped to a fixed, readable range instead of autoscaling to
    whatever the single most extreme document happens to be.

    In the underlying data, 98% of document-level stance scores fall
    within [-0.17, 0.16] and the entire 10-document rolling mean stays
    within [-0.08, 0.13] -- but a handful of outlier speeches (as extreme
    as -0.56) drag a naively autoscaled axis out to +/-0.6, which
    flattens the actually-informative rolling-mean line into a near-
    straight line at the scale needed to fit those outliers. Clipping to
    a fixed +/-0.2 keeps >98% of points on-plot at a scale where the
    rolling mean's real cyclical swings are visible, and reads
    comparably alongside the federal funds rate axis. The <2% of points
    outside this range are cropped, not hidden data -- see Figure 3 for
    the unclipped version.

    ``start_date`` crops the x-axis (default 2004, past the thin/noisy
    1996-2003 tail of the corpus). The federal-funds-rate axis is
    plotted with SYMMETRIC limits (-max, +max) so its zero sits at the
    vertical center of the plot, exactly where the stance axis's zero
    already sits (stance_ylim is symmetric by construction) -- the two
    zero-lines align at the same height even though the funds rate
    itself never goes negative, which leaves the bottom half of that
    axis empty by design, not by accident.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    df = documents_df.sort_values("event_date")
    relevant = df[df["monetary_policy_relevance"] >= 0.10]
    if start_date is not None:
        relevant = relevant[relevant["event_date"] >= pd.Timestamp(start_date)]

    ax.scatter(
        relevant["event_date"], relevant["stance_score"],
        s=8, color=CATEGORICAL_PALETTE[2], alpha=0.45, label="Document stance (dictionary model)",
        zorder=3, clip_on=True,
    )
    rolling = relevant.set_index("event_date")["stance_score"].rolling(rolling_window, min_periods=3).mean()
    ax.plot(rolling.index, rolling.values, color=CATEGORICAL_PALETTE[3], linewidth=2.4,
            label=f"{rolling_window}-document rolling mean", zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=1.0, zorder=0)
    ax.set_ylim(*stance_ylim)
    if start_date is not None and len(relevant):
        ax.set_xlim(pd.Timestamp(start_date), relevant["event_date"].max())

    if len(relevant):
        add_episode_shading(ax, relevant["event_date"].min(), relevant["event_date"].max())

    ax.set_title("Federal Reserve Hawkish/Dovish Stance Through Time (rescaled)")
    ax.set_ylabel("Stance score (+ hawkish / - dovish)")

    handles, labels = ax.get_legend_handles_labels()
    if fed_funds_df is not None and not fed_funds_df.empty:
        ffr = fed_funds_df
        if start_date is not None:
            ffr = ffr[ffr["date"] >= pd.Timestamp(start_date)]
        ax2 = ax.twinx()
        ax2.plot(
            ffr["date"], ffr["value"],
            color=CATEGORICAL_PALETTE[0], linewidth=1.8, linestyle="--", alpha=0.9,
            label="Effective federal funds rate (%)", zorder=2,
        )
        ax2.set_ylabel("Effective federal funds rate (%)")
        ffr_max = float(ffr["value"].max()) * 1.05 if not ffr.empty else 1.0
        ax2.set_ylim(-ffr_max, ffr_max)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    ax.legend(handles, labels, loc="lower left", fontsize=8, frameon=False)
    n_clipped = int(((relevant["stance_score"] < stance_ylim[0]) | (relevant["stance_score"] > stance_ylim[1])).sum())
    if n_clipped:
        ax.annotate(
            f"{n_clipped} of {len(relevant)} points ({100 * n_clipped / len(relevant):.1f}%) fall outside "
            f"[{stance_ylim[0]}, {stance_ylim[1]}] and are cropped, not omitted from the underlying data.",
            xy=(0.5, -0.16), xycoords="axes fraction", ha="center", fontsize=7, color=CATEGORICAL_PALETTE[1],
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_speaker_stance(documents_df: pd.DataFrame, out_path: Path, min_documents: int = 8) -> None:
    apply_paper_style()
    df = documents_df[(documents_df["monetary_policy_relevance"] >= 0.10) & documents_df["speaker"].notna()]
    counts = df["speaker"].value_counts()
    speakers = counts[counts >= min_documents].index.tolist()
    df = df[df["speaker"].isin(speakers)]

    stats = df.groupby("speaker")["stance_score"].agg(["mean", "std", "count"]).sort_values("mean")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(stats))))
    y = range(len(stats))
    ax.barh(y, stats["mean"], xerr=stats["std"].fillna(0), color=CATEGORICAL_PALETTE[0], alpha=0.9, height=0.6,
            ecolor=CATEGORICAL_PALETTE[1])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} (n={int(c)})" for s, c in zip(stats.index, stats["count"])], fontsize=7)
    ax.axvline(0, color=GRID_COLOR, linewidth=1.0)
    ax.set_xlabel("Mean stance score (+ hawkish / - dovish), +/- 1 s.d.")
    ax.set_title(f"Speaker-Level Average Stance (speakers with >= {min_documents} relevant documents)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_mention_frequency(mentions_df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    apply_paper_style()
    counts = mentions_df["specific_measure"].value_counts().head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(counts))))
    ax.barh(range(len(counts)), counts.values, color=CATEGORICAL_PALETTE[3])
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.set_xlabel("Number of direct mentions across corpus")
    ax.set_title("Direct References to Inflation-Expectations Measures")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_family_divergence(family_panel: pd.DataFrame, date_col: str, out_path: Path) -> None:
    """Standardized-spread 'divergence' figure (spec section 25 concept,
    shown descriptively here): consumer - market and professional -
    market standardized spreads over time.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    spreads = {
        "Household - Market": (CATEGORICAL_PALETTE[0], "household_minus_market"),
        "Professional - Market": (CATEGORICAL_PALETTE[3], "professional_minus_market"),
    }
    plotted = False
    for label, (color, col) in spreads.items():
        if col in family_panel.columns:
            d = family_panel[[date_col, col]].dropna()
            if not d.empty:
                ax.plot(d[date_col], d[col], color=color, linewidth=1.4, label=label)
                plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0, color=GRID_COLOR, linewidth=1.0, zorder=0)
    ax.set_title("Standardized Divergence Across Expectation-Measure Families")
    ax.set_ylabel("Standardized spread (z-score units)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
