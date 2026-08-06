"""Figures for Fed hawkish/dovish stance (spec section 12, Figures 3-4, 9)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.style import CATEGORICAL_PALETTE, GRID_COLOR, add_episode_shading, apply_paper_style


def plot_stance_over_time(documents_df: pd.DataFrame, out_path: Path, rolling_window: int = 10) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    df = documents_df.sort_values("event_date")
    relevant = df[df["monetary_policy_relevance"] >= 0.10]

    ax.scatter(
        relevant["event_date"], relevant["stance_score"],
        s=8, color=CATEGORICAL_PALETTE[2], alpha=0.45, label="Document stance (dictionary model)",
    )
    rolling = relevant.set_index("event_date")["stance_score"].rolling(rolling_window, min_periods=3).mean()
    ax.plot(rolling.index, rolling.values, color=CATEGORICAL_PALETTE[3], linewidth=2.2,
            label=f"{rolling_window}-document rolling mean")
    ax.axhline(0, color=GRID_COLOR, linewidth=1.0, zorder=0)

    if len(relevant):
        add_episode_shading(ax, relevant["event_date"].min(), relevant["event_date"].max())

    ax.set_title("Federal Reserve Hawkish/Dovish Stance Through Time")
    ax.set_ylabel("Stance score (+ hawkish / - dovish)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
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
