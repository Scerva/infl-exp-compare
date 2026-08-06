"""Figures for inflation-expectations time series (spec section 12,
Figures 1-2, Table 3)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.style import CATEGORICAL_PALETTE, add_episode_shading, apply_paper_style


def plot_series_by_family(
    series: dict[str, tuple[pd.DataFrame, str, str, str]],
    title: str,
    ylabel: str,
    out_path: Path,
    standardized: bool = False,
) -> None:
    """series: {display_name: (df, date_col, value_col, family)}.

    One line per series, colored by a hash of series NAME within a fixed
    palette rotation (family grouping is conveyed via the legend text,
    not color, since more than 8 individual series would exceed the
    CVD-safe categorical count -- see dataviz skill's all-pairs cap).
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    date_min, date_max = None, None
    for i, (label, (df, date_col, value_col, family)) in enumerate(series.items()):
        d = df[[date_col, value_col]].dropna()
        if d.empty:
            continue
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        ax.plot(d[date_col], d[value_col], label=label, color=color, linewidth=1.4)
        date_min = d[date_col].min() if date_min is None else min(date_min, d[date_col].min())
        date_max = d[date_col].max() if date_max is None else max(date_max, d[date_col].max())

    if date_min is not None:
        add_episode_shading(ax, date_min, date_max)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="#c3c2b7", linewidth=0.8, zorder=0) if standardized else None
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, out_path: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(val) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
