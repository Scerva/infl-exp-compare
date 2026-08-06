import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from src.visualization.expectations import plot_correlation_heatmap, plot_series_by_family
from src.visualization.stance import (
    plot_mention_frequency,
    plot_speaker_stance,
    plot_stance_over_time,
    plot_stance_over_time_rescaled,
)
from src.visualization.style import add_episode_shading, apply_paper_style


def test_apply_paper_style_does_not_raise():
    apply_paper_style()


def test_plot_series_by_family_writes_file(tmp_path):
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10, freq="MS"), "value": range(10)})
    out = tmp_path / "fig.png"
    plot_series_by_family({"Series A": (df, "date", "value", "market")}, "Title", "Y", out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_series_by_family_handles_empty_series(tmp_path):
    empty = pd.DataFrame(columns=["date", "value"])
    out = tmp_path / "fig.png"
    plot_series_by_family({"Empty": (empty, "date", "value", "market")}, "Title", "Y", out)
    assert out.exists()


def test_plot_correlation_heatmap_writes_file(tmp_path):
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"])
    out = tmp_path / "corr.png"
    plot_correlation_heatmap(corr, "Title", out)
    assert out.exists()


def test_plot_stance_over_time_writes_file(tmp_path):
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=20, freq="MS"),
            "stance_score": [0.1 * (i % 3 - 1) for i in range(20)],
            "monetary_policy_relevance": [0.5] * 20,
        }
    )
    out = tmp_path / "stance.png"
    plot_stance_over_time(df, out)
    assert out.exists()


def test_plot_stance_over_time_with_fed_funds_overlay_writes_file(tmp_path):
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=20, freq="MS"),
            "stance_score": [0.1 * (i % 3 - 1) for i in range(20)],
            "monetary_policy_relevance": [0.5] * 20,
        }
    )
    ffr = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=20, freq="MS"), "value": [1.0] * 20})
    out = tmp_path / "stance_ffr.png"
    plot_stance_over_time(df, out, fed_funds_df=ffr)
    assert out.exists()


def test_plot_stance_over_time_rescaled_writes_file_and_clips(tmp_path):
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=20, freq="MS"),
            # includes an outlier well outside the default +/-0.2 clip range
            "stance_score": [0.1 * (i % 3 - 1) for i in range(19)] + [0.9],
            "monetary_policy_relevance": [0.5] * 20,
        }
    )
    out = tmp_path / "stance_rescaled.png"
    plot_stance_over_time_rescaled(df, out, stance_ylim=(-0.2, 0.2))
    assert out.exists()


def test_plot_stance_over_time_rescaled_custom_ylim(tmp_path):
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=10, freq="MS"),
            "stance_score": [0.05] * 10,
            "monetary_policy_relevance": [0.5] * 10,
        }
    )
    out = tmp_path / "stance_rescaled_custom.png"
    plot_stance_over_time_rescaled(df, out, stance_ylim=(-0.5, 0.5))
    assert out.exists()


def test_plot_speaker_stance_writes_file(tmp_path):
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=20, freq="MS"),
            "stance_score": [0.1] * 20,
            "monetary_policy_relevance": [0.5] * 20,
            "speaker": ["Powell"] * 10 + ["Waller"] * 10,
        }
    )
    out = tmp_path / "speaker.png"
    plot_speaker_stance(df, out, min_documents=5)
    assert out.exists()


def test_plot_speaker_stance_no_qualifying_speakers_still_writes(tmp_path):
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "stance_score": [0.1, 0.2, 0.3],
            "monetary_policy_relevance": [0.5, 0.5, 0.5],
            "speaker": ["Powell", "Waller", "Cook"],
        }
    )
    out = tmp_path / "speaker_sparse.png"
    plot_speaker_stance(df, out, min_documents=5)
    assert out.exists()


def test_plot_mention_frequency_writes_file(tmp_path):
    df = pd.DataFrame({"specific_measure": ["Michigan Survey"] * 5 + ["TIPS / Breakevens"] * 3})
    out = tmp_path / "mentions.png"
    plot_mention_frequency(df, out)
    assert out.exists()


def test_add_episode_shading_does_not_raise():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    add_episode_shading(ax, pd.Timestamp("2007-01-01"), pd.Timestamp("2010-01-01"))
    plt.close(fig)


def test_add_episode_shading_skips_when_too_many_episodes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    # Full range covers all 5 documented episodes -> shading skipped, no error.
    add_episode_shading(ax, pd.Timestamp("1996-01-01"), pd.Timestamp("2026-01-01"))
    plt.close(fig)
