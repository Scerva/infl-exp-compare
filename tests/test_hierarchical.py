import numpy as np
import pandas as pd
import pytest

from src.models.hierarchical import fit_hierarchical_speaker_model


def _synthetic_sample(n_speakers=6, docs_per_speaker=15, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_speakers):
        speaker_slope = rng.normal(0.1, 0.05)  # each speaker has a somewhat different slope
        for _ in range(docs_per_speaker):
            x = rng.normal(0, 1)
            control = rng.normal(0, 1)
            y = speaker_slope * x + 0.02 * control + rng.normal(0, 0.05)
            rows.append({"speaker": f"speaker_{i}", "x_zscore": x, "control": control, "dv": y})
    df = pd.DataFrame(rows)
    df["event_date"] = pd.date_range("2018-01-01", periods=len(df), freq="D")
    return df


def test_fit_hierarchical_speaker_model_returns_committee_average():
    df = _synthetic_sample()
    result = fit_hierarchical_speaker_model(df, "dv", "x_zscore", control_cols=["control"], min_speaker_docs=5)
    assert result.n_groups == 6
    assert result.committee_avg_coef > 0  # all speaker slopes were drawn positive on average


def test_fit_hierarchical_speaker_model_speaker_effects_cover_all_speakers():
    df = _synthetic_sample()
    result = fit_hierarchical_speaker_model(df, "dv", "x_zscore", control_cols=["control"], min_speaker_docs=5)
    assert set(result.speaker_effects["speaker"]) == {f"speaker_{i}" for i in range(6)}
    assert (result.speaker_effects["n_documents"] == 15).all()


def test_fit_hierarchical_speaker_model_raises_with_too_few_speakers():
    df = _synthetic_sample(n_speakers=2)
    with pytest.raises(ValueError):
        fit_hierarchical_speaker_model(df, "dv", "x_zscore", control_cols=["control"], min_speaker_docs=5)


def test_fit_hierarchical_speaker_model_marks_low_count_speakers_ineligible():
    df = _synthetic_sample(n_speakers=5, docs_per_speaker=15)
    sparse_rows = df[df["speaker"] == "speaker_0"].head(2).copy()
    sparse_rows["speaker"] = "speaker_sparse"
    df = pd.concat([df, sparse_rows], ignore_index=True)
    result = fit_hierarchical_speaker_model(df, "dv", "x_zscore", control_cols=["control"], min_speaker_docs=5)
    sparse_row = result.speaker_effects[result.speaker_effects["speaker"] == "speaker_sparse"]
    assert not sparse_row.empty
    assert not sparse_row.iloc[0]["eligible_for_own_slope"]
