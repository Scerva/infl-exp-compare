import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "stage4", Path(__file__).resolve().parents[1] / "scripts" / "04_build_model_panel.py"
)
stage4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage4)


def test_fred_release_date_same_day_rule():
    dates = pd.Series(pd.to_datetime(["2024-01-15", "2024-02-20"]))
    out = stage4._fred_release_date("T5YIE", dates)
    assert list(out) == list(dates)


def test_fred_release_date_mid_month_rule():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-03-01"]))
    out = stage4._fred_release_date("MICH", dates)
    assert out.iloc[0] == pd.Timestamp("2024-01-14")
    assert out.iloc[1] == pd.Timestamp("2024-03-14")


def test_fred_release_date_unknown_series_raises():
    with pytest.raises(KeyError):
        stage4._fred_release_date("NOT_A_SERIES", pd.Series(pd.to_datetime(["2024-01-01"])))


def test_add_speaker_relative_features_first_speech_has_no_delta():
    panel = pd.DataFrame(
        {
            "document_id": ["a", "b", "c"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "speaker": ["Powell", "Powell", "Waller"],
            "document_type": ["speech", "speech", "speech"],
            "stance_score": [0.2, 0.5, -0.1],
        }
    )
    out = stage4._add_speaker_relative_features(panel)
    powell = out[out["speaker"] == "Powell"].sort_values("event_date")
    assert pd.isna(powell.iloc[0]["delta_stance"])
    assert powell.iloc[1]["delta_stance"] == pytest.approx(0.3)
    waller = out[out["speaker"] == "Waller"]
    assert pd.isna(waller.iloc[0]["delta_stance"])


def test_add_speaker_relative_features_fomc_documents_grouped_by_type():
    panel = pd.DataFrame(
        {
            "document_id": ["m1", "m2"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "speaker": [None, None],
            "document_type": ["fomc_minutes", "fomc_minutes"],
            "stance_score": [0.1, 0.4],
        }
    )
    out = stage4._add_speaker_relative_features(panel)
    assert out.iloc[1]["delta_stance"] == pytest.approx(0.3)


def test_add_speaker_relative_features_rolling_mean_uses_prior_history_only():
    panel = pd.DataFrame(
        {
            "document_id": ["a", "b", "c"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "speaker": ["Powell", "Powell", "Powell"],
            "document_type": ["speech"] * 3,
            "stance_score": [0.0, 0.2, 0.4],
        }
    )
    out = stage4._add_speaker_relative_features(panel)
    # third row's rolling mean should be mean of [0.0, 0.2] = 0.1, not including 0.4 itself
    assert out.iloc[2]["speaker_rolling_mean_stance"] == pytest.approx(0.1)
    assert out.iloc[2]["stance_vs_speaker_rolling_mean"] == pytest.approx(0.3)
