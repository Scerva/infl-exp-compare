import numpy as np
import pandas as pd
import pytest

from src.text.annotation import (
    ANNOTATION_COLUMNS,
    _predicted_label,
    build_annotation_sample,
    build_sentence_pool,
    score_annotations,
)


def _fake_documents_df():
    return pd.DataFrame(
        [
            {
                "document_id": "doc1",
                "document_type": "speech",
                "event_date": pd.Timestamp("2015-03-01"),
                "speaker": "Speaker A",
                "clean_text": (
                    "The Committee decided to raise the target range for the federal funds rate. "
                    "Inflation remains elevated and upside risks to inflation persist. "
                    "Thank you for having me today."
                ),
            },
            {
                "document_id": "doc2",
                "document_type": "fomc_statement",
                "event_date": pd.Timestamp("2023-06-01"),
                "speaker": None,
                "clean_text": (
                    "The Committee decided to lower the target range for the federal funds rate. "
                    "Downside risks to employment have increased."
                ),
            },
        ]
    )


def test_predicted_label_mapping():
    assert _predicted_label(0.5) == "HAWKISH"
    assert _predicted_label(-0.5) == "DOVISH"
    assert _predicted_label(0.0) == "NEUTRAL"


def test_build_sentence_pool_only_includes_relevant_sentences():
    pool = build_sentence_pool(_fake_documents_df())
    # "Thank you for having me today." has no topic hits -> excluded
    assert not pool["sentence"].str.contains("Thank you").any()
    assert len(pool) >= 3
    assert "decade" in pool.columns
    assert (pool.loc[pool["document_id"] == "doc1", "decade"] == "2010s").all()


def test_build_sentence_pool_empty_text_skipped():
    df = pd.DataFrame([{"document_id": "d", "document_type": "speech", "event_date": pd.Timestamp("2020-01-01"), "speaker": "X", "clean_text": ""}])
    pool = build_sentence_pool(df)
    assert pool.empty


def test_build_annotation_sample_returns_expected_columns():
    sample = build_annotation_sample(_fake_documents_df(), n_target=10, seed=1)
    assert list(sample.columns) == ANNOTATION_COLUMNS
    assert (sample["human_label"] == "").all()
    assert (sample["human_confidence"] == "").all()
    assert len(sample) > 0


def test_build_annotation_sample_deterministic_with_seed():
    s1 = build_annotation_sample(_fake_documents_df(), n_target=5, seed=7)
    s2 = build_annotation_sample(_fake_documents_df(), n_target=5, seed=7)
    assert list(s1["sentence_id"]) == list(s2["sentence_id"])


def test_score_annotations_perfect_agreement():
    df = pd.DataFrame(
        {
            "model_predicted_label": ["HAWKISH", "DOVISH", "NEUTRAL", "HAWKISH"],
            "model_stance_score": [0.8, -0.6, 0.0, 0.5],
            "human_label": ["HAWKISH", "DOVISH", "NEUTRAL", "HAWKISH"],
        }
    )
    report = score_annotations(df)
    assert report.n_labeled == 4
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0


def test_score_annotations_ignores_blank_human_labels():
    df = pd.DataFrame(
        {
            "model_predicted_label": ["HAWKISH", "DOVISH", "NEUTRAL"],
            "model_stance_score": [0.8, -0.6, 0.0],
            "human_label": ["HAWKISH", "", "NEUTRAL"],
        }
    )
    report = score_annotations(df)
    assert report.n_labeled == 2


def test_score_annotations_raises_if_nothing_labeled():
    df = pd.DataFrame(
        {
            "model_predicted_label": ["HAWKISH"],
            "model_stance_score": [0.5],
            "human_label": [""],
        }
    )
    with pytest.raises(ValueError):
        score_annotations(df)


def test_score_annotations_confusion_matrix_shape():
    df = pd.DataFrame(
        {
            "model_predicted_label": ["HAWKISH", "DOVISH", "NEUTRAL", "DOVISH"],
            "model_stance_score": [0.8, -0.6, 0.0, -0.3],
            "human_label": ["HAWKISH", "NEUTRAL", "NEUTRAL", "DOVISH"],
        }
    )
    report = score_annotations(df)
    assert report.confusion.shape == (3, 3)
