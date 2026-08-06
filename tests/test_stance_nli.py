import os
from unittest.mock import MagicMock

import pytest

from src.text.stance_nli import HYPOTHESES, score_sentence_nli, score_sentences_nli


def _fake_pipe_result(hawk=0.6, dove=0.3, neutral=0.1):
    return {
        "labels": [HYPOTHESES["hawkish"], HYPOTHESES["dovish"], HYPOTHESES["neutral"]],
        "scores": [hawk, dove, neutral],
    }


def test_score_sentence_nli_maps_labels_to_probabilities():
    fake_pipe = MagicMock(return_value=_fake_pipe_result(0.6, 0.3, 0.1))
    result = score_sentence_nli("some sentence", pipe=fake_pipe)
    assert result["p_hawkish"] == 0.6
    assert result["p_dovish"] == 0.3
    assert result["p_neutral"] == 0.1
    assert result["stance"] == pytest.approx(0.3)


def test_score_sentence_nli_dovish_dominant():
    fake_pipe = MagicMock(return_value=_fake_pipe_result(0.1, 0.7, 0.2))
    result = score_sentence_nli("some sentence", pipe=fake_pipe)
    assert result["stance"] == pytest.approx(-0.6)


def test_score_sentences_nli_batches_and_preserves_order(monkeypatch):
    import src.text.stance_nli as mod

    results_by_batch = [
        [_fake_pipe_result(0.8, 0.1, 0.1), _fake_pipe_result(0.1, 0.8, 0.1)],
        [_fake_pipe_result(0.2, 0.2, 0.6)],
    ]
    call_count = {"n": 0}

    def fake_pipe(batch, candidate_labels, multi_label):
        r = results_by_batch[call_count["n"]]
        call_count["n"] += 1
        return r

    monkeypatch.setattr(mod, "_load_pipeline", lambda model_name: fake_pipe)
    out = score_sentences_nli(["a", "b", "c"], batch_size=2)
    assert len(out) == 3
    assert out[0]["stance"] == pytest.approx(0.7)
    assert out[1]["stance"] == pytest.approx(-0.7)
    assert out[2]["stance"] == pytest.approx(0.0)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_score_sentence_nli_live_smoke():
    hawkish = score_sentence_nli("The Committee decided to raise the target range for the federal funds rate.")
    dovish = score_sentence_nli("The Committee decided to lower the target range for the federal funds rate.")
    assert hawkish["stance"] > dovish["stance"]
