import os
from unittest.mock import MagicMock

import pytest

from src.text.stance_finbert import score_sentence_finbert, score_sentences_finbert


def _fake_result(pos=0.6, neg=0.1, neu=0.3):
    return [
        {"label": "positive", "score": pos},
        {"label": "negative", "score": neg},
        {"label": "neutral", "score": neu},
    ]


def test_score_sentence_finbert_computes_tone():
    fake_pipe = MagicMock(return_value=_fake_result(0.6, 0.1, 0.3))
    result = score_sentence_finbert("some sentence", pipe=fake_pipe)
    assert result["p_positive"] == 0.6
    assert result["p_negative"] == 0.1
    assert result["finbert_tone"] == pytest.approx(0.5)


def test_score_sentence_finbert_handles_nested_list_result():
    fake_pipe = MagicMock(return_value=[_fake_result(0.2, 0.7, 0.1)])
    result = score_sentence_finbert("some sentence", pipe=fake_pipe)
    assert result["finbert_tone"] == pytest.approx(-0.5)


def test_score_sentences_finbert_batches(monkeypatch):
    import src.text.stance_finbert as mod

    def fake_pipe(batch):
        return [_fake_result(0.5, 0.2, 0.3) for _ in batch]

    monkeypatch.setattr(mod, "_load_pipeline", lambda model_name: fake_pipe)
    out = score_sentences_finbert(["a", "b", "c"], batch_size=2)
    assert len(out) == 3
    assert all(r["finbert_tone"] == pytest.approx(0.3) for r in out)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SKIP_NETWORK_TESTS"), reason="network tests disabled")
def test_score_sentence_finbert_live_smoke():
    result = score_sentence_finbert("Profits declined sharply amid rising costs.")
    assert result["p_negative"] > result["p_positive"]
