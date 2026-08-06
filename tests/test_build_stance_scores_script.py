import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "stage3", Path(__file__).resolve().parents[1] / "scripts" / "03_build_stance_scores.py"
)
stage3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage3)


def test_process_document_hawkish_text():
    text = (
        "The Committee decided to raise the target range for the federal funds rate. "
        "Inflation remains elevated and upside risks to inflation persist. "
        "The Michigan survey showed longer-term inflation expectations ticked up."
    )
    scores, sentences, doc_rel = stage3._process_document("test1", "speech", text)
    assert scores["stance_score"] > 0
    assert scores["hawk_score"] > 0
    assert scores["monetary_policy_relevance"] > 0
    assert scores["inflation_relevance"] > 0
    assert len(sentences) == 3


def test_process_document_empty_text():
    scores, sentences, doc_rel = stage3._process_document("test2", "speech", "")
    assert sentences == []
    assert scores["monetary_policy_relevance"] == 0.0
    assert scores["stance_score"] == 0.0


def test_process_document_none_text():
    scores, sentences, doc_rel = stage3._process_document("test3", "speech", None)
    assert sentences == []
    assert scores["n_sentences"] == 0
