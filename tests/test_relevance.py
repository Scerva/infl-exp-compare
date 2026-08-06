from src.text.relevance import score_document, score_sentence


def test_score_sentence_detects_inflation_topic():
    sr = score_sentence("Inflation has eased over the past year but remains elevated.")
    assert "inflation" in sr.topics
    assert sr.is_relevant


def test_score_sentence_detects_inflation_expectations_topic():
    sr = score_sentence("Longer-term inflation expectations have remained well anchored.")
    assert "inflation_expectations" in sr.topics


def test_score_sentence_detects_labor_market_topic():
    sr = score_sentence("The unemployment rate has remained low and job gains have moderated.")
    assert "labor_market" in sr.topics


def test_score_sentence_detects_interest_rates_topic():
    sr = score_sentence("The Committee decided to maintain the target range for the federal funds rate.")
    assert "interest_rates" in sr.topics


def test_score_sentence_can_hit_multiple_topics():
    sr = score_sentence("The Committee raised the federal funds rate to combat rising inflation.")
    assert "monetary_policy" in sr.topics
    assert "interest_rates" in sr.topics
    assert "inflation" in sr.topics


def test_score_sentence_irrelevant_regulatory_text():
    sr = score_sentence("Banks should maintain robust cybersecurity practices and third-party risk management.")
    assert not sr.is_relevant
    assert sr.topics == []


def test_score_document_empty_sentences():
    doc = score_document([])
    assert doc.n_sentences == 0
    assert doc.monetary_policy_relevance_score == 0.0
    assert not doc.is_relevant


def test_score_document_all_relevant():
    sentences = [
        "Inflation remains above the Committee's 2 percent objective.",
        "The labor market has cooled somewhat but remains resilient.",
    ]
    doc = score_document(sentences)
    assert doc.n_relevant_sentences == 2
    assert doc.monetary_policy_relevance_score == 1.0
    assert doc.is_relevant


def test_score_document_mixed_relevance():
    sentences = [
        "Inflation remains above target.",  # relevant
        "Thank you for having me here today.",  # not relevant
        "It is a pleasure to be in this beautiful city.",  # not relevant
        "The unemployment rate ticked up slightly.",  # relevant
    ]
    doc = score_document(sentences)
    assert doc.n_relevant_sentences == 2
    assert doc.monetary_policy_relevance_score == 0.5


def test_score_document_below_threshold_flagged_not_relevant():
    sentences = ["Thank you."] * 9 + ["Inflation is elevated."]
    doc = score_document(sentences)
    assert doc.monetary_policy_relevance_score == 0.1
    assert doc.is_relevant  # exactly at default threshold (0.10)

    sentences_below = ["Thank you."] * 20 + ["Inflation is elevated."]
    doc_below = score_document(sentences_below)
    assert doc_below.monetary_policy_relevance_score < 0.10
    assert not doc_below.is_relevant
