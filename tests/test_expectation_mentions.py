from src.text.expectation_mentions import find_mentions_in_sentences, summarize_mentions


def test_finds_michigan_survey_mention():
    sentences = [
        "Inflation has been moderating in recent months.",
        "According to the University of Michigan survey, sentiment held steady.",
        "The Committee will continue to monitor incoming data.",
    ]
    mentions = find_mentions_in_sentences(sentences)
    # This sentence should hit exactly the household/Michigan pattern
    # (kept free of "long-run expectations" wording so it doesn't also
    # trigger the separate horizon_generic pattern -- see the dedicated
    # multi-pattern test below for that overlap case).
    assert len(mentions) == 1
    assert mentions[0].specific_measure == "Michigan Survey"
    assert mentions[0].measure_family == "household"
    assert mentions[0].sentence_index == 1


def test_single_sentence_can_match_multiple_measure_patterns():
    # "Michigan survey" + "long-run expectations" legitimately reference
    # two distinct concepts in the same sentence; both should be captured
    # rather than arbitrarily picking one.
    sentences = ["The Michigan survey shows long-run expectations remain stable."]
    mentions = find_mentions_in_sentences(sentences)
    measures = {m.specific_measure for m in mentions}
    assert "Michigan Survey" in measures
    assert "Longer-Term (generic horizon)" in measures


def test_finds_tips_breakeven_mention():
    sentences = ["TIPS breakeven inflation rates have declined this quarter."]
    mentions = find_mentions_in_sentences(sentences)
    assert any(m.specific_measure == "TIPS / Breakevens" for m in mentions)


def test_finds_spf_mention():
    sentences = ["The Survey of Professional Forecasters shows stable long-run expectations."]
    mentions = find_mentions_in_sentences(sentences)
    families = [m.measure_family for m in mentions]
    assert "professional" in families


def test_no_mentions_returns_empty_list():
    sentences = ["The weather in Washington was pleasant today.", "Thank you for the invitation."]
    mentions = find_mentions_in_sentences(sentences)
    assert mentions == []


def test_context_window_includes_neighboring_sentences_and_excludes_far_ones():
    sentences = [
        "Sentence minus two.",
        "Sentence minus one.",
        "Sentence zero.",
        "Sentence one.",
        "The Michigan survey ticked up.",
        "Sentence three.",
        "Sentence four.",
        "Sentence five.",
        "Sentence six.",
    ]
    mentions = find_mentions_in_sentences(sentences, context_window=2)
    assert len(mentions) == 1
    ctx = mentions[0].context
    assert "Sentence zero." in ctx
    assert "Sentence one." in ctx
    assert "Sentence three." in ctx
    assert "Sentence four." in ctx
    assert "Sentence minus one." not in ctx  # 3 sentences before the mention, outside +/-2
    assert "Sentence five." not in ctx  # 3 sentences after the mention, outside +/-2


def test_context_window_size_one():
    sentences = ["A.", "The SPF reported higher forecasts.", "B.", "C."]
    mentions = find_mentions_in_sentences(sentences, context_window=1)
    assert mentions[0].context == "A. The SPF reported higher forecasts. B."


def test_summarize_mentions_counts_by_family_and_measure():
    sentences = [
        "The Michigan survey rose.",
        "TIPS breakevens fell.",
        "The Michigan survey remains a key indicator.",
    ]
    mentions = find_mentions_in_sentences(sentences)
    summary = summarize_mentions(mentions, n_relevant_sentences=3)
    assert summary.n_mentions == 3
    assert summary.mentions_by_measure["Michigan Survey"] == 2
    assert summary.mentions_by_family["household"] == 2
    assert summary.mentions_by_family["market"] == 1
    assert summary.mention_share_of_relevant == 1.0


def test_summarize_mentions_zero_relevant_sentences_no_division_error():
    summary = summarize_mentions([], n_relevant_sentences=0)
    assert summary.mention_share_of_relevant == 0.0
