from src.text.sentence_split import split_paragraphs_to_sentences, split_sentences


def test_split_on_simple_sentence_boundary():
    out = split_sentences("The unemployment rate fell. Inflation rose.")
    assert out == ["The unemployment rate fell.", "Inflation rose."]


def test_does_not_split_on_decimal_numbers():
    out = split_sentences("The unemployment rate fell to 3.5 percent in May. Core PCE rose 2.1 percent.")
    assert len(out) == 2
    assert "3.5 percent in May." in out[0]
    assert "2.1 percent." in out[1]


def test_does_not_split_on_title_abbreviations():
    out = split_sentences("Mr. Powell spoke today. He was measured in tone.")
    assert out == ["Mr. Powell spoke today.", "He was measured in tone."]


def test_does_not_split_on_us_abbreviation():
    out = split_sentences("The U.S. economy grew 2.5 percent. The E.U. lagged.")
    assert len(out) == 2
    assert out[0] == "The U.S. economy grew 2.5 percent."
    assert out[1] == "The E.U. lagged."


def test_does_not_split_on_gov_abbreviation():
    out = split_sentences("Gov. Waller dissented. He preferred a smaller cut.")
    assert out == ["Gov. Waller dissented.", "He preferred a smaller cut."]


def test_single_sentence_no_trailing_split():
    assert split_sentences("Thank you.") == ["Thank you."]


def test_empty_input_returns_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_paragraphs_to_sentences_indexes_correctly():
    paragraphs = ["First sentence. Second sentence.", "Only one here."]
    out = split_paragraphs_to_sentences(paragraphs)
    assert out == [
        (0, 0, "First sentence."),
        (0, 1, "Second sentence."),
        (1, 0, "Only one here."),
    ]
