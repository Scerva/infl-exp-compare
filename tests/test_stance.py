from src.text.stance import score_document_stance, score_sentence_stance


def test_hawkish_phrase_detected():
    ss = score_sentence_stance("The Committee decided to raise the target range for the federal funds rate.")
    assert ss.hawk_hits >= 1
    assert ss.dove_hits == 0
    assert ss.stance == 1.0


def test_dovish_phrase_detected():
    ss = score_sentence_stance("The Committee decided to lower the target range for the federal funds rate.")
    assert ss.dove_hits >= 1
    assert ss.hawk_hits == 0
    assert ss.stance == -1.0


def test_negated_hawkish_phrase_is_discounted():
    negated = score_sentence_stance("It would not be appropriate to raise the target range at this time.")
    plain = score_sentence_stance("The Committee decided to raise the target range for the federal funds rate.")
    assert negated.hawk_hits < plain.hawk_hits


def test_no_phrase_hits_gives_neutral_stance():
    ss = score_sentence_stance("The meeting was held in Washington, D.C.")
    assert ss.hawk_hits == 0
    assert ss.dove_hits == 0
    assert ss.stance == 0.0


def test_mixed_hawk_dove_sentence_nets_out():
    ss = score_sentence_stance(
        "While inflation shows upside risks, the Committee also sees downside risks to the labor market."
    )
    assert ss.hawk_hits >= 1
    assert ss.dove_hits >= 1


def test_forward_looking_sentence_gets_weight_bonus():
    fwd = score_sentence_stance("The Committee anticipates raising the target range further.")
    non_fwd = score_sentence_stance("The Committee raised the target range.")
    assert fwd.is_forward_looking
    assert not non_fwd.is_forward_looking


def test_score_document_stance_all_hawkish():
    sentences = [
        "The Committee decided to raise the target range for the federal funds rate.",
        "Inflation remains elevated and upside risks to inflation persist.",
    ]
    doc, _ = score_document_stance(sentences)
    assert doc.stance_mean_weighted > 0
    assert doc.hawk_score > 0
    assert doc.dove_score == 0
    assert doc.proportion_hawkish == 1.0


def test_score_document_stance_all_dovish():
    sentences = [
        "The Committee decided to lower the target range for the federal funds rate.",
        "Downside risks to employment have increased and the labor market is softening.",
    ]
    doc, _ = score_document_stance(sentences)
    assert doc.stance_mean_weighted < 0
    assert doc.dove_score > 0
    assert doc.hawk_score == 0


def test_score_document_stance_no_relevant_sentences():
    sentences = ["Thank you for having me.", "It is a pleasure to be here."]
    doc, _ = score_document_stance(sentences)
    assert doc.n_relevant_sentences == 0
    assert doc.stance_mean_weighted == 0.0
    assert doc.neutral_score == 1.0


def test_score_document_stance_empty_input():
    doc, sent_stances = score_document_stance([])
    assert doc.n_sentences == 0
    assert sent_stances == []
