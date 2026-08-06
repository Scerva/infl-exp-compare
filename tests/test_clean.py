from src.text.clean import normalize_text, remove_boilerplate_lines


def test_normalize_text_collapses_whitespace():
    assert normalize_text("Hello   World") == "Hello World"


def test_normalize_text_collapses_blank_lines():
    assert normalize_text("line1\n\n\n\nline2") == "line1\n\nline2"


def test_normalize_text_strips_leading_trailing_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_normalize_text_empty_string():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_normalize_text_is_idempotent():
    text = "Some   text\n\n\nwith   spacing"
    once = normalize_text(text)
    twice = normalize_text(once)
    assert once == twice


def test_remove_boilerplate_lines_drops_short_lines():
    out = remove_boilerplate_lines("real content line\n\nx\nanother real line", min_len=3)
    assert "x" not in out.split("\n")
    assert "real content line" in out
    assert "another real line" in out
