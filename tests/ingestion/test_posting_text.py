from jobpdf.ingestion.posting_text import (
    MAX_INPUT_CHARS,
    Line,
    is_usable,
    posting_lines,
)


def test_html_becomes_one_line_per_block_with_headings_and_bullets_marked() -> None:
    html = ("<p>We build tools.</p><h3>Requirements</h3><ul><li>Python &amp; SQL</li>"
            "<li>Docker</li></ul><p><strong>Nice to have</strong></p><p>Go<br>Rust</p>")
    assert posting_lines(html) == [
        Line("We build tools."),
        Line("Requirements", is_heading=True),
        Line("Python & SQL", is_bullet=True),
        Line("Docker", is_bullet=True),
        Line("Nice to have", is_heading=True),
        Line("Go", is_heading=False),
        Line("Rust"),
    ]


def test_a_bold_word_inside_a_sentence_is_not_a_heading() -> None:
    (line,) = posting_lines("<p>You know <b>Python</b> well.</p>")
    assert line == Line("You know Python well.")


def test_scripts_and_styles_are_dropped() -> None:
    html = "<style>p {color: red}</style><script>alert(1)</script><p>Text</p>"
    assert posting_lines(html) == [Line("Text")]


def test_plain_text_bullets_and_colon_headings_are_recognised() -> None:
    text = "About the job\nRequirements:\n- 3+ years Python\n• SQL\n2) Docker\n\n"
    assert posting_lines(text) == [
        Line("About the job", is_title=True),
        Line("Requirements:", is_heading=True),
        Line("3+ years Python", is_bullet=True),
        Line("SQL", is_bullet=True),
        Line("Docker", is_bullet=True),
    ]


def test_a_long_line_ending_in_a_colon_is_not_a_heading() -> None:
    text = "In this role you will work closely with many teams across the company on:"
    assert not posting_lines(text)[0].is_heading


def test_empty_descriptions_give_no_lines() -> None:
    assert posting_lines(None) == []
    assert posting_lines("   ") == []
    assert posting_lines("<p> </p>") == []


def test_very_long_descriptions_are_cut_before_parsing() -> None:
    lines = posting_lines("word " * MAX_INPUT_CHARS)
    assert sum(len(line.text) for line in lines) <= MAX_INPUT_CHARS


def test_short_text_is_not_usable() -> None:
    assert not is_usable(posting_lines("Python developer wanted."))
    assert is_usable(posting_lines("x" * 200))


def test_title_shaped_lines_are_marked_but_not_headings() -> None:
    lines = posting_lines("Your Future at Kyndryl\nTailwind CSS\nWe hire people.")
    assert [(line.is_heading, line.is_title) for line in lines] == [
        (False, True), (False, False), (False, False)]
