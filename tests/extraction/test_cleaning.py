import pytest

from jobpdf.extraction.cleaning import clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ofﬁce workﬂow", "office workflow"),  # ligatures
        ("develop-\nment", "development"),
        ("Front-\nEnd", "Front-\nEnd"),  # capitalised continuation keeps the hyphen
        ("2019-\n2021", "2019-\n2021"),  # date ranges stay intact
        ("  Python \t SQL  \n\n\n Docker ", "Python SQL\nDocker"),
        (" \n \n", ""),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_soft_hyphen_line_break_is_joined_and_stray_ones_removed() -> None:
    soft = chr(0xAD)
    assert clean_text(f"develop{soft}\nment and co{soft}operation") == "development and cooperation"
