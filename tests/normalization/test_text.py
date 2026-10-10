import pytest

from jobpdf.normalization.text import normalize


def test_case_and_surrounding_space_do_not_matter() -> None:
    assert normalize("  JavaScript ") == normalize("javascript") == "javascript"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Machine\tLearning\n", "machine learning"),
        ("data   engineering", "data engineering"),
        ("C++", "c++"),
        ("C#", "c#"),
        (".NET", ".net"),
        ("Node.js", "node.js"),
        ("ＰＹＴＨＯＮ", "python"),  # full-width characters (NFKC)
        ("Python (computer programming)", "python (computer programming)"),
        ("Програмування", "програмування"),
        ("", ""),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected
