import random
import unicodedata

import pytest

from jobpdf.extraction.grounding import _CHAR_MAP, normalize_with_map


def assert_map_points_at_source(original: str, normalized: str, index_map: list[int]) -> None:
    assert len(index_map) == len(normalized)
    assert index_map == sorted(index_map)
    for i, ch in enumerate(normalized):
        at = index_map[i]
        end = at + 1
        while end < len(original) and unicodedata.combining(original[end]):
            end += 1
        source = unicodedata.normalize("NFKC", original[at:end])
        folded = "".join(_CHAR_MAP.get(c, c) for c in source)
        assert (ch == " " and folded.isspace()) or ch in folded, (i, ch, source)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Data\nEngineer", "Data Engineer"),
        ("Data \n\t  Engineer", "Data Engineer"),
        ("software develop-\nment", "software development"),
        ("Front-\nEnd developer", "Front-End developer"),
        ("2019-\n2021", "2019-2021"),
        ("2019 -\n2021", "2019 - 2021"),
        ("eﬃcient ﬁles", "efficient files"),
        ("“quoted” ‘single’ «guillemets»", "\"quoted\" 'single' \"guillemets\""),
        ("2019–2021 — now", "2019-2021 - now"),
        ("Senior Engineer", "Senior Engineer"),
        ("devel­opment", "development"),
        ("zero​width", "zerowidth"),
        ("Андрій", "Андрій"),
    ],
)
def test_normalize_steps(text: str, expected: str) -> None:
    normalized, index_map = normalize_with_map(text)

    assert normalized == expected
    assert_map_points_at_source(text, normalized, index_map)


def test_expanded_ligature_maps_every_char_to_the_ligature() -> None:
    normalized, index_map = normalize_with_map("aﬁb")

    assert normalized == "afib"
    assert index_map == [0, 1, 1, 2]


def test_empty_text() -> None:
    assert normalize_with_map("") == ("", [])


def test_randomized_breaks_map_back_to_the_right_characters() -> None:
    rng = random.Random(12345)
    separators = [" ", "  ", "\t", "\n", " \n  ", " ", "\r\n"]
    breaks = ["-\n", "-\n  ", "- \n", "-\r\n"]
    for _ in range(200):
        words = [
            "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 9)))
            for _ in range(rng.randint(1, 12))
        ]
        expected = " ".join(words)
        pieces: list[str] = []
        for w, word in enumerate(words):
            if w:
                pieces.append(rng.choice(separators))
            if len(word) > 3 and rng.random() < 0.5:
                cut = rng.randint(1, len(word) - 1)
                filler = rng.choice(breaks + ["­"])
                word = word[:cut] + filler + word[cut:]
            pieces.append(word)
        original = "".join(pieces)

        normalized, index_map = normalize_with_map(original)

        assert normalized == expected, repr(original)
        assert_map_points_at_source(original, normalized, index_map)
        for i, ch in enumerate(normalized):
            if ch != " ":
                assert original[index_map[i]] == ch
