import random
import unicodedata

import pytest

from jobpdf.extraction.grounding import (
    _CHAR_MAP,
    FUZZY_MIN_SCORE,
    MIN_FUZZY_QUOTE_LEN,
    WARNING_AMBIGUOUS,
    WARNING_FUZZY,
    WARNING_LENGTH_MISMATCH,
    WARNING_UNGROUNDED,
    GroundingResult,
    GroundingStatus,
    ground,
    normalize_with_map,
)
from jobpdf.extraction.models import ParsedDocument, RawBlock, Section
from jobpdf.extraction.parser import _assemble
from jobpdf.extraction.prompt import MASK_CHAR, mask_pii
from jobpdf.extraction.schema import CandidateProfile, ExtractedJob, Skill

S = GroundingStatus


def make_doc(*texts: str) -> ParsedDocument:
    raw = [RawBlock(text=t, page=0, bbox=None, font_size=None, is_bold=False) for t in texts]
    return _assemble("pdf", raw, True, [])


def section(doc: ParsedDocument, types: list[str], *block_indexes: int) -> Section:
    blocks = [doc.blocks[i] for i in block_indexes]
    return Section(
        types=types, heading=None, blocks=blocks, content_start=blocks[0].start,
        end=blocks[-1].end, confidence="high", label_source="keyword",
    )


def assert_invariants(result: GroundingResult, doc: ParsedDocument) -> None:
    """Checked after every grounding in this file."""
    full_text = doc.full_text
    for g in result.groundings:
        if g.status is S.UNGROUNDED:
            assert (g.start, g.end, g.matched_text, g.score) == (None, None, None, None)
            assert g.low_confidence
            continue
        assert 0 <= g.start < g.end <= len(full_text)
        assert full_text[g.start : g.end] == g.matched_text
        assert g.low_confidence is (g.status is S.FUZZY)
        assert (g.score is not None) is (g.status is S.FUZZY)
        if g.status is S.EXACT and MASK_CHAR not in g.quote:
            assert g.matched_text in g.quote
    assert sum(result.counts.values()) == len(result.groundings)
    assert set(result.counts) == set(S)


def run(extraction, doc: ParsedDocument, sections=(), searched_text=None) -> GroundingResult:
    result = ground(extraction, doc, list(sections), searched_text)
    assert_invariants(result, doc)
    return result


def skill(quote: str, found_in: str = "skills_list") -> dict:
    return {"name": "x", "kind": "hard", "found_in": found_in, "quote": quote}


# A small synthetic CV: headings are their own blocks, as JM-10 sees them.
CV = make_doc(
    "SUMMARY",
    "Data engineer with a Computer Science degree.",
    "EXPERIENCE",
    "Data Engineer, Acme Corp\nBuilt pipelines in Python and SQL.",
    "SKILLS",
    "Python, Docker, Kubernetes",
    "EDUCATION",
    "BSc Computer Science, Kyiv University",
)
CV_SECTIONS = [
    section(CV, ["summary"], 1),
    section(CV, ["experience"], 3),
    section(CV, ["skills"], 5),
    section(CV, ["education"], 7),
]


def test_exact_unique_quote() -> None:
    result = run({"experience": [{"quote": "Data Engineer, Acme Corp"}]}, CV, CV_SECTIONS)

    g = result.groundings[0]
    assert (g.path, g.status, g.ambiguous) == ("experience[0]", S.EXACT, False)
    assert g.start == CV.full_text.index("Data Engineer, Acme Corp")
    assert result.warnings == []


def test_repeated_short_quote_is_resolved_by_found_in() -> None:
    extraction = {"skills": [skill("Python", "experience"), skill("Python", "skills_list")]}

    in_experience, in_skills = run(extraction, CV, CV_SECTIONS).groundings

    assert in_experience.start == CV.blocks[3].text.index("Python") + CV.blocks[3].start
    assert in_skills.start == CV.blocks[5].start
    assert not in_experience.ambiguous and not in_skills.ambiguous


def test_repeated_quote_is_resolved_by_the_list_it_sits_in() -> None:
    result = run({"education": [{"quote": "Computer Science"}]}, CV, CV_SECTIONS)

    g = result.groundings[0]
    assert CV.blocks[7].start <= g.start < CV.blocks[7].end
    assert not g.ambiguous


def test_unresolvable_repeat_takes_the_first_hit_and_is_ambiguous() -> None:
    result = run({"skills": [skill("Python", "other")]}, CV, CV_SECTIONS)

    g = result.groundings[0]
    assert g.status is S.EXACT and g.ambiguous
    assert g.start == CV.full_text.index("Python")
    assert [w.model_dump() for w in result.warnings] == [
        {"code": WARNING_AMBIGUOUS, "path": "skills[0]"}
    ]


@pytest.mark.parametrize(
    ("cv_text", "quote"),
    [
        ("Senior Data\nEngineer at Acme", "Senior Data Engineer at Acme"),
        ("Led software develop-\nment for payments", "Led software development for payments"),
        ("Front-\nEnd developer at Acme", "Front-End developer at Acme"),
        ("Built eﬃcient ﬁle stores", "Built efficient file stores"),
        ("Led the “Cloud First” programme", 'Led the "Cloud First" programme'),
        ("Acme, 2019–2021", "Acme, 2019-2021"),
        ("Acme — Data Engineer", "Acme - Data Engineer"),
        ("Senior Engineer at Acme", "Senior Engineer at Acme"),
        ("Led devel­opment of the API", "Led development of the API"),
    ],
    ids=["line_break", "hyphenation", "kept_hyphen", "ligature", "curly_quotes", "en_dash",
         "em_dash", "nbsp", "soft_hyphen"],
)
def test_normalized_match_for_each_quirk(cv_text: str, quote: str) -> None:
    doc = make_doc("EXPERIENCE", cv_text)

    result = run({"experience": [{"quote": quote}]}, doc)

    g = result.groundings[0]
    assert g.status is S.NORMALIZED
    assert g.matched_text == cv_text
    assert result.warnings == []


@pytest.mark.parametrize(
    "quote",
    [
        '<section types="experience" confidence="high">\nData Engineer, Acme Corp',
        "Data Engineer, Acme Corp\n</section>",
    ],
)
def test_section_marker_tags_are_stripped_from_quotes(quote: str) -> None:
    result = run({"experience": [{"quote": quote}]}, CV, CV_SECTIONS)

    g = result.groundings[0]
    assert g.status is S.EXACT
    assert g.matched_text == "Data Engineer, Acme Corp"


def test_quote_that_is_only_a_marker_tag_is_ungrounded() -> None:
    result = run({"skills": [skill("</section>")]}, CV, CV_SECTIONS)

    assert result.groundings[0].status is S.UNGROUNDED


def test_masked_quote_matches_the_masked_text_with_offsets_into_full_text() -> None:
    doc = make_doc("REFERENCES", "Contact jane.doe@example.com for references")
    masked, _ = mask_pii(doc.full_text)
    quote = masked[doc.blocks[1].start : doc.blocks[1].end]
    assert MASK_CHAR in quote

    result = run({"certifications": [{"quote": quote}]}, doc, searched_text=masked)

    g = result.groundings[0]
    assert g.status is S.EXACT
    assert (g.start, g.end) == (doc.blocks[1].start, doc.blocks[1].end)
    assert g.matched_text == doc.blocks[1].text  # the CV's own text, not the mask


def test_searched_text_of_the_wrong_length_falls_back_to_full_text() -> None:
    result = run(
        {"experience": [{"quote": "Data Engineer, Acme Corp"}]}, CV, CV_SECTIONS,
        searched_text=CV.full_text + " extra",
    )

    assert result.groundings[0].status is S.EXACT
    assert [w.model_dump() for w in result.warnings] == [
        {"code": WARNING_LENGTH_MISMATCH, "path": None}
    ]


def test_llm_corrected_typo_matches_fuzzily_with_its_score() -> None:
    doc = make_doc("EXPERIENCE", "Deployed services to Kubernetis clusters on AWS")
    sections = [section(doc, ["experience"], 1)]

    result = run(
        {"skills": [skill("Deployed services to Kubernetes clusters on AWS", "experience")]},
        doc, sections,
    )

    g = result.groundings[0]
    assert g.status is S.FUZZY and g.low_confidence
    assert FUZZY_MIN_SCORE <= g.score < 100
    assert "Kubernetis" in g.matched_text
    assert [w.code for w in result.warnings] == [WARNING_FUZZY]


def test_fuzzy_prefers_the_expected_section() -> None:
    doc = make_doc(
        "SUMMARY", "Ran Kubernetis clusters for the payments team",
        "EXPERIENCE", "Ran Kubernetis clusters for the payments team",
    )
    sections = [section(doc, ["summary"], 1), section(doc, ["experience"], 3)]

    result = run(
        {"experience": [{"quote": "Ran Kubernetes clusters for the payments team"}]},
        doc, sections,
    )

    g = result.groundings[0]
    assert g.status is S.FUZZY
    assert g.start == doc.blocks[3].start


def test_short_typo_quote_is_never_matched_fuzzily() -> None:
    assert len("Kubernetis") < MIN_FUZZY_QUOTE_LEN
    doc = make_doc("SKILLS", "Python, Docker, Kubernetes")

    result = run({"skills": [skill("Kubernetis")]}, doc)

    assert result.groundings[0].status is S.UNGROUNDED


def test_invented_quote_is_kept_ungrounded_with_a_warning_without_its_text() -> None:
    extraction = {"experience": [
        {"quote": "Data Engineer, Acme Corp"},
        {"quote": "Head of Platform at Globex Corporation"},
    ]}

    result = run(extraction, CV, CV_SECTIONS)

    kept = result.groundings[1]
    assert (kept.path, kept.status, kept.quote) == (
        "experience[1]", S.UNGROUNDED, "Head of Platform at Globex Corporation"
    )
    assert result.counts[S.UNGROUNDED] == 1 and result.counts[S.EXACT] == 1
    assert [w.model_dump() for w in result.warnings] == [
        {"code": WARNING_UNGROUNDED, "path": "experience[1]"}
    ]
    assert "Globex" not in str([w.model_dump() for w in result.warnings])


def test_candidate_profile_skill_quotes_get_their_own_paths() -> None:
    profile = CandidateProfile(
        summary=None,
        skills=[Skill(name="Python", kind="hard", found_in=["experience", "skills_list"],
                      quotes=["Python and SQL", "Python, Docker"])],
        experience=[ExtractedJob(
            title="Data Engineer", company="Acme Corp", location=None, start="2021",
            end="present", description=None, skills_used=["Python"],
            quote="Data Engineer, Acme Corp",
        )],
        education=[], languages=[], certifications=[], total_experience_months=None,
    )

    result = run(profile, CV, CV_SECTIONS)

    assert [g.path for g in result.groundings] == [
        "skills[0].quotes[0]", "skills[0].quotes[1]", "experience[0]"
    ]
    assert result.counts == {S.EXACT: 3, S.NORMALIZED: 0, S.FUZZY: 0, S.UNGROUNDED: 0}


def test_no_quotes_and_empty_document_do_not_crash() -> None:
    doc = make_doc()

    result = run({"skills": [skill("Python")], "summary": None}, doc)

    assert result.groundings[0].status is S.UNGROUNDED
    assert run({"languages": [{"language": "English"}]}, doc).groundings == []


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
