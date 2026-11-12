"""JM-49: remote, hybrid or on site, and where. Only the posting's own words are used."""

import pytest
from pydantic import ValidationError

from jobpdf.ingestion.sources.models import SourcePosting
from jobpdf.ingestion.work_place import (
    HYBRID,
    ON_SITE,
    REMOTE,
    classify_work_mode,
    posting_country,
    posting_text,
)


def post(title="Data Engineer", location=None, remote=None, description=None, tags=(),
         country=None) -> SourcePosting:
    return SourcePosting(source="s", source_id="1", url="https://example.com/1", title=title,
                         location_raw=location, remote=remote, description=description,
                         tags=list(tags), country=country)


# --- the source's own flag ---------------------------------------------------------------------


def test_a_remote_flag_means_remote_whatever_the_place_says() -> None:
    assert classify_work_mode(post(location="Berlin", remote=True)) == REMOTE
    assert classify_work_mode(post(location="Berlin", remote=True,
                                   description="Join our hybrid team")) == REMOTE


def test_a_false_flag_means_on_site_or_hybrid() -> None:
    assert classify_work_mode(post(location="Berlin", remote=False)) == ON_SITE
    assert classify_work_mode(post(remote=False, description="A hybrid role, 2 days at home")) == \
        HYBRID
    assert classify_work_mode(post(remote=False, location="Remote")) == ON_SITE


@pytest.mark.parametrize("text", [
    "This is not remote.", "No remote work.", "Non-remote position", "On-site only",
    "office only", "Remote work is not possible",
])
def test_a_posting_that_says_it_is_not_remote_is_on_site_even_when_flagged_remote(text) -> None:
    assert classify_work_mode(post(remote=True, description=text)) == ON_SITE
    assert classify_work_mode(post(location="London", description=f"Fully remote. {text}")) == \
        ON_SITE


# --- the words, when the source sends no flag --------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"title": "Remote Python Developer"},
    {"location": "Remote"},
    {"location": "Remote - Germany"},
    {"tags": ["Remote", "python"]},
    {"description": "We are a fully remote company."},
    {"description": "100% remote, work from anywhere in the world."},
    {"description": "Work from home, any day."},
    {"description": "A remote-first team."},
    {"description": "This is a remote position."},
    {"description": "You will work remotely."},
    {"description": "We have teams that sometimes remote in."},  # the bare word counts as remote
])
def test_remote_words_make_a_posting_remote(kwargs) -> None:
    assert classify_work_mode(post(location=kwargs.pop("location", "London"), **kwargs)) == REMOTE


@pytest.mark.parametrize("where", ["title", "location", "description", "tags"])
def test_hybrid_is_hybrid_wherever_it_is_written(where) -> None:
    kwargs = {"title": {"title": "Hybrid engineer"}, "location": {"location": "Hybrid, London"},
              "description": {"description": "A hybrid arrangement."},
              "tags": {"tags": ["hybrid"]}}[where]
    kwargs.setdefault("location", "London")

    assert classify_work_mode(post(**kwargs)) == HYBRID


def test_hybrid_that_also_offers_fully_remote_work_is_remote() -> None:
    assert classify_work_mode(post(title="Hybrid or fully remote", location="Berlin")) == REMOTE
    assert classify_work_mode(post(location="Remote / Hybrid")) == REMOTE
    assert classify_work_mode(post(location="Berlin",
                                   description="Hybrid, or work from home if you prefer")) == REMOTE


@pytest.mark.parametrize("text", [
    "This is an on-site role.", "onsite", "On site in Munich", "In-office, five days",
    "office-based", "We offer a relocation package.", "You must be based in Poland.",
    "Must live within 30 km of the office.",
])
def test_on_site_words_make_a_posting_on_site(text) -> None:
    assert classify_work_mode(post(location="Munich", description=text)) == ON_SITE


def test_remote_wins_over_a_passing_on_site_word_but_not_over_not_remote() -> None:
    text = "Fully remote. Occasional on-site meetings."

    assert classify_work_mode(post(location="Munich", description=text)) == REMOTE


def test_a_real_place_and_no_word_about_remote_work_is_taken_as_on_site() -> None:
    assert classify_work_mode(post(location="London", description="Build pipelines.")) == ON_SITE
    assert classify_work_mode(post(location="Warsaw, Poland")) == ON_SITE
    # the source's own country is enough
    assert classify_work_mode(post(location="Anywhere", country="GB")) == ON_SITE


def test_a_posting_that_says_nothing_and_has_no_known_place_is_unknown() -> None:
    assert classify_work_mode(post()) is None
    assert classify_work_mode(post(location="Worldwide", description="Join us.")) is None
    assert classify_work_mode(post(location="Europe")) is None
    assert classify_work_mode(post(location="Georgia")) is None


def test_html_in_the_description_is_read_as_text() -> None:
    html = "<div><p>We are <b>fully</b> <i>remote</i>.</p><script>no remote</script></div>"

    assert "no remote" not in posting_text(post(description=html))
    assert classify_work_mode(post(location="London", description=html)) == REMOTE


def test_only_the_first_part_of_a_huge_description_is_read() -> None:
    text = "x" * 30_000 + " fully remote"

    assert classify_work_mode(post(location="London", description=text)) == ON_SITE


def test_the_same_posting_always_gives_the_same_answer() -> None:
    p = post(location="Berlin", description="Hybrid, 3 days in the office.")

    assert {classify_work_mode(p) for _ in range(5)} == {HYBRID}


# --- the country -------------------------------------------------------------------------------


def test_the_country_the_source_states_wins_over_the_place_text() -> None:
    assert posting_country(post(location="London", country="de")) == "DE"
    assert posting_country(post(location="Berlin, Germany")) == "DE"
    assert posting_country(post(location="Worldwide")) is None
    assert posting_country(post()) is None


@pytest.mark.parametrize(
    "value, stored", [("gb", "GB"), (" pl ", "PL"), ("UA", "UA"), (None, None)])
def test_a_stated_country_is_a_normalised_two_letter_code(value, stored) -> None:
    assert post(country=value).country == stored


@pytest.mark.parametrize("value", ["", "g", "gbr", "G1", "u k", "../"])
def test_a_stated_country_that_is_not_a_code_is_refused(value) -> None:
    with pytest.raises(ValidationError):
        post(country=value)


@pytest.mark.parametrize("text", [
    "This is an on-site role.", "In-office, five days", "We offer a relocation package.",
])
def test_on_site_words_work_without_any_place_named(text) -> None:
    assert classify_work_mode(post(location=None, description=text)) == ON_SITE
    assert posting_country(post(location=None, description=text)) is None


def test_no_place_and_nothing_about_where_it_is_worked_stays_unknown() -> None:
    assert classify_work_mode(post(location=None, description="Build pipelines.")) is None
