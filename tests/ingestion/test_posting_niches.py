"""JM-51: a posting's industries are read from its title, company, tags, category and text."""

from jobpdf.ingestion.posting_niches import posting_niches
from jobpdf.ingestion.vacancies import build_features
from jobpdf.ingestion.work_place import MAX_TEXT_CHARS
from jobpdf.ranking.models import VacancyFeatures
from tests.ingestion.ingest_fakes import FakeEmbedder, FakeExtractor, FakeService, posting


def post(title="Data Engineer", company="Acme", tags=(), category=None, description=None):
    return posting(1, title=title, company=company, tags=list(tags), category=category,
                   description=description)


def test_a_title_company_tag_or_category_word_is_enough() -> None:
    assert posting_niches(post(title="Blockchain Developer")) == ["crypto"]
    assert posting_niches(post(company="Lucky Casino Group")) == ["gambling"]
    assert posting_niches(post(tags=["python", "forex"])) == ["forex"]
    assert posting_niches(post(category="Adult Entertainment")) == ["adult"]


def test_a_plain_posting_has_no_industry() -> None:
    assert posting_niches(post()) == []
    assert posting_niches(post(description="<p>Build things with Python and SQL.</p>")) == []
    assert posting_niches(post(company=None, description=None)) == []


def test_the_description_counts_when_it_names_the_industry_twice() -> None:
    twice = "<p>We run an online casino.</p><ul><li>Our casino has live tables</li></ul>"
    once = "<p>We run an online casino.</p><p>Build things.</p>"

    assert posting_niches(post(description=twice)) == ["gambling"]
    assert posting_niches(post(description=once)) == []


def test_html_markup_and_links_are_not_text() -> None:
    markup = ('<p><a href="https://casino.example/jobs">Apply</a></p>'
              '<img src="https://casino.example/logo.png" alt="">'
              '<div class="casino casino"><span data-x="blockchain blockchain">Hi</span></div>')

    assert posting_niches(post(description=markup)) == []


def test_plain_text_descriptions_work_like_html_ones() -> None:
    assert posting_niches(post(description="Crypto exchange. We list every crypto asset.")) == [
        "crypto"]


def test_only_the_start_of_a_very_long_description_is_read() -> None:
    filler = "<p>Build things with teams.</p>" * (MAX_TEXT_CHARS // 20)
    late = filler + "<p>casino casino</p>"

    assert posting_niches(post(description=late)) == []
    assert posting_niches(post(description="<p>casino casino</p>" + filler)) == ["gambling"]


def test_several_industries_come_back_in_the_tables_order() -> None:
    assert posting_niches(post(title="Crypto Casino Backend")) == ["gambling", "crypto"]


def test_the_answer_for_one_posting_does_not_depend_on_another() -> None:
    first = post(title="Blockchain Developer")
    other = post(title="Data Engineer")

    assert posting_niches(first) == ["crypto"]
    assert posting_niches(other) == []
    assert posting_niches(first) == ["crypto"]


# --- the stored vacancy ------------------------------------------------------------------------


def build(**changes) -> VacancyFeatures:
    features, _ = build_features(post(**changes), extractor=FakeExtractor(),
                                 normalizer=FakeService(), embedder=FakeEmbedder())
    return features


def test_the_vacancy_is_stored_with_its_industries() -> None:
    assert build(title="Crypto Casino Backend").niches == ["gambling", "crypto"]
    assert build().niches == []


def test_the_industries_survive_the_json_round_trip_the_store_uses() -> None:
    features = build(title="Blockchain Developer")

    again = VacancyFeatures.model_validate_json(features.model_dump_json())

    assert again.niches == ["crypto"]


def test_a_vacancy_stored_before_the_field_existed_loads_with_none() -> None:
    old = build().model_dump(mode="json")
    del old["niches"]

    assert VacancyFeatures.model_validate(old).niches == []
