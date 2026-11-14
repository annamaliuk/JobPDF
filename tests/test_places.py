"""JM-49: the country word list. It must say nothing rather than guess."""

import pytest

from jobpdf import places
from jobpdf.places import COUNTRY_NAMES, country_name, is_country, resolve_country


@pytest.mark.parametrize("text, code", [
    ("Berlin, Germany", "DE"),
    ("Germany", "DE"),
    ("germany", "DE"),
    ("  Berlin ,  Germany  ", "DE"),
    ("Deutschland", "DE"),
    ("Köln", "DE"),
    ("Koln", "DE"),
    ("London", "GB"),
    ("London, UK", "GB"),
    ("Edinburgh, Scotland", "GB"),
    ("Wales", "GB"),
    ("Austin, TX", "US"),
    ("New York, NY", "US"),
    ("San Francisco, CA", "US"),
    ("Atlanta, Georgia", "US"),
    ("Texas", "US"),
    ("USA", "US"),
    ("U.S.", "US"),
    ("United States of America", "US"),
    ("Toronto, ON", "CA"),
    ("Vancouver, British Columbia", "CA"),
    ("Kraków, Poland", "PL"),
    ("Warsaw", "PL"),
    ("Київ, Україна", "UA"),
    ("Львів", "UA"),
    ("Україна", "UA"),
    ("Remote - Germany", "DE"),
    ("Remote (US)", "US"),
    ("Warsaw / Berlin", None),  # two countries
    ("Dublin, Ireland", "IE"),
    ("Tbilisi", "GE"),
    ("Berlin, DE", "DE"),  # "DE" alone is also Delaware, but the city says it
])
def test_a_place_that_names_one_country_resolves_to_it(text, code) -> None:
    assert resolve_country(text) == code


@pytest.mark.parametrize("text", [
    None, "", "   ", "Worldwide", "Remote", "Anywhere", "Europe", "EMEA", "Global",
    "Georgia",  # a country and a US state
    "CA", "DE", "IN", "IL",  # a US state and a country code
    "Springfield, IL",
    "London, Ontario",  # England and Canada
    "Paris",  # not in the list on purpose
    "Birmingham", "Cambridge",
    "USA Only",
    "xx", "12345", "Berlin Heights",
    "ca", "de",  # a code only counts when it is written as one
])
def test_a_place_that_names_no_country_or_is_unsure_resolves_to_nothing(text) -> None:
    assert resolve_country(text) is None


def test_every_country_name_resolves_to_its_own_code_except_the_ambiguous_one() -> None:
    wrong = {code: resolve_country(name) for code, name in COUNTRY_NAMES.items()
             if resolve_country(name) != code}

    assert wrong == {"GE": None}  # "Georgia" is also a US state


def test_every_city_names_a_country_that_exists_and_is_found_by_its_own_name() -> None:
    assert places._CITY_WORDS
    for city, code in places._CITY_WORDS.items():
        assert code in COUNTRY_NAMES, city
    assert resolve_country("Madrid") == "ES" and resolve_country("Lisbon") == "PT"


def test_codes_are_two_capital_letters_and_names_are_unique_and_plain() -> None:
    assert all(len(c) == 2 and c.isupper() for c in COUNTRY_NAMES)
    assert len(set(COUNTRY_NAMES.values())) == len(COUNTRY_NAMES)
    assert COUNTRY_NAMES["GE"] == "Georgia" and "(country)" not in " ".join(COUNTRY_NAMES.values())
    assert COUNTRY_NAMES["UA"] == "Ukraine" and COUNTRY_NAMES["GB"] == "United Kingdom"


def test_country_names_and_codes_are_looked_up_without_regard_to_case() -> None:
    assert country_name("ua") == "Ukraine" and country_name("UA") == "Ukraine"
    assert country_name("XX") is None and country_name(None) is None and country_name("") is None
    assert is_country("pl") and not is_country("zz") and not is_country(None)


def test_the_two_letter_state_and_province_codes_are_the_real_ones() -> None:
    assert len(places._STATE_CODES) == 51  # 50 states and DC
    assert len(places._PROVINCE_CODES) == 10
    assert places._STATE_CODES.isdisjoint(places._PROVINCE_CODES)


def test_a_long_text_is_handled_quickly() -> None:
    assert resolve_country("Berlin, " * 5000 + "Germany") == "DE"


@pytest.mark.parametrize("text", ["us", "it", "in", "me", "de", "pl", "ua", "be", "no", "is"])
def test_ordinary_two_letter_words_are_not_country_codes(text) -> None:
    assert resolve_country(text) is None
    assert resolve_country(f"Sofa, {text}") is None


@pytest.mark.parametrize("text, code", [
    ("TX", "US"), ("Plano, TX", "US"), ("BC", "CA"), ("Smalltown, BC", "CA"), ("NY", "US"),
    ("UA", "UA"), ("PL", "PL"),
])
def test_a_capital_code_names_its_state_province_or_country(text, code) -> None:
    assert resolve_country(text) == code


def test_a_code_that_is_both_a_state_and_a_country_is_left_unsaid() -> None:
    assert resolve_country("Springfield, MA") is None  # Massachusetts or Morocco
    assert resolve_country("Boston, MA") == "US"  # the city decides when the code cannot


# --- JM-47: the places a remote job is open to ---------------------------------------------------


@pytest.mark.parametrize("text, codes", [
    ("USA", {"US"}),
    ("USA Only", {"US"}),
    ("US-based", {"US"}),
    ("USA, Canada", {"US", "CA"}),
    ("Remote - Germany", {"DE"}),
    ("Poland/Ukraine", {"PL", "UA"}),
    ("Austin, TX", {"US"}),
    ("North America", {"US", "CA", "MX"}),
])
def test_a_place_text_gives_the_countries_it_is_open_to(text, codes) -> None:
    assert places.allowed_countries(text) == codes


def test_europe_includes_ukraine_and_the_eu_does_not() -> None:
    assert "UA" in places.allowed_countries("Europe")
    assert "UA" not in places.allowed_countries("EU only")
    assert "UA" not in places.allowed_countries("EEA")
    assert "PL" in places.allowed_countries("EU only")


@pytest.mark.parametrize("text", [
    None, "", "  ", "Remote", "Worldwide", "Anywhere", "USA, Worldwide",
    "CET +/- 3 HOURS", "UTC-5 to UTC+1", "CIS", "USA, Mars", "Georgia",
])
def test_an_open_or_unsure_place_text_limits_nothing(text) -> None:
    assert places.allowed_countries(text) is None


def test_every_region_is_made_of_countries_the_form_offers() -> None:
    for codes in places._REGION_SETS.values():
        assert codes <= set(COUNTRY_NAMES)
