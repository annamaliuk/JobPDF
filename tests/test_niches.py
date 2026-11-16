"""JM-51: which risky industries a posting is about. Built to say nothing when unsure."""

import re

import pytest

from jobpdf import niches
from jobpdf.niches import (
    CODES,
    MIN_MENTIONS,
    NICHES,
    chosen,
    detect_niches,
    field_name,
    known,
    label,
    short_name,
)

# --- the table ---------------------------------------------------------------------------------


def test_the_form_offers_these_industries_in_this_order() -> None:
    assert CODES == ("gambling", "crypto", "forex", "adult", "mlm")


def test_codes_labels_and_names_are_unique_plain_and_complete() -> None:
    for attr in ("code", "label", "short"):
        values = [getattr(niche, attr) for niche in NICHES]
        assert len(set(values)) == len(values)
        assert all(value and value == value.strip() for value in values)
    assert all(re.fullmatch(r"[a-z]+", niche.code) for niche in NICHES)


def test_lookups_by_code() -> None:
    assert label("crypto") == "Crypto and blockchain"
    assert short_name("forex") == "forex"
    assert label("nope") is None and short_name("nope") is None
    assert field_name("gambling") == "avoid_gambling"


def test_known_keeps_the_listed_codes_once_in_table_order() -> None:
    assert known(["mlm", "gambling", "gambling", "nope", ""]) == ("gambling", "mlm")
    assert known(None) == () and known([]) == ()
    assert known(iter(["crypto"])) == ("crypto",)


@pytest.mark.parametrize("fields, expected", [
    ({}, ()),
    ({"avoid_gambling": "1"}, ("gambling",)),
    ({"avoid_mlm": "1", "avoid_crypto": "1"}, ("crypto", "mlm")),  # the table's order
    ({"avoid_gambling": ""}, ()),
    ({"avoid_gambling": "on"}, ()),  # only the value the form sends
    ({"avoid_gambling": "0"}, ()),
    ({"avoid_nope": "1", "lang": "uk"}, ()),
])
def test_a_form_ticks_an_industry_only_with_the_exact_value(fields, expected) -> None:
    assert chosen(fields) == expected


# --- what is detected --------------------------------------------------------------------------


def about(code: str, head: str = "", text: str = "") -> bool:
    return code in detect_niches(head, text)


@pytest.mark.parametrize("code, head", [
    ("gambling", "Senior Developer, Online Casino"),
    ("gambling", "Sportsbook Backend Engineer"),
    ("gambling", "Poker Platform Engineer"),
    ("gambling", "iGaming Data Analyst"),
    ("gambling", "i-gaming QA"),
    ("gambling", "Bookmaker Odds Trader"),
    ("gambling", "Kasino Entwickler"),
    ("gambling", "Kasinobetreiber Backend"),
    ("gambling", "Casinomitarbeiter"),
    ("gambling", "Casinos Platform Engineer"),
    ("gambling", "Розробник в казино"),
    ("gambling", "Аналітик, букмекерська компанія"),
    ("crypto", "Blockchain Developer"),
    ("crypto", "Blockchainentwickler"),
    ("crypto", "Blockchains for Finance"),
    ("crypto", "Crypto Exchange Backend"),
    ("crypto", "Cryptocurrency Analyst"),
    ("crypto", "Web3 Engineer"),
    ("crypto", "Web 3 Engineer"),
    ("crypto", "Solidity Developer"),
    ("crypto", "DeFi Researcher"),
    ("crypto", "Bitcoin Core Contributor"),
    ("crypto", "Krypto Händler"),
    ("crypto", "Розробник криптовалютної біржі"),
    ("crypto", "Блокчейн розробник"),
    ("forex", "Forex Platform Developer"),
    ("forex", "Binary Options Support"),
    ("forex", "CFD Trading Backend"),
    ("forex", "Аналітик форекс"),
    ("adult", "Adult Entertainment Platform Engineer"),
    ("adult", "Backend for OnlyFans creators"),
    ("adult", "NSFW Content Moderator"),
    ("adult", "Porn site developer"),
    ("adult", "Erotik Shop Manager"),
    ("mlm", "Network Marketing Manager"),
    ("mlm", "Multi-level marketing coach"),
    ("mlm", "Multilevel marketing sales"),
    ("mlm", "MLM Distributor Support"),
    ("mlm", "MLMs and their distributors"),
    ("mlm", "Мережевий маркетинг, менеджер"),
])
def test_one_industry_word_in_the_head_is_enough(code, head) -> None:
    assert about(code, head=head)
    assert detect_niches(head, "") == [code]


@pytest.mark.parametrize("code, text", [
    ("gambling", "We run an online casino. Our casino has millions of players."),
    ("gambling", "We are a sportsbook and our poker room is new."),
    ("gambling", "Gambling operator. Responsible gambling is our priority."),
    ("crypto", "A crypto exchange. We list every crypto asset."),
    ("crypto", "We build on Ethereum and Bitcoin."),
    ("crypto", "Die Kryptowährung ist unser Geschäft, Krypto für alle."),
    ("crypto", "Наша біржа працює з криптовалютою та блокчейн."),
    ("forex", "Forex broker offering binary options."),
    ("adult", "Adult content platform; porn is not allowed to be stored in the clear."),
    ("mlm", "Network marketing company with a large downline."),
])
def test_the_words_of_an_industry_twice_in_the_description_are_enough(code, text) -> None:
    assert about(code, text=text)


@pytest.mark.parametrize("text", [
    "Experience with blockchain is a plus.",
    "We moved our payments to a casino-grade fraud engine.",
    "Our culture is gambling-free.",
    "Tell us about a time you were betting on the wrong horse.",
    "Join the poker night on Fridays.",
    "",
])
def test_one_passing_mention_in_the_description_is_not_enough(text) -> None:
    assert MIN_MENTIONS == 2
    assert detect_niches("Backend Engineer", text) == []


def test_the_head_may_be_one_mention_while_the_text_needs_two() -> None:
    assert about("crypto", head="Blockchain Dev", text="")
    assert not about("crypto", head="Dev", text="blockchain")
    assert about("crypto", head="Dev", text="blockchain, blockchain")


@pytest.mark.parametrize("text", [
    "We write cryptography libraries and cryptographic protocols. Cryptography everywhere.",
    "Video gaming engines and gaming consoles; game developers welcome.",
    "CFD solver for computational fluid dynamics; CFD meshing; CFD post-processing.",
    "Time slots and slot allocation; book a slot; slots slots slots.",
    "Place your bet on us; we bet big; a bet worth taking.",
    "Adult learning platform for adult education; adult learners and adult courses.",
    "Network engineering and network security; marketing and network operations.",
    "Pokerface, pokér and bingo-free zone.",
])
def test_words_that_only_look_like_an_industry_are_not_one(text) -> None:
    assert detect_niches("Backend Engineer", text) == []


def test_the_casino_cafeteria_case_is_the_two_mention_rule_at_work() -> None:
    # "casino" twice does mean gambling to this list; that is the documented price of the rule.
    assert detect_niches("", "casino casino") == ["gambling"]
    assert detect_niches("", "a casino-style cafeteria") == []


def test_case_does_not_matter() -> None:
    assert about("crypto", head="BLOCKCHAIN DEVELOPER")
    assert about("gambling", head="Online CASINO")
    assert about("gambling", head="КАЗИНО")


def test_crypto_counts_only_as_a_whole_word() -> None:
    assert about("crypto", head="Crypto.com backend")
    assert about("crypto", head="Crypto-exchange backend")
    assert not about("crypto", head="Cryptography engineer")
    assert not about("crypto", head="Криптографія та безпека")
    assert not about("crypto", head="Cryptographer")


def test_several_industries_come_back_in_the_tables_order() -> None:
    found = detect_niches("Crypto casino developer", "")
    assert found == ["gambling", "crypto"]


def test_nothing_is_found_in_empty_or_blank_text() -> None:
    assert detect_niches("", "") == []
    assert detect_niches("   ", "\n\t") == []


def test_every_industry_has_a_pattern_that_finds_its_own_name_in_a_head() -> None:
    samples = {"gambling": "gambling", "crypto": "cryptocurrency", "forex": "forex",
               "adult": "adult entertainment", "mlm": "network marketing"}
    assert set(samples) == set(CODES)
    for code, sample in samples.items():
        assert detect_niches(sample, "") == [code]


def test_a_long_text_is_handled_quickly() -> None:
    import time

    text = ("We build tools for teams. " * 4000) + "casino casino"
    started = time.perf_counter()
    assert detect_niches("Dev", text) == ["gambling"]
    assert time.perf_counter() - started < 1.0


def test_the_module_exposes_one_table_and_no_stray_state() -> None:
    assert isinstance(niches.NICHES, tuple)
    assert niches.FIELD_PREFIX == "avoid_" and niches.CHECKED == "1"


@pytest.mark.parametrize("code, head", [
    ("gambling", "Francasino Developer"),
    ("gambling", "Superpoker Engineer"),
    ("crypto", "Subcrypto Developer"),
    ("crypto", "Preblockchain Analyst"),
    ("forex", "Reforex Specialist"),
    ("adult", "Superporn Developer"),
    ("mlm", "Emlm Developer"),
])
def test_an_industry_word_must_start_a_word(code, head) -> None:
    assert not about(code, head=head)
    assert not about(code, text=f"{head}. {head}.")


def test_an_acronym_is_not_part_of_a_longer_word() -> None:
    assert not about("mlm", head="MLMX Developer")
    assert not about("mlm", text="mlmx and mlmy, mlmx")
    assert about("mlm", head="MLM Developer") and about("mlm", head="MLMs")
