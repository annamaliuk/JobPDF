"""Risky industries a person may not want to work in, and how a posting is tagged with them
(JM-51).

``detect_niches(head, text)`` says which of the listed industries a posting is about. It is a word
list, not a classifier, and it is built to say nothing when unsure: a job that is wrongly hidden
is lost to the person, a job that is wrongly shown costs a glance.

A posting is tagged with an industry when one of that industry's words

* is in its *head* (title, company, tags, category): the posting is about it, or
* appears at least ``MIN_MENTIONS`` times in the description: it is what the company does,
  not a passing remark ("blockchain experience is a plus").

Only words that mean the industry and little else are listed. "Cryptography", "gaming",
"bet", "CFD" or "slots" alone are not, and "crypto" counts only as a whole word. A word may
continue ("Kasinobetreiber", "Blockchainentwickler") but must start a word ("Subcrypto" is not
"crypto"). English, with
the German and Ukrainian words of the same industries, because those job boards send them.

``NICHES`` is the one table: the upload form offers its entries in this order and the stored
codes are their ``code``. A new industry is a new entry; the stored field only grows, and stored
vacancies learn about it when they are processed again.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

MIN_MENTIONS = 2  # description-only: how many words make a posting "about" an industry
FIELD_PREFIX = "avoid_"  # form field of an industry: avoid_<code>, value "1" when ticked
CHECKED = "1"


@dataclass(frozen=True)
class Niche:
    code: str  # stored on the vacancy and in the form's field name; never renamed
    label: str  # the form's checkbox
    short: str  # inside a sentence: "N postings in <short> left out"
    pattern: re.Pattern[str]


def _words(*alternatives: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")", re.IGNORECASE)


NICHES: tuple[Niche, ...] = (
    Niche("gambling", "Gambling and betting", "gambling", _words(
        r"casino\w*", r"kasino\w*", r"казино\b", r"gambl(?:ing|ers?)\b",
        r"sports?[- ]?books?\b", r"(?:sports?|online) betting\b", r"betting (?:site|app|compan|"
        r"platform|operator|market)\w*", r"bookmak(?:er|ers|ing)\b", r"wagering\b",
        r"i-?gaming\b", r"poker\b", r"slot (?:machine|game)s?\b", r"online slots\b", r"bingo\b",
        r"gl(?:ü|u|ue)cksspiel\w*", r"sportwetten\b", r"wettanbieter\w*",
        r"букмекер\w*", r"гемблінг\w*", r"азартн\w+ гр\w+", r"ставки на спорт\w*")),
    Niche("crypto", "Crypto and blockchain", "crypto", _words(
        r"crypto-?currenc(?:y|ies)\b", r"crypto\b", r"cryptos\b", r"bitcoin\w*", r"ethereum\b",
        r"blockchain\w*", r"web ?3\b", r"defi\b", r"nfts?\b", r"stablecoins?\b", r"altcoins?\b",
        r"smart contracts?\b", r"solidity\b", r"kryptow(?:ä|ae|a)hrung\w*", r"krypto\b",
        r"криптовалют\w*", r"блокчейн\w*", r"біткоїн\w*", r"крипто(?:біржа\w*|\b)")),
    Niche("forex", "Forex and binary options", "forex", _words(
        r"forex\b", r"binary options?\b", r"cfd trading\b", r"contracts? for difference\b",
        r"spread betting\b", r"форекс\w*", r"бінарн\w+ опціон\w*")),
    Niche("adult", "Adult content", "adult content", _words(
        r"porn\w*", r"adult (?:content|entertainment|industry|videos?|sites?|platform)\b",
        r"onlyfans\b", r"nsfw\b", r"sex toys?\b", r"erotic\w*", r"cam (?:sites?|models?)\b",
        r"erotik\b", r"порно\w*", r"еротичн\w*")),
    Niche("mlm", "Network marketing (MLM)", "network marketing", _words(
        r"multi-?level marketing\b", r"network marketing\b", r"mlms?\b", r"downlines?\b",
        r"мережев\w+ маркетинг\w*")),
)

_BY_CODE: Mapping[str, Niche] = {niche.code: niche for niche in NICHES}
CODES: tuple[str, ...] = tuple(_BY_CODE)


def field_name(code: str) -> str:
    return FIELD_PREFIX + code


def label(code: str) -> str | None:
    niche = _BY_CODE.get(code)
    return niche.label if niche else None


def short_name(code: str) -> str | None:
    niche = _BY_CODE.get(code)
    return niche.short if niche else None


def known(codes: Iterable[str] | None) -> tuple[str, ...]:
    """The listed industries among ``codes``, in the table's order, each once."""
    wanted = set(codes or ())
    return tuple(code for code in CODES if code in wanted)


def chosen(fields: Mapping[str, str]) -> tuple[str, ...]:
    """The industries a submitted form ticked: its ``avoid_<code>`` field is exactly "1"."""
    return tuple(code for code in CODES if fields.get(field_name(code)) == CHECKED)


def detect_niches(head: str, text: str) -> list[str]:
    """The codes of the industries a posting is about, in the table's order; [] when none."""
    found = []
    for niche in NICHES:
        if niche.pattern.search(head) or len(niche.pattern.findall(text)) >= MIN_MENTIONS:
            found.append(niche.code)
    return found
