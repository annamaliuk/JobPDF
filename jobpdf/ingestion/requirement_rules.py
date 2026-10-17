"""Rule-based requirement extraction from posting lines (JM-48).

No LLM: every value comes from a word list, a regular expression or the skill taxonomy, so
the same posting always gives the same result, and each item keeps the line it was found
on as its quote. Steps:

1. ``split_sections``: headings are sorted into requirements, nice-to-have,
   responsibilities and other (benefits, about us) with word lists in English, German and
   Ukrainian; the longest matching phrase wins.
2. ``SkillVocabulary.scan``: the longest taxonomy alias (JM-16 ``alias_norm``) of up to
   ``MAX_ALIAS_WORDS`` words, line by line. Very short or everyday-word aliases ("Go", "R",
   "Spring") only count when written with a capital on a line that names another skill.
   Aliases of ESCO's non-digital skills ("innovation", "help customers") are "weak": they
   count only in requirement sections, because in company blurbs and duties they are
   ordinary words. Boilerplate sentences (equal opportunity, benefits, how to apply) are
   never read.
3. Years, education and languages: regular expressions over requirement lines, skipping
   lines that only say "nice to have".
4. Responsibilities: the lines of the responsibilities sections (bullets when there are).

Word lists are deliberately plain; the human check (scripts/check_vacancy_requirements.py)
is the place to find what they miss.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jobpdf.ingestion.posting_text import Line
from jobpdf.normalization.taxonomy import EXACT_ONLY_MAX_LEN
from jobpdf.normalization.text import normalize

REQUIREMENTS = "requirements"
NICE_TO_HAVE = "nice_to_have"
RESPONSIBILITIES = "responsibilities"
OTHER = "other"  # benefits, about us, how to apply: skipped
GENERAL = "general"  # before any heading, or under one that is not recognised

MAX_ALIAS_WORDS = 5
MAX_RESPONSIBILITIES = 15
MAX_RESPONSIBILITY_CHARS = 200
MAX_YEARS = 15  # a larger number is a company age or a typo, not a requirement

# Phrases are compared after lowercasing and turning punctuation into spaces.
SECTION_PHRASES: dict[str, list[str]] = {
    NICE_TO_HAVE: [
        "nice to have", "nice to haves", "bonus", "bonus points", "plus", "a plus",
        "preferred", "preferred qualifications", "good to have", "would be great",
        "desirable", "optional", "extra points", "ideally",
        "wünschenswert", "von vorteil", "pluspunkte",
        "буде плюсом", "буде перевагою", "перевагою буде", "бажано", "плюсом",
    ],
    OTHER: [
        "benefits", "perks", "what we offer", "we offer", "why join", "why join us",
        "why work with us", "about us", "about the company", "who we are", "compensation",
        "salary", "how to apply", "application process", "interview process", "our values",
        "culture", "equal opportunity", "diversity", "what s in it for you",
        "wir bieten", "was wir bieten", "über uns", "deine vorteile", "unser angebot",
        "ми пропонуємо", "що ми пропонуємо", "про нас", "про компанію", "умови",
    ],
    RESPONSIBILITIES: [
        "responsibilities", "key responsibilities", "your responsibilities",
        "what you ll do", "what you will do", "what you ll be doing", "what you will be doing",
        "your role", "the role", "your tasks", "your mission", "day to day", "in this role",
        "duties", "what you do", "you will",
        "aufgaben", "deine aufgaben", "ihre aufgaben", "tätigkeiten", "das erwartet dich",
        "обов язки", "обовʼязки", "твої задачі", "задачі", "завдання", "що потрібно робити",
    ],
    REQUIREMENTS: [
        "requirements", "requirement", "qualifications", "minimum qualifications",
        "basic qualifications", "what you bring", "what we re looking for",
        "what we are looking for", "who you are", "about you", "you have", "you bring",
        "must have", "must haves", "skills", "required skills", "your profile",
        "experience", "tech stack", "our stack", "stack", "technologies",
        "what you need", "what you ll need", "you will need", "what you will need",
        "anforderungen", "dein profil", "ihr profil", "qualifikationen",
        "was du mitbringst", "was sie mitbringen", "das bringst du mit", "voraussetzungen",
        "вимоги", "ми очікуємо", "що ми очікуємо", "кваліфікація", "навички",
        "необхідні навички", "твій досвід", "досвід",
    ],
}
_PRIORITY = [NICE_TO_HAVE, OTHER, RESPONSIBILITIES, REQUIREMENTS]  # ties go to the earlier
_HEADING_KEYS = {" ".join(p.replace("ʼ", " ").split()) for phrases in SECTION_PHRASES.values()
                 for p in phrases if len(p.split()) > 1 or len(p) > 6} - {"experience"}

# A sentence holding one of these is about a preference, not a requirement.
PREFERRED_CUES = [
    "nice to have", "is a plus", "a plus", "is a bonus", "bonus", "preferred", "preferably",
    "ideally", "would be great", "would be nice", "is an advantage", "an advantage",
    "desirable", "good to have", "optional",
    "von vorteil", "wünschenswert", "ein plus", "idealerweise",
    "буде плюсом", "буде перевагою", "перевагою", "бажано",
]
# "a huge plus", "a big plus"; not "5 plus years".
_PLUS_CUE = re.compile(r"\b(?:a|an|huge|big|great|strong|definite|real|nice)\s+plus\b",
                       re.IGNORECASE)
# Sentences that are about the employer, not the job: no skill or requirement is read there.
_BOILERPLATE = re.compile(
    r"equal (?:employment )?opportunit|regardless of|discriminat|protected (?:veteran|by law)|"
    r"reasonable accommodation|affirmative action|e-?verify|benefits? (?:include|to support)|"
    r"paid time off|401\(?k|stock purchase|health insurance|how did you hear|"
    r"application process|employee referral|privacy (?:policy|notice)|"
    r"find more .{0,40}\bjobs\b",
    re.IGNORECASE,
)
# A sentence that states what the candidate needs: weak skills are trusted there too.
_REQUIREMENT_CUE = re.compile(
    r"\b(?:skills?|experience (?:with|in)|ability to|able to|proficien\w*|knowledge of|"
    r"familiar\w*|you (?:have|bring|are|can)|must|required|strong|excellent|"
    r"looking for|ideal for|who can)\b",
    re.IGNORECASE,
)
_CLAUSE = re.compile(r"\s*[,;]\s*|\s+but\s+")

# Real skills that are also everyday words: they need a capital and another skill nearby.
AMBIGUOUS = {
    "go", "r", "c", "d", "spring", "rest", "swift", "express", "less", "make", "chef",
    "puppet", "ant", "dart", "rust", "ember", "flask", "sass", "elm", "julia", "crystal",
    "salt", "maven", "gradle", "jest", "mocha", "vue", "meteor", "backbone", "nest",
}
# Aliases so generic that, alone, they say nothing about a posting's requirements.
GENERIC = {
    "data", "support", "management", "development", "software", "engineering", "work",
    "team", "business", "product", "service", "services", "it", "web", "tools", "tool",
    "technology", "technologies", "systems", "system", "solutions", "experience",
    "knowledge", "skills", "skill", "office", "computer", "internet", "email", "online",
    "digital", "platform", "process", "processes", "quality", "project", "projects",
    "technical skills", "soft skills", "lead", "plan", "design", "call", "source", "clean",
    "discipline", "schedule", "feedback", "reporting", "activity", "interaction", "craft",
    "history", "purchase", "rights", "brands", "fuel", "did", "scratch", "reserve",
    "innovation", "innovate", "processor", "credibility", "judgment", "commitment",
    "dedication", "loyalty", "confidence", "hybrid model", "remote",
}
# Source tags that describe the job, not a skill.
TAG_STOP = {"remote", "hybrid", "onsite", "on site", "full time", "part time", "contract",
            "freelance", "internship", "worldwide", "anywhere"}

_PUNCT = re.compile(r"[^\w]+")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")
# Characters a skill token may contain besides letters and digits: c++, c#, node.js, ci/cd.
_TOKEN = re.compile(r"[\w][\w+#./\-']*[\w+#]|[\w][+#]*|\.[\w]+")


def phrase_key(text: str) -> str:
    """Lowercase words separated by single spaces, punctuation dropped."""
    return " ".join(_PUNCT.sub(" ", text.casefold().replace("ʼ", " ")).split())


def _contains(haystack: str, phrase: str) -> bool:
    return f" {phrase} " in f" {haystack} "


def classify_heading(text: str) -> str:
    """The section a heading opens; GENERAL when no phrase matches."""
    key = phrase_key(text)
    best, best_len = GENERAL, 0
    for section in _PRIORITY:
        for phrase in SECTION_PHRASES[section]:
            if len(phrase) > best_len and _contains(key, phrase_key(phrase)):
                best, best_len = section, len(phrase)
    return best


@dataclass(frozen=True)
class SectionLine:
    line: Line
    section: str


def split_sections(lines: Sequence[Line]) -> list[SectionLine]:
    """Each non-heading line with the section it sits in. A title-shaped line opens the
    section it names; one that names none ends the current section and stays a line. A
    line that is exactly a heading phrase ("Responsibilities") is a heading too."""
    current = GENERAL
    out: list[SectionLine] = []
    for line in lines:
        if line.is_heading or phrase_key(line.text) in _HEADING_KEYS:
            current = classify_heading(line.text)
            continue
        if line.is_title:
            current = classify_heading(line.text)
            if current != GENERAL:
                continue
        out.append(SectionLine(line, current))
    return out


def has_preferred_cue(sentence: str) -> bool:
    key = phrase_key(sentence)
    return (any(_contains(key, phrase_key(cue)) for cue in PREFERRED_CUES)
            or bool(_PLUS_CUE.search(sentence)))


def is_boilerplate(sentence: str) -> bool:
    return bool(_BOILERPLATE.search(sentence))


def has_requirement_cue(sentence: str) -> bool:
    return bool(_REQUIREMENT_CUE.search(sentence))


def clauses(sentence: str) -> list[str]:
    return [c for c in _CLAUSE.split(sentence) if c.strip()]


_ABBREVIATIONS = re.compile(r"\b(e\.g|i\.e|etc|incl|approx|vs)\.", re.IGNORECASE)
_KEEP = "\u2024"  # one-dot leader: stands in for an abbreviation's dot while splitting


def sentences(text: str) -> list[str]:
    """Split after . ! ? ; but not after "e.g." or "etc."."""
    guarded = _ABBREVIATIONS.sub(lambda m: m.group(1) + _KEEP, text)
    return [s.replace(_KEEP, ".") for s in _SENTENCE.split(guarded) if s.strip()]


# --- skills -------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillHit:
    text: str  # as written in the posting
    norm: str  # normalize(text): equals a taxonomy alias_norm
    quote: str  # the sentence it was found in
    preferred: bool
    strong: bool = True  # False: a non-digital ESCO skill, trusted in requirement sections only


@dataclass
class SkillVocabulary:
    """Taxonomy aliases by ``alias_norm``. ``complete`` is False when only the committed
    custom skills could be read (the full ESCO build is missing). ``weak`` holds the norms
    that only name non-digital ESCO skills."""

    norms: set[str] = field(default_factory=set)
    complete: bool = True
    max_words: int = 1
    weak: set[str] = field(default_factory=set)

    @classmethod
    def from_norms(cls, norms: Iterable[str], complete: bool = True,
                   weak: Iterable[str] = ()) -> SkillVocabulary:
        kept = {n for n in norms if n and n not in GENERIC}
        longest = max((len(n.split()) for n in kept), default=1)
        return cls(kept, complete, min(longest, MAX_ALIAS_WORDS), set(weak) & kept)

    @classmethod
    def load(cls, taxonomy_dir: Path = Path("data/taxonomy")) -> SkillVocabulary:
        """``aliases.csv`` from the taxonomy build (with ``concepts.csv`` saying which skills
        are digital); without it, ``custom_skills.csv``."""
        aliases = taxonomy_dir / "aliases.csv"
        if aliases.is_file():
            strong_ids = _strong_concepts(taxonomy_dir / "concepts.csv")
            norms: set[str] = set()
            strong: set[str] = set()
            with aliases.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    norms.add(row["alias_norm"])
                    if (strong_ids is None or row["canonical_id"] in strong_ids
                            or row.get("alias_kind") == "custom"):
                        strong.add(row["alias_norm"])
            return cls.from_norms(norms, weak=norms - strong)
        custom = taxonomy_dir / "custom_skills.csv"
        norms: list[str] = []
        if custom.is_file():
            with custom.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    norms += [normalize(row.get("canonical_name") or ""),
                              normalize(row.get("alias") or "")]
        return cls.from_norms(norms, complete=False)

    def scan(self, text: str, preferred: bool = False) -> list[SkillHit]:
        """Skills named in one line, in order, each once; boilerplate sentences skipped."""
        hits: list[SkillHit] = []
        for sentence in sentences(text):
            if is_boilerplate(sentence):
                continue
            pref = preferred or has_preferred_cue(sentence)
            hits += self._scan_sentence(sentence, pref)
        unambiguous = [h for h in hits if not _is_ambiguous(h.norm)]
        kept = [h for h in hits
                if not _is_ambiguous(h.norm) or (_capitalised(h.text) and
                                                 any(o.norm != h.norm for o in unambiguous))]
        seen: set[str] = set()
        out = []
        for hit in kept:
            if hit.norm not in seen:
                seen.add(hit.norm)
                out.append(hit)
        return out

    def _scan_sentence(self, sentence: str, preferred: bool) -> list[SkillHit]:
        tokens = [_trim(t) for t in _TOKEN.findall(sentence)]
        tokens = [t for t in tokens if t]
        hits: list[SkillHit] = []
        i = 0
        while i < len(tokens):
            for size in range(min(self.max_words, len(tokens) - i), 0, -1):
                span = " ".join(tokens[i:i + size])
                norm = normalize(span)
                if norm in self.norms:
                    hits.append(SkillHit(span, norm, sentence, preferred,
                                         norm not in self.weak))
                    i += size
                    break
            else:
                if "/" in tokens[i]:  # "Python/Django": try each part
                    for part in tokens[i].split("/"):
                        norm = normalize(part)
                        if part and norm in self.norms:
                            hits.append(SkillHit(part, norm, sentence, preferred,
                                                 norm not in self.weak))
                i += 1
        return hits


def _strong_concepts(concepts_csv: Path) -> set[str] | None:
    """IDs of digital or custom concepts; None when concepts.csv is missing (all trusted)."""
    if not concepts_csv.is_file():
        return None
    with concepts_csv.open(encoding="utf-8", newline="") as f:
        return {row["canonical_id"] for row in csv.DictReader(f)
                if row.get("is_digital") == "true" or row.get("source") == "custom"}


def _trim(token: str) -> str:
    """Drop a sentence-ending dot ("Python.") but keep ".NET" and "Node.js"."""
    while token.endswith((".", "-", "/", "'")) and len(token) > 1:
        token = token[:-1]
    return token


def _is_ambiguous(norm: str) -> bool:
    return len(norm) <= EXACT_ONLY_MAX_LEN or norm in AMBIGUOUS


def _capitalised(text: str) -> bool:
    return text[:1].isupper()


# --- years --------------------------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10,
}
_NUM = r"(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")"
_RANGE = rf"{_NUM}\s*(?:\+|plus)?\s*(?:(?:-|–|—|to|bis|до)\s*{_NUM}\s*\+?)?"
_YEARS = re.compile(
    rf"(?<![\w.]){_RANGE}\s*\+?\s*(?:years?|yrs?|jahre?n?|рок(?:и|ів|у)|років)(?!\s*(?:old|ago))",
    re.IGNORECASE,
)
_YEARS_FROM = re.compile(rf"(?:від|понад|from)\s+{_NUM}\s*(?:рок(?:и|ів|у)|років|years?)",
                         re.IGNORECASE)


def _number(text: str) -> int:
    return int(text) if text.isdigit() else _NUMBER_WORDS[text.lower()]


def years_in(sentence: str) -> list[int]:
    """Lower bounds of the experience spans a sentence states."""
    found = [_number(m.group(1)) for m in _YEARS.finditer(sentence)]
    found += [_number(m.group(1)) for m in _YEARS_FROM.finditer(sentence)]
    return [n for n in found if 0 < n <= MAX_YEARS]


# --- education ----------------------------------------------------------------------

EDUCATION_LEVELS: list[tuple[str, list[str]]] = [  # lowest first
    ("bachelor", ["bachelor", "bachelors", "bachelor s", "bsc", "b sc", "bs degree",
                  "ba degree", "b a degree", "b s degree", "бакалавр", "бакалавра"]),
    ("master", ["master s", "master degree", "masters degree", "msc", "m sc", "ms degree",
                "mba", "магістр", "магістра"]),
    ("phd", ["phd", "ph d", "doctorate", "doctoral degree", "кандидат наук"]),
]
GENERIC_DEGREE = ["university degree", "college degree", "academic degree", "degree in",
                  "a degree", "higher education", "studium", "hochschulabschluss",
                  "вища освіта", "вищу освіту", "вищої освіти"]
_NOT_EDUCATION = ["or equivalent", "or similar experience", "or relevant experience",
                  "or comparable", "oder vergleichbar", "oder vergleichbare", "або еквівалентний",
                  "to a degree", "to some degree"]
_FIELD = re.compile(r"(?i:\b(?:degree|bachelor'?s?|master'?s?|bsc|msc|phd)\b[^.;]*?\bin)\s+"
                    r"([A-Z][\w&\- ]{2,60}?)(?=\s*(?:,|\bor\b|\band\b|\.|;|\(|$))")


@dataclass(frozen=True)
class EducationHit:
    level: str
    field: str | None
    quote: str


def education_in(sentence: str) -> EducationHit | None:
    """The lowest degree a sentence names ("Bachelor's or Master's" -> bachelor); "degree"
    when it names none in particular. None when it is optional ("or equivalent experience")."""
    key = phrase_key(sentence)
    if any(_contains(key, phrase_key(p)) for p in _NOT_EDUCATION):
        return None
    level = next((lvl for lvl, phrases in EDUCATION_LEVELS
                  if any(_contains(key, phrase_key(p)) for p in phrases)), None)
    if level is None and any(_contains(key, phrase_key(p)) for p in GENERIC_DEGREE):
        level = "degree"
    if level is None:
        return None
    match = _FIELD.search(sentence)
    return EducationHit(level, match.group(1).strip() if match else None, sentence)


# --- languages ----------------------------------------------------------------------

LANGUAGES: dict[str, list[str]] = {  # name -> regex stems
    "English": [r"english", r"englisch\w*", r"англійськ\w*"],
    "German": [r"german", r"deutsch\w*", r"німецьк\w*"],
    "Ukrainian": [r"ukrainian", r"ukrainisch\w*", r"українськ\w*"],
    "French": [r"french", r"französisch\w*", r"французьк\w*"],
    "Spanish": [r"spanish", r"spanisch\w*", r"іспанськ\w*"],
    "Polish": [r"polish", r"polnisch\w*", r"польськ\w*"],
    "Italian": [r"italian", r"italienisch\w*", r"італійськ\w*"],
    "Portuguese": [r"portuguese", r"portugiesisch\w*", r"португальськ\w*"],
    "Dutch": [r"dutch", r"niederländisch\w*", r"нідерландськ\w*"],
    "Russian": [r"russian", r"russisch\w*", r"російськ\w*"],
    "Chinese": [r"chinese", r"mandarin", r"cantonese", r"chinesisch\w*", r"китайськ\w*"],
    "Japanese": [r"japanese", r"japanisch\w*", r"японськ\w*"],
    "Korean": [r"korean", r"koreanisch\w*", r"корейськ\w*"],
    "Arabic": [r"arabic", r"arabisch\w*", r"арабськ\w*"],
    "Turkish": [r"turkish", r"türkisch\w*", r"турецьк\w*"],
    "Hindi": [r"hindi"],
}
_LANGUAGE_RES = {name: re.compile(r"(?<!\w)(?:" + "|".join(stems) + r")(?!\w)", re.IGNORECASE)
                 for name, stems in LANGUAGES.items()}
_PROFICIENCY = re.compile(
    r"(?<!\w)(?:fluent\w*|fluency|proficien\w*|native|business|professional|written|spoken|"
    r"speak\w*|read and write|bilingual|"
    r"excellent|strong|good|advanced|upper|intermediate|level|language|communication|"
    r"[abc][12]|fließend\w*|verhandlungssicher\w*|sehr gute\w*|gute\w*|kenntnisse|"
    r"muttersprach\w*|sprache\w*|рівень|рівні|вільн\w*|мов[аиою]\w*|знання)(?!\w)",
    re.IGNORECASE,
)


def languages_in(sentence: str) -> list[str]:
    """Languages a sentence requires: clauses that only state a preference ("German would
    be a big plus") are left out. Latin-script names must be capitalised, so "polish the
    UI" is not Polish."""
    if not _PROFICIENCY.search(sentence) or is_boilerplate(sentence):
        return []
    found: list[str] = []
    for clause in clauses(sentence):
        if has_preferred_cue(clause):
            continue
        for name, pattern in _LANGUAGE_RES.items():
            if name not in found and any(_language_word(m.group(0))
                                         for m in pattern.finditer(clause)):
                found.append(name)
    return found


def is_language_name(text: str) -> bool:
    """ESCO lists languages as skills ("English", "speak Spanish"); they belong in
    ``languages`` instead."""
    return any(_language_word(m.group(0))
               for p in _LANGUAGE_RES.values() for m in p.finditer(text))


def _language_word(word: str) -> bool:
    return not word.isascii() or word[:1].isupper()


# --- responsibilities ---------------------------------------------------------------


def responsibilities_of(lines: Sequence[SectionLine]) -> list[str]:
    """Lines of the responsibilities sections; only their bullets when they have any."""
    own = [s.line for s in lines if s.section == RESPONSIBILITIES]
    if any(line.is_bullet for line in own):
        own = [line for line in own if line.is_bullet]
    return [_cut(line.text) for line in own][:MAX_RESPONSIBILITIES]


def _cut(text: str, limit: int = MAX_RESPONSIBILITY_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0]
    return head.rstrip(",;:") + "…"
