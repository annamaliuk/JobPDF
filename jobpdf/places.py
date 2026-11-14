"""Countries in a posting's location text (JM-49).

``resolve_country("Berlin, Germany")`` -> ``"DE"``. It is a small word list, not a geocoder, and
it is built to be wrong by saying nothing: when the text names no country, or names two, or
uses a word that is both a country and a US state ("Georgia") or a code that is both ("CA",
"DE"), the answer is ``None`` and the caller keeps the vacancy. A wrong country would hide a
job the person could take; an unknown one only shows a job they may not.

What it reads, in this order, on the comma-separated parts of the text from the right:
a country name or its usual variants (English and Ukrainian), a US state or Canadian
province, a UK nation, and last the largest cities ("London"). A two-letter code counts only
when it is written in capitals and is not also a US state.

Codes are ISO 3166-1 alpha-2. ``COUNTRY_NAMES`` is the list the web form offers.

``allowed_countries("USA, Canada")`` -> ``{"US", "CA"}`` (JM-47) reads the place a remote job is
open to: every part must be a country, a US state or province, or a region ("Europe", "EU",
"LATAM"). One part it does not know, or an open word ("Worldwide"), and the answer is ``None``.
"Europe" includes Ukraine; "EU" and "EEA" do not.
"""

from __future__ import annotations

import re
import unicodedata

# code | English name | other names, ";"-separated (all compared in lower case, accents removed)
_COUNTRIES = """
AL|Albania
DZ|Algeria
AR|Argentina
AM|Armenia
AU|Australia
AT|Austria|österreich
AZ|Azerbaijan
BH|Bahrain
BD|Bangladesh
BY|Belarus|білорусь|білорусія
BE|Belgium|belgique;belgië
BA|Bosnia and Herzegovina|bosnia
BR|Brazil|brasil
BG|Bulgaria|болгарія
CA|Canada|канада
CL|Chile
CN|China|китай
CO|Colombia
CR|Costa Rica
HR|Croatia|hrvatska;хорватія
CY|Cyprus|кіпр
CZ|Czechia|czech republic;česko;чехія
DK|Denmark|danmark;данія
DO|Dominican Republic
EC|Ecuador
EG|Egypt|єгипет
EE|Estonia|eesti;естонія
FI|Finland|suomi;фінляндія
FR|France|франція
GE|Georgia (country)|грузія;tbilisi
DE|Germany|deutschland;німеччина
GH|Ghana
GR|Greece|ελλάδα;греція
HK|Hong Kong
HU|Hungary|magyarország;угорщина
IS|Iceland|ísland
IN|India|індія
ID|Indonesia
IE|Ireland|éire;republic of ireland;ірландія
IL|Israel|ізраїль
IT|Italy|italia;італія
JP|Japan|日本;японія
JO|Jordan
KZ|Kazakhstan|казахстан
KE|Kenya
KW|Kuwait
LV|Latvia|latvija;латвія
LB|Lebanon
LT|Lithuania|lietuva;литва
LU|Luxembourg|люксембург
MY|Malaysia
MT|Malta|мальта
MX|Mexico|méxico;мексика
MD|Moldova|молдова
ME|Montenegro|чорногорія
MA|Morocco
NL|Netherlands|the netherlands;holland;nederland;нідерланди
NZ|New Zealand|нова зеландія
NG|Nigeria
MK|North Macedonia|macedonia
NO|Norway|norge;норвегія
OM|Oman
PK|Pakistan
PA|Panama
PE|Peru
PH|Philippines
PL|Poland|polska;польща
PT|Portugal|португалія
QA|Qatar
RO|Romania|românia;румунія
RU|Russia|russian federation;росія
SA|Saudi Arabia
RS|Serbia|srbija;сербія
SG|Singapore|сінгапур
SK|Slovakia|slovensko;словаччина
SI|Slovenia|slovenija;словенія
ZA|South Africa
KR|South Korea|korea;republic of korea
ES|Spain|españa;іспанія
LK|Sri Lanka
SE|Sweden|sverige;швеція
CH|Switzerland|schweiz;suisse;svizzera;швейцарія
TW|Taiwan
TH|Thailand
TN|Tunisia
TR|Turkey|türkiye;turkiye;туреччина
UA|Ukraine|україна;украина;ukraina
AE|United Arab Emirates|uae;u.a.e.;оае
GB|United Kingdom|uk;u.k.;great britain;britain;england;scotland;wales;
   northern ireland;велика британія;британія
US|United States|usa;u.s.;u.s.a.;united states of america;america;сша
UY|Uruguay
UZ|Uzbekistan|узбекистан
VE|Venezuela
VN|Vietnam
"""

_US_STATES = """
AL|Alabama;AK|Alaska;AZ|Arizona;AR|Arkansas;CA|California;CO|Colorado;CT|Connecticut;
DE|Delaware;DC|District of Columbia;FL|Florida;GA|Georgia;HI|Hawaii;ID|Idaho;IL|Illinois;
IN|Indiana;IA|Iowa;KS|Kansas;KY|Kentucky;LA|Louisiana;ME|Maine;MD|Maryland;
MA|Massachusetts;MI|Michigan;MN|Minnesota;MS|Mississippi;MO|Missouri;MT|Montana;
NE|Nebraska;NV|Nevada;NH|New Hampshire;NJ|New Jersey;NM|New Mexico;NY|New York;
NC|North Carolina;ND|North Dakota;OH|Ohio;OK|Oklahoma;OR|Oregon;PA|Pennsylvania;
RI|Rhode Island;SC|South Carolina;SD|South Dakota;TN|Tennessee;TX|Texas;UT|Utah;
VT|Vermont;VA|Virginia;WA|Washington;WV|West Virginia;WI|Wisconsin;WY|Wyoming
"""

_CA_PROVINCES = """
AB|Alberta;BC|British Columbia;MB|Manitoba;NB|New Brunswick;NL|Newfoundland and Labrador;
NS|Nova Scotia;ON|Ontario;PE|Prince Edward Island;QC|Quebec;SK|Saskatchewan
"""

# The largest cities that postings name without a country. Only names that are not also a
# well-known place elsewhere ("Birmingham", "Cambridge", "Paris, TX" are left out on purpose).
_CITIES = """
GB|london;manchester;edinburgh;glasgow;leeds;bristol;liverpool;sheffield;cardiff;belfast
US|new york;new york city;nyc;san francisco;los angeles;seattle;chicago;boston;austin;
   denver;atlanta;san diego;san jose;washington dc;miami;dallas;houston;portland;
   philadelphia;phoenix;palo alto;mountain view;sunnyvale;san mateo;redwood city
CA|toronto;vancouver;montreal;ottawa;calgary;edmonton;waterloo
DE|berlin;munich;münchen;hamburg;frankfurt;cologne;köln;stuttgart;düsseldorf;leipzig;dresden
FR|lyon;marseille;toulouse;nantes;bordeaux;lille
NL|amsterdam;rotterdam;utrecht;eindhoven;the hague;den haag
ES|madrid;barcelona;valencia;seville;sevilla;malaga;bilbao
PT|lisbon;lisboa;porto;braga
IT|rome;roma;milan;milano;turin;torino;naples;bologna;florence
PL|warsaw;warszawa;krakow;kraków;wroclaw;wrocław;gdansk;gdańsk;poznan;poznań;lodz;łódź;
   katowice;szczecin;lublin;варшава;краків;вроцлав;гданськ;познань
UA|kyiv;kiev;lviv;kharkiv;odesa;odessa;dnipro;zaporizhzhia;київ;львів;харків;одеса;дніпро;
   запоріжжя;івано-франківськ;чернівці;вінниця
CZ|prague;praha;brno;ostrava
AT|vienna;wien;graz;salzburg
CH|zurich;zürich;geneva;genève;basel;bern;lausanne
IE|dublin;cork;galway
SE|stockholm;gothenburg;göteborg;malmö
DK|copenhagen;københavn;aarhus
NO|oslo;bergen;trondheim
FI|helsinki;espoo;tampere
LT|vilnius;kaunas
LV|riga
EE|tallinn;tartu
RO|bucharest;cluj;cluj-napoca;timisoara;timișoara;iasi;iași
BG|sofia;plovdiv;varna
HU|budapest
RS|belgrade;novi sad
HR|zagreb;split
SK|bratislava;kosice
IN|bangalore;bengaluru;hyderabad;mumbai;pune;chennai;delhi;new delhi;gurgaon;gurugram;noida
SG|singapore
JP|tokyo;osaka
AU|sydney;melbourne;brisbane;perth;canberra
AE|dubai;abu dhabi
IL|tel aviv;jerusalem;haifa
TR|istanbul;ankara
"""


def _fold(text: str) -> str:
    """Lower case, accents removed (so "Köln" and "Koln" meet), whitespace collapsed."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def _build() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    names: dict[str, str] = {}
    words: dict[str, str] = {}
    for line in _COUNTRIES.strip().replace("\n   ", "").splitlines():
        code, name, *rest = line.split("|")
        name = name.removesuffix(" (country)")
        names[code] = name
        for alias in [name, *(rest[0].split(";") if rest else [])]:
            words[_fold(alias)] = code
    states: dict[str, str] = {}
    for item in _US_STATES.replace("\n", "").strip().strip(";").split(";"):
        code, name = item.strip().split("|")
        states[_fold(name)] = code
    provinces: dict[str, str] = {}
    for item in _CA_PROVINCES.replace("\n", "").strip().strip(";").split(";"):
        code, name = item.strip().split("|")
        provinces[_fold(name)] = code
    cities: dict[str, str] = {}
    for block in _CITIES.strip().replace("\n   ", "").splitlines():
        code, listing = block.split("|")
        for city in listing.split(";"):
            cities[_fold(city)] = code
    return names, words, {**states, **provinces}, cities


COUNTRY_NAMES, _COUNTRY_WORDS, _REGION_WORDS, _CITY_WORDS = _build()
_STATE_CODES = frozenset(re.findall(r"(?:^|;)\s*([A-Z]{2})\|", _US_STATES.replace("\n", "")))
_PROVINCE_CODES = frozenset(re.findall(r"(?:^|;)\s*([A-Z]{2})\|", _CA_PROVINCES.replace("\n", "")))
_REGION_COUNTRY = {**{n: "US" for n in _STATE_CODES}, **{n: "CA" for n in _PROVINCE_CODES}}
_PART_SPLIT = re.compile(r"[,;/|()\[\]•·–—]|\s-\s")


def country_name(code: str | None) -> str | None:
    """The English name of an ISO code from the list, else None."""
    return COUNTRY_NAMES.get((code or "").upper())


def is_country(code: str | None) -> bool:
    return (code or "").upper() in COUNTRY_NAMES


def resolve_country(text: str | None) -> str | None:
    """The one country ``text`` names; None when it names none, several, or is unsure."""
    if not text or not text.strip():
        return None
    found: set[str] = set()
    for part in (p.strip() for p in _PART_SPLIT.split(text)):
        if not part:
            continue
        code = _resolve_part(part)
        if code:
            found.add(code)
    return next(iter(found)) if len(found) == 1 else None


def _resolve_part(part: str) -> str | None:
    folded = _fold(part)
    if folded in _COUNTRY_WORDS and folded in _REGION_WORDS:
        return None  # "Georgia" is a country and a US state: say nothing
    if folded in _COUNTRY_WORDS:
        return _COUNTRY_WORDS[folded]
    if folded in _REGION_WORDS:
        return _REGION_COUNTRY[_REGION_WORDS[folded]]
    if part.isupper() and len(part) == 2:  # a code, only when written as one
        if part in _STATE_CODES and part in COUNTRY_NAMES:
            return None  # "CA", "DE", "IN": a state and a country
        if part in _STATE_CODES:
            return "US"
        if part in _PROVINCE_CODES:
            return "CA"
        if part in COUNTRY_NAMES:
            return part
        return None
    return _CITY_WORDS.get(folded)


# Regions a remote posting may be limited to (JM-47), as codes from the list above. A word not
# here (a time zone, "CIS") is unknown, and an unknown part means the job is never hidden.
_EU = "AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE"
_EUROPE = f"{_EU} AL BA BY CH GB IS MD ME MK NO RS RU UA"
_LATAM = "AR BR CL CO CR DO EC MX PA PE UY VE"
_APAC = "AU BD CN HK ID IN JP KR LK MY NZ PH PK SG TH TW VN"
_REGIONS = {
    "europe": _EUROPE,
    "eu": _EU, "european union": _EU,
    "eea": f"{_EU} IS NO",
    "emea": f"{_EUROPE} AE AM AZ BH DZ EG GE GH IL JO KE KW LB MA NG OM QA SA TN TR ZA",
    "north america": "US CA MX",
    "latam": _LATAM, "latin america": _LATAM,
    "south america": "AR BR CL CO EC PE UY VE",
    "americas": f"US CA {_LATAM}",
    "apac": _APAC, "asia pacific": _APAC,
}
_REGION_SETS = {word: frozenset(codes.split()) for word, codes in _REGIONS.items()}
_OPEN = frozenset({"worldwide", "anywhere", "global", "anywhere in the world"})
_ONLY = re.compile(r"\b(?:only|based)\b", re.I)


def allowed_countries(text: str | None) -> frozenset[str] | None:
    """The countries ``text`` limits a remote job to; None when it is open, empty or unsure."""
    allowed: set[str] = set()
    for part in _PART_SPLIT.split(text or ""):
        part = _ONLY.sub(" ", part).strip(" -")
        folded = _fold(part)
        if folded in ("", "remote"):
            continue
        if folded in _OPEN:
            return None
        if folded in _REGION_SETS:
            allowed |= _REGION_SETS[folded]
        elif code := _resolve_part(part):
            allowed.add(code)
        else:
            return None
    return frozenset(allowed) or None
