"""Heading synonyms per section type, English and Ukrainian.

Keep entries lowercase and in plain words; matching normalises headings the
same way (numbering, punctuation and letter-spacing removed) before lookup.
Headings that ended up as "other" in scripts/eval_sections.py are the best
source of new synonyms.
"""

from __future__ import annotations

from jobpdf.extraction.models import SectionType

SECTION_KEYWORDS: dict[SectionType, list[str]] = {
    "contact": [
        "contact",
        "contacts",
        "contact information",
        "contact details",
        "personal details",
        "контакти",
        "контактна інформація",
    ],
    "summary": ["summary", "profile", "about me", "objective", "про себе"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "career history",
        "internships",
        "досвід роботи",
        "досвід",
    ],
    "education": ["education", "academic background", "qualifications", "освіта"],
    "skills": [
        "skills",
        "technical skills",
        "core competencies",
        "tech stack",
        "technologies",
        "tools",
        "навички",
        "ключові навички",
    ],
    "projects": ["projects", "personal projects", "проєкти"],
    "certifications": [
        "certifications",
        "certificates",
        "courses",
        "licenses",
        "сертифікати",
        "курси",
    ],
    "languages": ["languages", "мови"],
}
