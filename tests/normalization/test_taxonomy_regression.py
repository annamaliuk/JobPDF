"""Must-have tech skills resolve in the real built taxonomy.

Runs only where scripts/build_taxonomy.py has produced data/taxonomy/aliases.csv.
If a skill fails here, the fix is a row in data/taxonomy/custom_skills.csv,
not a change to this list.
"""

import csv
from pathlib import Path

import pytest

from jobpdf.normalization.text import normalize

ALIASES = Path(__file__).parents[2] / "data" / "taxonomy" / "aliases.csv"

MUST_HAVE = [
    # languages
    "python", "sql", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
    "php", "ruby", "scala", "kotlin", "swift", "r", "matlab", "html", "css",
    # frameworks and runtimes
    "react", "angular", "vue.js", "node.js", "django", ".net",
    # data stores
    "postgresql", "mysql", "mongodb", "nosql", "redis",
    # data engineering
    "spark", "hadoop", "kafka", "airflow", "dbt", "bigquery", "pandas",
    # infrastructure
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible", "jenkins",
    "linux", "git", "devops",
    # data and ML
    "machine learning", "deep learning", "nlp", "data science", "statistics",
    "excel", "tableau", "power bi",
    # ways of working
    "agile", "scrum", "jira",
]

pytestmark = [
    pytest.mark.taxonomy,
    pytest.mark.skipif(not ALIASES.is_file(), reason="run scripts/build_taxonomy.py first"),
]


@pytest.fixture(scope="module")
def alias_norms() -> set[str]:
    with ALIASES.open(encoding="utf-8", newline="") as f:
        return {row["alias_norm"] for row in csv.DictReader(f)}


@pytest.mark.parametrize("skill", MUST_HAVE)
def test_must_have_skill_resolves(skill: str, alias_norms: set[str]) -> None:
    assert normalize(skill) in alias_norms
