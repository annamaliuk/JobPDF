"""JM-43: the page's files must agree with each other and with the rules the page promises
(nothing inline, nothing from another site, no request from the script)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from jobpdf.ranking.score import Band
from jobpdf.web import routes
from jobpdf.web.routes import STATIC_DIR, TEMPLATE_DIR

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = sorted(TEMPLATE_DIR.glob("*.html"))
STATIC = sorted(p for p in STATIC_DIR.iterdir() if p.is_file())
CSS = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
JS = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_page_files_exist() -> None:
    assert [p.name for p in TEMPLATES] == ["base.html", "error.html", "index.html", "report.html"]
    assert {p.name for p in STATIC} == {"app.js", "favicon.svg", "style.css"}


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_a_template_has_no_inline_script_style_or_handler(template) -> None:
    html = template.read_text(encoding="utf-8")

    assert not re.search(r"<script(?![^>]*\bsrc=)", html, re.I), "inline <script>"
    assert not re.search(r"<style\b", html, re.I), "inline <style>"
    assert not re.search(r"\sstyle\s*=", html, re.I), "style attribute"
    assert not re.search(r"\son[a-z]+\s*=", html, re.I), "event handler attribute"
    assert "javascript:" not in html.lower()


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_a_template_never_turns_escaping_off(template) -> None:
    html = template.read_text(encoding="utf-8")

    assert "|safe" not in html.replace(" ", "") and "autoescape false" not in html
    assert "Markup(" not in html


@pytest.mark.parametrize("path", [*TEMPLATES, *STATIC], ids=lambda p: p.name)
def test_nothing_loads_from_another_site(path) -> None:
    text = path.read_text(encoding="utf-8")
    allowed = ("http://www.w3.org/2000/svg",)  # an XML namespace, not a request

    for url in re.findall(r"(?:https?:)?//[A-Za-z0-9.-]+\.[a-z]{2,}[^\s\"')]*", text):
        assert url.startswith(allowed), f"{path.name} refers to {url}"
    assert "@import" not in text and "url(http" not in text.replace(" ", "")


def test_the_script_makes_no_requests_and_keeps_nothing() -> None:
    for word in ("fetch(", "XMLHttpRequest", "sendBeacon", "WebSocket", "EventSource",
                 "localStorage", "sessionStorage", "indexedDB", "document.cookie", "eval(",
                 "innerHTML", "new Function", "import("):
        assert word not in JS, word


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_script_is_valid_javascript() -> None:
    result = subprocess.run(["node", "--check", str(STATIC_DIR / "app.js")],
                            capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stderr


def test_every_static_file_a_page_names_exists() -> None:
    names = set()
    for template in TEMPLATES:
        names |= set(re.findall(r"/static/([A-Za-z0-9._-]+)", template.read_text("utf-8")))

    assert names == {"style.css", "app.js", "favicon.svg"}
    for name in names:
        assert (STATIC_DIR / name).is_file()


def test_every_template_compiles_and_only_uses_what_the_pages_pass() -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), undefined=StrictUndefined,
                      autoescape=True)

    for template in TEMPLATES:
        env.get_template(template.name)


@pytest.mark.parametrize("band", [b.value for b in Band])
def test_every_band_has_a_style(band) -> None:
    assert re.search(rf"\.band\.{band}\b", CSS), band


def test_the_meter_does_not_rely_on_colour_alone() -> None:
    assert re.search(r"\.meter i\.on", CSS) and re.search(r"\.meter i\.off", CSS)
    on = re.search(r"\.meter i\.on\s*\{([^}]*)\}", CSS).group(1)
    off = re.search(r"\.meter i\.off\s*\{([^}]*)\}", CSS).group(1)
    assert on != off and ("border" in off or "background" in off)


def test_the_stylesheet_respects_dark_mode_and_print() -> None:
    assert "prefers-color-scheme: dark" in CSS and "@media print" in CSS
    assert ":focus-visible" in CSS


def test_the_package_ships_its_templates_and_static_files() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert not any(line.strip().strip("/").startswith(("jobpdf/web", "**/*.html", "*.html",
                                                       "*.css", "*.js"))
                   for line in dockerignore)
    assert "COPY jobpdf ./jobpdf" in dockerfile
    assert (routes.PACKAGE_DIR / "__init__.py").is_file()


# --- wiring ------------------------------------------------------------------------------------


def test_jinja2_is_a_declared_dependency() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M).group(1)

    assert re.search(r'"jinja2>=3\.\d+"', dependencies)


@pytest.fixture(scope="module")
def compose() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def test_the_app_reads_the_visitor_address_from_the_proxy_only_while_it_is_not_published(
        compose) -> None:
    app = compose["app"]

    assert app["environment"]["FORWARDED_ALLOW_IPS"] == "*"
    assert "ports" not in app and app["expose"] == ["8000"]  # "*" is safe only then


def test_the_proxy_is_the_only_public_entry(compose) -> None:
    published = {name: s["ports"] for name, s in compose.items() if "ports" in s}

    assert published["proxy"] == ["80:80", "443:443"]
    assert "app" not in published and "monitor" not in published


def test_the_readme_describes_the_page_and_its_limits() -> None:
    section = README.split("### The web page")[1].split("\n---\n")[0]

    for text in ("5 MB", "Google Gemini", "`FORWARDED_ALLOW_IPS`", "no-store",
                 "5 uploads an hour and 20 a day"):
        assert text in section, text
    assert "`GET /`, `POST /report`" in README
    assert "You have uploaded a lot of CVs" in README
    assert "FORWARDED_ALLOW_IPS" in README.split("## 6. Settings")[1]


def test_the_readme_limits_match_the_code() -> None:
    from jobpdf.api.access import IP_LIMITS

    assert [(r.limit, r.seconds) for r in IP_LIMITS] == [(5, 3600), (20, 86400)]
