import pytest

from jobpdf.api.response import WarningOut, warning_out
from jobpdf.ranking.score import Band
from jobpdf.report.build import (
    ALL_MATCHES_ROWS,
    BEST_MATCHES,
    SOURCE_NOTE,
    build_report,
)
from jobpdf.report.models import Report
from tests.report.fixtures import (
    TODAY,
    insights,
    job,
    match,
    profile,
    response,
    skill,
)

B = Band


def report(**kwargs) -> Report:
    return build_report(response(**kwargs), today=TODAY)


# --- summary -----------------------------------------------------------------------


def test_the_full_summary() -> None:
    matches = [match(f"s{i}", B.STRONG, rank=i) for i in range(12)]
    matches += [match(f"p{i}", B.PARTIAL, rank=20 + i) for i in range(47)]
    r = report(matches=matches, hidden=253,
               insights=insights(learn=[("Kubernetes", 9, 18)]))

    assert r.headline == "12 remote jobs fit you strongly right now"
    assert r.headline_note is None
    assert r.summary == [
        "You have about 4½ years of experience, most recently as Data Engineer.",
        "We recognised 2 skills.",
        "Out of 312 current remote vacancies, 12 fit you strongly and 47 partly.",
        "The one skill that would open the most new options for you is Kubernetes.",
    ]
    assert [(t.value, t.label) for t in r.stats] == [
        ("12", "strong fits"), ("47", "partial fits"), ("2", "skills recognised"),
        ("4½ yrs", "of experience")]
    assert r.generated_on == TODAY and r.source_note == SOURCE_NOTE


def test_one_strong_match_is_singular() -> None:
    r = report(matches=[match("a")])

    assert r.headline == "1 remote job fits you strongly right now"
    assert "Out of 1 current remote vacancy, 1 fits you strongly and 0 partly." in r.summary
    assert (r.stats[0].value, r.stats[0].label) == ("1", "strong fit")


def test_zero_strong_matches_point_to_within_reach() -> None:
    r = report(matches=[match("p", B.PARTIAL, required=(["A", "B", "C"], ["D"]))],
               insights=insights(within=[("p", ["D"], 3, 4)]))

    assert r.headline == "No remote job fits you strongly yet"
    assert "Within reach" in r.headline_note
    assert r.best_matches == []  # no empty section
    assert r.strong_count == 0
    assert len(r.within_reach) == 1


def test_zero_strong_and_nothing_within_reach_has_no_pointer() -> None:
    r = report(matches=[match("w", B.WEAK)])

    assert r.headline == "No remote job fits you strongly yet"
    assert r.headline_note is None


def test_no_vacancies_says_so() -> None:
    r = report(matches=[])

    assert r.headline == "There are no vacancies to compare your CV with right now"
    assert r.best_matches == [] and r.all_matches == [] and r.all_matches_intro is None
    assert not any("Out of" in s for s in r.summary)


def test_no_recognised_skills_says_so() -> None:
    prof = profile(skills=[skill("Pipewright", recognised=False)], jobs=[job("Analyst")])
    r = report(prof=prof)

    assert "We couldn't recognise any skills in your CV" in r.summary[1]
    assert r.skills.recognised == []
    assert r.skills.empty_sentence == r.summary[1]
    assert r.stats[2].label == "skills recognised" and r.stats[2].value == "0"


def test_unknown_experience_and_role_leave_sentences_out() -> None:
    r = report(prof=profile(months=None, jobs=[]))

    assert r.summary[0] == "We recognised 2 skills."
    assert r.stats[3].value == "—"


def test_the_latest_role_is_the_current_one() -> None:
    jobs = [job("Intern", start="2015", end="2016"),
            job("Data Engineer", start="2021-03", end="present"),
            job("Analyst", start="2017", end="2021-02")]
    r = report(prof=profile(jobs=jobs))
    assert r.summary[0].endswith("most recently as Data Engineer.")

    jobs = [job("Analyst", start="2017", end="2021-02"), job("Intern", start="2015", end="2016")]
    r = report(prof=profile(jobs=jobs))
    assert r.summary[0].endswith("most recently as Analyst.")


# --- notes -------------------------------------------------------------------------


def test_each_warning_becomes_one_sentence_and_unknown_codes_are_kept() -> None:
    warnings = [warning_out("no_text_layer", "parse"), warning_out("no_text_layer", "parse"),
                warning_out("something_new")]
    r = report(warnings=warnings)

    assert r.notes == [
        "The PDF has no text layer (likely a scan), so little could be read.",
        "Something reduced the quality of this result.",
    ]


def test_no_warnings_no_notes() -> None:
    assert report(warnings=[]).notes == []


# --- matches -----------------------------------------------------------------------


def test_best_match_cards() -> None:
    m = match("a", required=(["Python", "SQL"], ["Kubernetes"]),
              preferred=(["dbt"], ["Kafka", "Terraform"]), semantic=0.45)
    card = report(matches=[m]).best_matches[0]

    assert (card.title, card.company, card.location) == ("Job a", "Co a", "Remote, Europe")
    assert card.posted == "2 days ago"
    assert card.band_name == "Strong fit"
    assert card.skills_sentence == (
        "You have 2 of the 3 required skills and 1 of the 3 nice-to-have ones.")
    assert card.required_short == "2 of 3"
    assert card.similarity == "medium"
    assert card.missing_required == ["Kubernetes"]
    assert card.missing_preferred == ["Kafka", "Terraform"]
    assert card.url == "https://remotive.com/remote-jobs/a"


def test_a_vacancy_without_required_skills() -> None:
    card = report(matches=[match("a", required=([], []))]).best_matches[0]

    assert card.skills_sentence == "This listing doesn't state required skills."
    assert card.required_short == "—"


def test_best_matches_are_strong_only_and_limited() -> None:
    matches = [match(f"s{i}", rank=i) for i in range(5)] + [match("p", B.PARTIAL, rank=9)]
    r = report(matches=matches)

    assert [c.vacancy_id for c in r.best_matches] == ["s0", "s1", "s2"]
    assert len(r.best_matches) == BEST_MATCHES
    assert r.strong_count == 5


def test_all_matches_lists_strong_and_partial_up_to_the_limit() -> None:
    matches = [match(f"v{i:03}", B.STRONG if i < 10 else B.PARTIAL, rank=i)
               for i in range(60)]
    matches.append(match("weak", B.WEAK, rank=99))
    r = report(matches=matches)

    assert len(r.all_matches) == ALL_MATCHES_ROWS
    assert "weak" not in {row.vacancy_id for row in r.all_matches}
    assert r.all_matches_intro == (
        "60 strong and partial matches, best first. Showing the first 50.")
    assert all(row.url for row in r.all_matches)
    assert r.all_matches[0].band_name == "Strong fit"
    assert r.all_matches[-1].band_name == "Partial fit"


def test_one_listed_match_intro() -> None:
    assert report(matches=[match("a")]).all_matches_intro == "1 strong or partial match."


def test_similarity_is_left_out_when_not_computed() -> None:
    assert report(matches=[match("a", semantic=None)]).best_matches[0].similarity is None


# --- insights ----------------------------------------------------------------------


def test_within_reach_cards() -> None:
    matches = [match("one", B.PARTIAL, required=(["A", "B", "C", "D"], ["E"])),
               match("two", B.PARTIAL, required=(["A", "B", "C", "D", "E", "F"],
                                                  ["G", "H"]))]
    ins = insights(within=[("one", ["E"], 4, 5), ("two", ["G", "H"], 6, 8)])
    cards = report(matches=matches, insights=ins).within_reach

    assert [c.vacancy_id for c in cards] == ["one", "two"]
    assert cards[0].sentence == (
        "You have 4 of the 5 required skills. Adding this one makes it a strong match.")
    assert cards[1].sentence == (
        "You have 6 of the 8 required skills. Adding these two makes it a strong match.")
    assert cards[1].add == ["G", "H"]
    assert (cards[0].band_now, cards[0].band_after) == ("Partial fit now", "Strong fit")
    assert cards[0].url == "https://remotive.com/remote-jobs/one"


def test_within_reach_for_a_vacancy_not_in_the_response_is_skipped() -> None:
    ins = insights(within=[("elsewhere", ["E"], 4, 5)])
    assert report(insights=ins).within_reach == []


def test_skills_worth_learning_and_in_demand() -> None:
    ins = insights(learn=[("Kubernetes", 9, 18), ("Terraform", 1, 1)],
                   demand=[("Python", 41), ("dbt", 1)])
    r = report(insights=ins)

    assert [(i.skill, i.required_in, i.unlocks) for i in r.skills_to_learn] == [
        ("Kubernetes", "Required in 18 of your 50 closest vacancies", "+9 strong matches"),
        ("Terraform", "Required in 1 of your 50 closest vacancies", "+1 strong match"),
    ]
    assert [(i.skill, i.text) for i in r.in_demand] == [("Python", "41 of 50"),
                                                         ("dbt", "1 of 50")]


def test_insights_without_a_working_set_size() -> None:
    r = report(insights=insights(learn=[("Kubernetes", 2, 3)], demand=[("Python", 4)],
                                 size=None))

    assert r.skills_to_learn[0].required_in == "Required in 3 vacancies"
    assert r.in_demand[0].text == "4"


def test_insights_not_computed_give_empty_sections() -> None:
    r = report()

    assert r.within_reach == [] and r.skills_to_learn == [] and r.in_demand == []


# --- your skills -------------------------------------------------------------------


def test_your_skills_readings_and_unrecognised() -> None:
    prof = profile(
        skills=[skill("GBQ", "Google BigQuery"), skill("python", "Python"),
                skill("Postgres", "PostgreSQL"), skill("Pipewright", recognised=False),
                skill("guessy", "Guess", low_confidence=True)],
        jobs=[job("Data Engineer", skills=[skill("GBQ", "Google BigQuery"), skill("dbt")])],
    )
    s = report(prof=prof).skills

    assert s.recognised == ["Google BigQuery", "Python", "PostgreSQL", "Guess", "dbt"]
    assert [(x.written, x.read_as) for x in s.readings] == [
        ("GBQ", "Google BigQuery"), ("Postgres", "PostgreSQL")]  # case-only and low confidence
    assert s.readings_sentence == (
        "We read “GBQ” as Google BigQuery and “Postgres” as PostgreSQL, "
        "so vacancies that use the standard names still count.")
    assert s.unrecognised == ["Pipewright"]
    assert s.unrecognised_sentence.startswith("We couldn't match this to a known skill")
    assert s.empty_sentence is None


def test_no_readings_no_sentence() -> None:
    s = report().skills
    assert s.readings == [] and s.readings_sentence is None and s.unrecognised_sentence is None


# --- how we score fit --------------------------------------------------------------


def test_scoring_text_uses_the_real_numbers() -> None:
    text = " ".join(report().scoring)

    assert "About 70% comes from how many of a vacancy's required skills" in text
    assert "nice-to-have skills count half as much" in text
    assert "The other 30% is how similar your experience" in text
    assert "at least 80% of the required skills" in text
    assert "at least 50% of them" in text


def test_scoring_text_follows_other_weights() -> None:
    text = " ".join(report(weights={"skill": 0.55, "semantic": 0.45}).scoring)
    assert "About 55%" in text and "The other 45%" in text

    skills_only = " ".join(report(weights={"skill": 1.0, "semantic": 0.0}).scoring)
    assert "Fit comes from how many of a vacancy's required skills" in skills_only


# --- privacy -----------------------------------------------------------------------


def test_contact_data_never_reaches_the_report() -> None:
    email, phone = "ann@example.com", "+380 67 123 4567"
    prof = profile(
        summary=f"Reach me at {email} or {phone}.",
        skills=[skill(f"Python {email}", "Python"), skill(email, recognised=False)],
        jobs=[job(f"Data Engineer {email}", company=f"Acme {phone}",
                  skills=[skill(phone, recognised=False)])],
    )
    r = report(prof=prof, warnings=[WarningOut(code="possible_personal_data",
                                               detail="Extracted details may contain "
                                                      "personal data.")])
    dumped = r.model_dump_json()

    assert email not in dumped and "example.com" not in dumped
    assert "123 4567" not in dumped and "+380" not in dumped
    assert r.summary[0].endswith("most recently as Data Engineer.")
    assert r.skills.unrecognised == []  # only contact data: dropped


def test_the_report_round_trips_through_json() -> None:
    r = report(insights=insights(within=[("a", ["X"], 1, 2)], learn=[("X", 1, 1)]))

    assert Report.model_validate_json(r.model_dump_json()) == r


@pytest.mark.parametrize("band", list(Band))
def test_every_band_has_a_name(band) -> None:
    from jobpdf.report.build import BAND_NAMES

    assert BAND_NAMES[band]
