from jobpdf.api.response import InDemand, Insights, SkillToLearn, WithinReach, insights_out
from jobpdf.ranking.insights import (
    InDemandItem,
    InsightsResult,
    SkillToLearnItem,
    WithinReachItem,
)
from jobpdf.ranking.models import SkillRef

KUBERNETES = SkillRef(concept_id="esco:k8s", raw="k8s", normalized="k8s", label="Kubernetes")
SPARK = SkillRef(raw="Spark", normalized="spark")  # unrecognised
PYTHON = SkillRef(concept_id="esco:py", raw="Python", normalized="python", label="Python")


def test_insights_out_maps_every_field() -> None:
    result = InsightsResult(
        working_set_size=12,
        within_reach=[WithinReachItem(vacancy_id="v1", added=[KUBERNETES, SPARK], fit_now=0.6,
                                      fit_after=0.8, required_matched_count=6,
                                      required_count=8)],
        skills_to_learn=[SkillToLearnItem(skill=KUBERNETES, gain=2, frequency=5,
                                          vacancy_ids=["v1", "v2"])],
        in_demand=[InDemandItem(skill=PYTHON, count=9)],
    )

    out = insights_out(result)

    assert out.computed is True
    assert out.working_set_size == 12
    assert out.within_reach == [WithinReach(
        vacancy_id="v1", missing=out.within_reach[0].missing, fit_now=0.6, fit_after=0.8,
        required_matched_count=6, required_count=8)]
    assert [(s.name, s.label, s.recognised) for s in out.within_reach[0].missing] == [
        ("k8s", "Kubernetes", True), ("Spark", None, False)]
    assert out.skills_to_learn == [SkillToLearn(
        skill=out.skills_to_learn[0].skill, unlocks=2, vacancy_ids=["v1", "v2"], required_in=5)]
    assert out.skills_to_learn[0].skill.label == "Kubernetes"
    assert out.in_demand == [InDemand(skill=out.in_demand[0].skill, count=9)]
    assert out.in_demand[0].skill.concept_id == "esco:py"


def test_empty_insights_are_still_computed() -> None:
    out = insights_out(InsightsResult(working_set_size=0))

    assert out == Insights(computed=True, working_set_size=0)


def test_the_default_is_not_computed_and_old_payloads_still_load() -> None:
    assert Insights().computed is False
    old = {"computed": True, "within_reach": [{"vacancy_id": "v", "missing": []}],
           "skills_to_learn": []}  # the JM-29 shape, without the JM-42 fields

    assert Insights.model_validate(old).within_reach[0].fit_after is None
