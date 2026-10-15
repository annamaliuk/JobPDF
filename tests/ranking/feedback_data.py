"""Synthetic feedback with a known true weight, shared by the JM-27 tests."""

from __future__ import annotations

import math

import numpy as np

from jobpdf.ranking.learned import FeedbackRow


def make_feedback(
    true_skill_weight: float,
    sessions: int,
    per_session: int,
    seed: int,
    slope: float = 8.0,
) -> list[FeedbackRow]:
    """Cards whose chance of being useful rises with
    ``true_skill_weight x skill + (1 - true_skill_weight) x semantic``.

    The labels come from that rule, never from the scorer under test, so a scorer
    can only do well by finding the rule.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(sessions):
        for v in range(per_session):
            skill, semantic = round(float(rng.random()), 3), round(float(rng.random()), 3)
            x = true_skill_weight * skill + (1.0 - true_skill_weight) * semantic
            useful = rng.random() < 1.0 / (1.0 + math.exp(-slope * (x - 0.5)))
            rows.append(FeedbackRow(session=f"s{s}", vacancy_id=f"v{v}", skill=skill,
                                    semantic=semantic, label=int(useful)))
    return rows
