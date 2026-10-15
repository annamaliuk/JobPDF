"""The two JM-27 scripts, called as functions (their main() takes argv)."""

import importlib.util
import json
from pathlib import Path

import pytest

from jobpdf.ranking.learned import WEIGHTS_MISSING, load_learned_scorer, select_scorer
from tests.ranking.feedback_data import make_feedback

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def feedback_file(tmp_path: Path) -> Path:
    path = tmp_path / "feedback.jsonl"
    rows = make_feedback(0.2, sessions=100, per_session=12, seed=1)
    path.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    return path


def test_fit_weights_writes_a_loadable_file_and_prints_numbers_only(
    feedback_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "w.json"

    code = load_script("fit_weights").main(["--feedback", str(feedback_file), "--out", str(out)])

    printed = capsys.readouterr().out
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert "rows: 1200 used" in printed and "learned : skill=" in printed
    assert "s0" not in printed and "v0" not in printed  # no session or vacancy IDs echoed
    assert stored["n_rows"] == 1200
    assert load_learned_scorer(out).weights == stored["weights"]


def test_fit_weights_refuses_too_little_feedback(tmp_path: Path, capsys) -> None:
    path = tmp_path / "small.jsonl"
    rows = make_feedback(0.2, sessions=2, per_session=5, seed=1)
    path.write_text("\n".join(r.model_dump_json() for r in rows), encoding="utf-8")
    out = tmp_path / "w.json"

    code = load_script("fit_weights").main(["--feedback", str(path), "--out", str(out)])

    assert code == 1 and not out.exists()
    assert "not enough feedback" in capsys.readouterr().out


def test_scripts_report_a_bad_feedback_file_without_echoing_it(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"session":"a","vacancy_id":"v","label":1,"cv_text":"secret words"}\n',
                    encoding="utf-8")

    for name in ("fit_weights", "compare_scorers"):
        assert load_script(name).main(["--feedback", str(path)]) == 1

    printed = capsys.readouterr().out
    assert "invalid feedback row" in printed and "secret" not in printed


def test_scripts_report_a_missing_feedback_file(tmp_path: Path, capsys) -> None:
    for name in ("fit_weights", "compare_scorers"):
        assert load_script(name).main(["--feedback", str(tmp_path / "nope.jsonl")]) == 1

    assert "cannot read feedback" in capsys.readouterr().out


def test_compare_scorers_prints_a_verdict(feedback_file: Path, capsys) -> None:
    code = load_script("compare_scorers").main(["--feedback", str(feedback_file)])

    printed = capsys.readouterr().out
    assert code == 0
    assert "verdict: learned_better" in printed
    assert "NDCG@10" in printed and "95% interval" in printed


def test_without_running_fit_the_flag_stays_on_linear(monkeypatch, tmp_path: Path) -> None:
    # Nothing in the repo ships a weights file, so the learned scorer is opt-in twice:
    # the flag, and a file that fit_weights.py has to write.
    from jobpdf.ranking import learned

    assert not learned.WEIGHTS_PATH.exists()
    monkeypatch.setattr(learned, "WEIGHTS_PATH", tmp_path / "absent.json")
    with pytest.raises(learned.WeightsUnusable) as caught:
        learned.load_learned_scorer(learned.WEIGHTS_PATH)
    assert caught.value.code == WEIGHTS_MISSING
    assert type(select_scorer("learned")).__name__ == "LinearScorer"
