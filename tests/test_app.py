"""Offline checks: no API key, no network, no cost. Run: python -m pytest -q"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402


class FakeClient:
    """Pretends to be Baseten. Returns reasoning + final answer, like Muse Spark Open."""

    def __init__(self, content):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._content = content

    def _create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content=self._content, reasoning_content="thinking...")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_all_75_questions_load_and_are_complete():
    assert len(app.QUESTIONS) == 75
    for qid, q in app.QUESTIONS.items():
        assert q["question_en"] and q["question_pt"], qid
        assert q["domain"] in {"Medical", "Legal", "Finance"}


def test_reasoning_is_removed():
    msg = SimpleNamespace(content="<think>long reasoning</think>\nFour times a day.")
    assert app.final_answer_only(msg) == "Four times a day."
    msg = SimpleNamespace(content="unclosed reasoning </think> Quatro vezes ao dia.")
    assert app.final_answer_only(msg) == "Quatro vezes ao dia."


def test_settings_sent_to_model(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_FILE", tmp_path / "answers.jsonl")
    fake = FakeClient("<think>x</think>Answer")
    en, pt, status = app.run_audit("M03", client=fake)
    assert len(fake.calls) == 2                      # once in EN, once in PT
    assert all(c["temperature"] == 0 for c in fake.calls)
    assert all(c["model"] == "meta-models/muse-spark-open-1026" for c in fake.calls)
    assert fake.calls[0]["messages"][0]["content"] == app.QUESTIONS["M03"]["question_en"]
    assert fake.calls[1]["messages"][0]["content"] == app.QUESTIONS["M03"]["question_pt"]
    assert "not advice" in en and "not advice" in pt
    assert len((tmp_path / "answers.jsonl").read_text().splitlines()) == 2


def test_missing_key_gives_plain_message(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "RESULTS_FILE", tmp_path / "answers.jsonl")
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    en, pt, _ = app.run_audit("L01")
    assert "No API key found" in en and "No API key found" in pt
