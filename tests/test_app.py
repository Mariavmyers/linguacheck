"""Offline checks: no API key, no network, no cost. Run: python -m pytest -q"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402
import judge  # noqa: E402

JUDGE_REPLY = json.dumps({"findings": [{
    "type": "fact_drift", "severity": "high",
    "en_quote": "Take 2.5 mL.", "pt_quote": "Tome 25 mL.",
    "explanation": "The dose is ten times higher in Portuguese.", "trap_id": None}],
    "consistent": False})


class FakeClient:
    """Pretends to be Baseten / Muse Spark Open: reasoning first, then the answer."""

    def __init__(self, answers=None, judge_reply=JUDGE_REPLY):
        self.calls = []
        self.answers = answers or {}
        self.judge_reply = judge_reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if kw["messages"][0]["role"] == "system":
            content = "<think>comparing</think>" + self.judge_reply
        else:
            q = kw["messages"][0]["content"]
            content = "<think>reasoning</think>" + self.answers.get(q, "Answer.")
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ---------------------------------------------------------------- data

def test_all_75_questions_load_and_are_complete():
    assert len(app.QUESTIONS) == 75
    for qid, q in app.QUESTIONS.items():
        assert q["question_en"] and q["question_pt"], qid
        assert q["domain"] in {"Medical", "Legal", "Finance"}


def test_trap_file_loads():
    assert len(judge.TRAPS) >= 75
    for d in ("Medical", "Legal", "Finance"):
        assert len(judge.traps_for(d)) >= 25
    assert all(t["correct_pt"] and t["trap_pt"] for t in judge.TRAPS)


# ---------------------------------------------------------------- model call

def test_reasoning_is_removed():
    msg = SimpleNamespace(content="<think>long reasoning</think>\nFour times a day.")
    assert app.final_answer_only(msg) == "Four times a day."
    msg = SimpleNamespace(content="unclosed reasoning </think> Quatro vezes ao dia.")
    assert app.final_answer_only(msg) == "Quatro vezes ao dia."


def test_full_audit_with_fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_FILE", tmp_path / "answers.jsonl")
    fake = FakeClient()
    en, pt, flags, status = app.run_audit("M03", use_traps=True, client=fake)
    assert len(fake.calls) == 3                      # EN, PT, judge
    assert all(c["temperature"] == 0 for c in fake.calls)
    assert all(c["model"] == "meta-models/muse-spark-open-1026" for c in fake.calls)
    assert fake.calls[0]["messages"][0]["content"] == app.QUESTIONS["M03"]["question_en"]
    assert fake.calls[1]["messages"][0]["content"] == app.QUESTIONS["M03"]["question_pt"]
    assert "INTERPRETER TRAP FILE" in fake.calls[2]["messages"][0]["content"]
    assert "not advice" in en and "not advice" in pt
    assert "1 flag" in flags and "Tome 25 mL." in flags and "High" in flags
    lines = (tmp_path / "answers.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[2])["kind"] == "judgment"


def test_trap_file_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_FILE", tmp_path / "answers.jsonl")
    fake = FakeClient()
    app.run_audit("L01", use_traps=False, client=fake)
    assert "INTERPRETER TRAP FILE" not in fake.calls[2]["messages"][0]["content"]


def test_missing_key_gives_plain_message(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "RESULTS_FILE", tmp_path / "answers.jsonl")
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    en, pt, flags, _ = app.run_audit("L01")
    assert "No API key found" in en and "No API key found" in pt


# ---------------------------------------------------------------- code checks

def test_brazilian_decimal_comma_is_not_flagged():
    cc = judge.code_checks("Take 2.5 mL twice a day.", "Tome 2,5 mL duas vezes ao dia.")
    assert not cc["numbers_only_in_en"] and not cc["numbers_only_in_pt"]


def test_brazilian_thousands_dot_is_not_flagged():
    cc = judge.code_checks("Pay $1,200 per month.", "Pague US$ 1.200 por mês.")
    assert not cc["numbers_only_in_en"] and not cc["numbers_only_in_pt"]


def test_ten_times_dose_is_flagged():
    cc = judge.code_checks("Take 2.5 mL.", "Tome 25 mL.")
    assert cc["numbers_only_in_en"] == ["2.5"] and cc["numbers_only_in_pt"] == ["25"]


def test_us_vs_brazilian_date_reading_is_flagged():
    cc = judge.code_checks("Do not use after March 4, 2027.",
                           "Não use depois de 3 de abril de 2027.")
    assert cc["dates_only_in_en"] == [(3, 4)] and cc["dates_only_in_pt"] == [(4, 3)]
    ok = judge.code_checks("Expires March 4, 2027.", "Vence em 4 de março de 2027.")
    assert not ok["dates_only_in_en"] and not ok["dates_only_in_pt"]


def test_list_markers_are_ignored():
    cc = judge.code_checks("(1) Dysgeusia. (2) Urine drug screen.",
                           "(1) Disgeusia. (2) Exame toxicológico de urina.")
    assert not cc["numbers_only_in_en"] and not cc["numbers_only_in_pt"]


# ---------------------------------------------------------------- judge reply parsing

def test_judge_json_is_found_inside_extra_text():
    parsed = judge.parse_judge_json("Here is my review:\n" + JUDGE_REPLY + "\nDone.")
    assert parsed["consistent"] is False and parsed["findings"][0]["pt_quote"] == "Tome 25 mL."


def test_bad_judge_reply_retries_then_reports():
    calls = []

    def chat(system, user):
        calls.append(user)
        return "I could not decide."
    q = app.QUESTIONS["M04"]
    result = judge.judge(q, "2.5 mL", "2,5 mL", chat)
    assert len(calls) == 2 and result.get("error")
