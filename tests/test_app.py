"""Offline checks: no API key, no network, no cost. Run: python -m pytest -q"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402
import audit  # noqa: E402
import judge  # noqa: E402
import storage  # noqa: E402

EN_ANSWER = "Take 2.5 mL twice a day. Call **right away** if you soak more than one pad in an hour."
PT_ANSWER = "Tome 25 mL duas vezes ao dia. Ligue se encharcar um absorvente por 2 horas seguidas ou mais."


def judge_reply(findings):
    return json.dumps({"findings": findings})


DEFAULT_FINDINGS = [
    {"category": "changed_value", "severity": "high", "problem_language": "PT",
     "en_quote": "Take 2.5 mL twice a day", "pt_quote": "Tome 25 mL duas vezes ao dia",
     "explanation": "Dose is ten times higher in PT.", "consequence": "Overdose.", "trap_id": None},
    {"category": "harmless_variation", "severity": "low", "problem_language": "unclear",
     "en_quote": "Call right away", "pt_quote": "Ligue",
     "explanation": "Wording differs.", "consequence": "", "trap_id": None},
    {"category": "contradiction", "severity": "high", "problem_language": "PT",
     "en_quote": "This sentence is not in the answer", "pt_quote": "(missing)",
     "explanation": "Invented evidence.", "consequence": "", "trap_id": None},
]


class FakeClient:
    """Pretends to be Baseten / Muse Spark Open: reasoning first, then the answer."""

    def __init__(self, answers=None, findings=None):
        self.calls = []
        self.answers = answers or {}
        self.findings = DEFAULT_FINDINGS if findings is None else findings
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if kw["messages"][0]["role"] == "system":
            content = "<think>comparing</think>" + judge_reply(self.findings)
        else:
            q = kw["messages"][0]["content"]
            is_pt = any(w in q for w in ("Uma ", "Um ", "Nos EUA", "Em um"))
            content = "<think>reasoning</think>" + self.answers.get(q, PT_ANSWER if is_pt else EN_ANSWER)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def local_storage(tmp_path, monkeypatch):
    """Every test uses a temporary local folder, never the real dataset."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(storage, "LOCAL", tmp_path / "local_data")
    monkeypatch.setattr(storage, "_reviews_cache", None)
    monkeypatch.setattr(app, "HERE", tmp_path)
    yield


# ---------------------------------------------------------------- data

def test_all_75_questions_load_and_are_complete():
    assert len(app.QUESTIONS) == 75
    for qid, q in app.QUESTIONS.items():
        assert q["question_en"] and q["question_pt"], qid


def test_trap_file_loads():
    assert len(judge.TRAPS) >= 74
    assert all(t["correct_pt"] and t["trap_pt"] for t in judge.TRAPS)


def test_reasoning_is_removed():
    assert app.final_answer_only(SimpleNamespace(content="<think>x</think>\nFour.")) == "Four."
    assert app.final_answer_only(SimpleNamespace(content="x </think> Quatro.")) == "Quatro."


# ---------------------------------------------------------------- evidence integrity

def test_quote_found_despite_markdown_punctuation_and_case():
    assert judge.quote_spans("call RIGHT AWAY if you soak more than one pad in an hour.", EN_ANSWER)


def test_invented_quote_is_unsupported():
    f = judge.validate_finding({"en_quote": "Never take this drug", "pt_quote": "(missing)"},
                               EN_ANSWER, PT_ANSWER)
    assert f["support"] == "unsupported" and f["en_found"] is False


def test_fragments_joined_by_ellipsis_must_all_exist_in_order():
    assert judge.quote_spans("Take 2.5 mL ... soak more than one pad", EN_ANSWER)
    assert judge.quote_spans("soak more than one pad ... Take 2.5 mL", EN_ANSWER) is None


def test_both_sides_missing_is_unsupported():
    f = judge.validate_finding({"en_quote": "(missing)", "pt_quote": "(missing)"}, EN_ANSWER, PT_ANSWER)
    assert f["support"] == "unsupported"


def test_duplicates_are_removed_and_ids_are_stable():
    q = app.QUESTIONS["M04"]
    dup = [DEFAULT_FINDINGS[0], dict(DEFAULT_FINDINGS[0], explanation="same thing again")]
    r1 = judge.judge(q, EN_ANSWER, PT_ANSWER, lambda s, u: judge_reply(dup))
    r2 = judge.judge(q, EN_ANSWER, PT_ANSWER, lambda s, u: judge_reply(dup))
    assert len(r1["findings"]) == 1
    assert r1["findings"][0]["id"] == r2["findings"][0]["id"]


def test_only_supported_issue_categories_count_as_flags():
    q = app.QUESTIONS["M04"]
    res = judge.judge(q, EN_ANSWER, PT_ANSWER, lambda s, u: judge_reply(DEFAULT_FINDINGS))
    counted = [f for f in res["findings"] if judge.is_issue(f)]
    assert [f["category"] for f in counted] == ["changed_value"]


def test_old_categories_are_mapped_not_lost():
    parsed = judge.parse_judge_json(json.dumps({"findings": [
        {"type": "missing", "severity": "high", "en_quote": "a b", "pt_quote": "(missing)",
         "explanation": "x"}]}))
    assert parsed["findings"][0]["category"] == "consequential_omission"


def test_judge_json_with_unescaped_quotes_is_repaired():
    bad = ('{"findings": [{"category": "contradiction", "severity": "high", "en_quote": "x y", '
           '"pt_quote": "Ela disse "pare" agora", "explanation": "x", "trap_id": null}]}')
    assert judge.parse_judge_json(bad)["findings"][0]["severity"] == "high"


def test_unreadable_judge_reply_is_kept_for_debugging():
    result = judge.judge(app.QUESTIONS["M04"], "2.5 mL", "2,5 mL", lambda s, u: "")
    assert result["error"] and result["raw"] == "(empty reply)"


def test_prompt_rules_against_overclassification():
    system, _ = judge.build_judge_prompt(app.QUESTIONS["L14"], "a", "b", "Legal", True,
                                         judge.code_checks("a", "b"))
    assert "Extra information is NOT a contradiction" in system
    assert "do not treat English as the reference" in system
    assert "INTERPRETER TRAP FILE" in system
    system2, _ = judge.build_judge_prompt(app.QUESTIONS["L14"], "a", "b", "Legal", False,
                                          judge.code_checks("a", "b"))
    assert "INTERPRETER TRAP FILE" not in system2


# ---------------------------------------------------------------- code checks

def test_brazilian_decimal_comma_and_thousands_dot_are_not_flagged():
    cc = judge.code_checks("Take 2.5 mL. Pay $1,200.", "Tome 2,5 mL. Pague US$ 1.200.")
    assert not cc["numbers_only_in_en"] and not cc["numbers_only_in_pt"]


def test_ten_times_dose_and_date_order_are_flagged():
    cc = judge.code_checks("Take 2.5 mL.", "Tome 25 mL.")
    assert cc["numbers_only_in_en"] == ["2.5"] and cc["numbers_only_in_pt"] == ["25"]
    cc = judge.code_checks("Do not use after March 4, 2027.", "Não use depois de 3 de abril de 2027.")
    assert cc["dates_only_in_en"] == [(3, 4)]


# ---------------------------------------------------------------- app: one question

def test_single_question_with_fake_model():
    fake = FakeClient()
    en, pt, flags, status = app.run_audit("M04", use_traps=True, client=fake)
    assert len(fake.calls) == 3 and all(c["temperature"] == 0 for c in fake.calls)
    assert "not advice" in en and "not advice" in pt
    assert "1 possible difference" in flags and "not yet reviewed" in flags
    assert "unsupported" in flags            # invented quote shown separately, not counted


def test_missing_key_gives_plain_message(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    en, pt, _, _ = app.run_audit("L01")
    assert "No API key found" in en


# ---------------------------------------------------------------- runs, reviews, metrics

def wait_job():
    for _ in range(200):
        if not app.JOB["running"]:
            return
        time.sleep(0.05)


def make_legacy_run():
    recs = [{"id": i, "domain": app.QUESTIONS[i]["domain"], "answer_en": EN_ANSWER,
             "answer_pt": PT_ANSWER, "with_traps": {"findings": []}} for i in ("M04", "L09", "F10")]
    return "\n".join(json.dumps(r) for r in recs)


def test_import_then_rejudge_same_answers_then_review_and_metrics():
    msg = app.import_run(make_legacy_run(), "import_test")
    assert "Imported 3" in msg
    app.start_rejudge("import_test", client=FakeClient())
    wait_job()
    assert "Finished" in app.job_status()
    runs = [r for r in storage.list_runs() if r.startswith("rejudge")]
    st = app.load_run_state(runs[0])
    assert len(st["records"]) == 3
    assert st["meta"]["answers_from_run"] == "import_test"
    assert st["records"][0]["answer_en"] == EN_ANSWER          # same saved answers, untouched

    m = audit.metrics(st["records"], storage.load_reviews())
    w = m["modes"]["with"]
    assert w["total_flags"] == 3 and w["questions_with_flags"] == 3
    assert w["unsupported_quotes"] == 3 and w["harmless_variation"] == 3
    assert w["precision"] is None and w["reviewed"] == 0

    fids = [f["id"] for r in st["records"] for f in r["with_traps"]["findings"] if judge.is_issue(f)]
    app.save_decision(st, fids[0], "Confirmed issue", "changed_value", "high", "PT",
                      "10x dose", "", "Tome 2,5 mL")
    app.save_decision(st, fids[1], "False alarm / acceptable variation", "changed_value", "low",
                      "unclear", "fine", "", "")
    app.save_decision(st, fids[2], "Needs specialist review", "changed_value", "high", "PT", "", "", "")
    storage._reviews_cache = None                               # reload from disk = permanent
    m = audit.metrics(st["records"], storage.load_reviews())
    w = m["modes"]["with"]
    assert (w["confirmed"], w["false_alarms"], w["needs_specialist"]) == (1, 1, 1)
    assert w["precision"] == 0.5 and w["adjudicated"] == 2       # specialist not in denominator
    # the same evidence in the WITHOUT judgment inherits the decision
    assert m["modes"]["without"]["confirmed"] == 1
    md = audit.metrics_markdown(m)
    assert "Recall: not measured" in md and "not proven errors" in md

    files, note = app.export_all(st)
    names = [Path(f).name for f in files]
    assert any(n.startswith("expert_corrections") for n in names)
    assert any(n.endswith(".pdf") for n in names), note
    corr = [json.loads(x) for x in Path([f for f in files if "corrections" in f][0]).read_text().splitlines()]
    assert len(corr) == 1 and corr[0]["expert_correct_version"] == "Tome 2,5 mL"


def test_review_queue_filters_and_evidence_highlight():
    app.import_run(make_legacy_run(), "import_q")
    app.start_rejudge("import_q", client=FakeClient())
    wait_job()
    st = app.load_run_state([r for r in storage.list_runs() if r.startswith("rejudge")][0])
    upd, info = app.refresh_queue(st, "With trap file", "Not reviewed", "low")
    assert "3 finding" in info
    fid = upd["value"]
    out = app.show_finding(st, fid)
    assert "<mark" in out[0] and "Tome 25 mL duas vezes ao dia" in out[0]
    assert "temperature" in out[0] and fid in out[0]


def test_full_run_saves_answers_and_both_judgments():
    app.QUESTIONS_BACKUP = app.QUESTIONS
    try:
        app.QUESTIONS = {k: app.QUESTIONS_BACKUP[k] for k in ("M04", "L09")}
        app.start_full_run(client=FakeClient())
        wait_job()
    finally:
        app.QUESTIONS = app.QUESTIONS_BACKUP
    run = [r for r in storage.list_runs() if r.startswith("full")][0]
    recs, meta = storage.load_run(run)
    assert len(recs) == 2 and meta["complete"] is True and meta["temperature"] == 0
    assert recs[0]["answer_en"] and recs[0]["with_traps"]["judge_version"] == judge.JUDGE_VERSION
    assert "without_traps" in recs[0]
