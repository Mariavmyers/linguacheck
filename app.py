"""
LinguaCheck: does Meta's AI say the same thing in English and Brazilian Portuguese?

Model under audit: Muse Spark Open (Meta open weights) served by Baseten.
Keys come only from environment variables / Space Secrets, never from code:
  BASETEN_API_KEY   the model
  HF_TOKEN          writes runs and the interpreter's reviews to the PRIVATE dataset
                    (DATA_REPO, default hugmarie/linguacheck-data)

Tabs:
  ★  Confirmed errors       expert-confirmed cases (the public results page)
  1. Ask one question         live EN vs PT-BR answers + judge
  2. Review findings          evidence screen + the interpreter's decisions (saved permanently)
  3. Results                  honest metrics, trap-file comparison, exports, PDF
  4. Runs                     full audit of all questions, re-judge saved answers
"""

import csv
import html
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

import audit
import judge
import storage

load_dotenv()

BASE_URL = os.getenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1")
MODEL = os.getenv("BASETEN_MODEL", "meta-models/muse-spark-open-1026")
MODEL_LABEL = "Muse Spark Open (Meta, open weights) via Baseten"
DISCLAIMER = "⚠️ Model output under audit — not advice."
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "16000"))  # room for the model's reasoning + answer
BATCH_WORKERS = int(os.getenv("BATCH_WORKERS", "8"))

HERE = Path(__file__).parent
QUESTIONS_FILE = HERE / "data" / "questions.csv"


# ---------------------------------------------------------------- data

def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"].strip() == "Verified"]
    return {r["id"]: r for r in rows}


QUESTIONS = load_questions()


def choice_label(q):
    text = q["question_en"]
    short = text if len(text) <= 90 else text[:87] + "..."
    return f"{q['id']} · {q['domain']} · {q['category']} — {short}"


CHOICES = [(choice_label(q), qid) for qid, q in QUESTIONS.items()]


# ---------------------------------------------------------------- model

def final_answer_only(message):
    """The model reasons first. Keep only the final answer."""
    text = (getattr(message, "content", None) or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1].strip()
    return text


def get_client():
    from openai import OpenAI

    key = os.getenv("BASETEN_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "No API key found. Put BASETEN_API_KEY=your-key in the .env file "
            "(or in the Space's Secrets) and restart the app."
        )
    return OpenAI(api_key=key, base_url=BASE_URL)


def chat(messages, client=None):
    """Send messages to the Meta model at temperature 0, return the final answer only."""
    client = client or get_client()
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0, max_tokens=MAX_TOKENS,
    )
    return final_answer_only(resp.choices[0].message)


def ask_model(question, client=None):
    """Send one question, return (final answer, seconds taken)."""
    start = time.time()
    answer = chat([{"role": "user", "content": question}], client=client)
    if not answer:
        answer = "(The model returned reasoning but no final answer. Try again.)"
    return answer, round(time.time() - start, 1)


def judge_chat(client=None):
    """The judge uses the same Meta model, with a system message."""
    def _chat(system, user):
        return chat([{"role": "system", "content": system},
                     {"role": "user", "content": user}], client=client)
    return _chat


def answer_and_judge(q, client=None):
    """New EN + PT answers, then the judge WITH and WITHOUT the trap file."""
    en, _ = ask_model(q["question_en"], client=client)
    pt, _ = ask_model(q["question_pt"], client=client)
    return audit.judge_record(q, en, pt, judge_chat(client))


audit_question = answer_and_judge  # kept for run_all.py


# ---------------------------------------------------------------- rendering

def esc(s):
    return html.escape(str(s or "")).replace("\n", "<br>")


SEV_ICON = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟡 Low"}
CAT_PRIORITY = {"contradiction": 0, "changed_value": 1, "consequential_omission": 2,
                "terminology_register": 3, "insufficient_evidence": 4, "harmless_variation": 5}
SEV_PRIORITY = {"high": 0, "medium": 1, "low": 2}


def highlight(answer, quote):
    """Answer as plain text with the quoted passage(s) marked. Never edits the words."""
    text = judge._plain(answer)
    spans = judge.quote_spans(quote, answer) or []
    out, pos = [], 0
    for a, b in spans:
        out.append(esc(text[pos:a]))
        out.append(f"<mark style='background:#fde68a;color:#111;padding:0 2px'>{esc(text[a:b])}</mark>")
        pos = b
    out.append(esc(text[pos:]))
    return "".join(out)


def finding_card(f, compact=False):
    trap = f" · trap {esc(f['trap_id'])}" if f.get("trap_id") else ""
    support = ("" if f.get("support") == "supported"
               else " · ⚠️ quote not found in the answer (unsupported)")
    lang = f.get("problem_language", "unclear")
    body = (f"<b>{SEV_ICON.get(f['severity'], f['severity'])} · "
            f"{esc(judge.CATEGORIES.get(f['category'], f['category']))}{trap}{support}</b>"
            f"<br>{esc(f['explanation'])}"
            + (f"<br><i>Possible consequence:</i> {esc(f.get('consequence'))}" if f.get("consequence") else "")
            + f"<br><small>Judge thinks the problem is in: {esc(lang)}</small>")
    if compact:
        return body
    return (
        "<div style='border:1px solid #8884;border-radius:8px;padding:10px;margin:8px 0'>"
        f"{body}<div style='display:flex;gap:10px;margin-top:8px;flex-wrap:wrap'>"
        f"<div style='flex:1;min-width:220px;background:#3b82f61a;padding:8px;border-radius:6px'>"
        f"<b>EN says</b><br>{esc(f['en_quote'])}</div>"
        f"<div style='flex:1;min-width:220px;background:#22c55e1a;padding:8px;border-radius:6px'>"
        f"<b>PT-BR says</b><br>{esc(f['pt_quote'])}</div></div></div>")


def render_judgment(result):
    if result.get("error"):
        return f"⚠️ Judge problem: {esc(result['error'])}"
    traps_note = "with the interpreter trap file" if result["use_traps"] else "WITHOUT the trap file"
    issues = [f for f in result["findings"] if judge.is_issue(f)]
    other = [f for f in result["findings"] if not judge.is_issue(f)]
    head = (f"<h3>{'✅ No meaningful difference flagged' if not issues else f'🚩 {len(issues)} possible difference(s)'}"
            f" <small>(automated judge: {esc(MODEL_LABEL)}, {traps_note}; not yet reviewed)</small></h3>")
    issues.sort(key=lambda f: (CAT_PRIORITY.get(f["category"], 9), SEV_PRIORITY.get(f["severity"], 9)))
    cards = "".join(finding_card(f) for f in issues)
    extra = ""
    if other:
        extra = ("<details><summary>Not counted ({}): harmless variation, insufficient evidence "
                 "or quotes not found</summary>{}</details>").format(
            len(other), "".join(finding_card(f) for f in other))
    code = (f"<details><summary>Code check (numbers and dates)</summary>"
            f"{esc(judge.code_checks_text(result['code_checks']))}</details>")
    return head + cards + extra + code


# ---------------------------------------------------------------- tab 1: one question

def show_question(qid):
    q = QUESTIONS.get(qid)
    if not q:
        return "", "", ""
    key = (
        f"**Interpreter answer key (verified by a 20-year medical/legal interpreter)**\n\n"
        f"- **Critical facts that must match:** {q['critical_facts']}\n"
        f"- **Likely AI trap:** {q['likely_trap']}\n"
        f"- **Risk if wrong:** {q['risk']}"
    )
    return q["question_en"], q["question_pt"], key


def run_audit(qid, use_traps=True, client=None):
    q = QUESTIONS.get(qid)
    if not q:
        msg = "Pick a question first."
        return msg, msg, "", ""
    out, answers = {}, {}
    for lang, field in (("EN", "question_en"), ("PT-BR", "question_pt")):
        try:
            answer, secs = ask_model(q[field], client=client)
            answers[lang] = answer
            out[lang] = f"{DISCLAIMER}\n\n{answer}\n\n*({secs} s)*"
        except Exception as e:  # show the problem in plain words, don't crash
            out[lang] = f"❌ Could not get an answer: {e}"
    if len(answers) < 2:
        return out["EN"], out["PT-BR"], "", "Could not compare: one answer is missing."
    try:
        result = judge.judge(q, answers["EN"], answers["PT-BR"],
                             judge_chat(client), use_traps=use_traps)
        flags = render_judgment(result)
    except Exception as e:
        flags = f"❌ The judge could not run: {esc(e)}"
    status = f"Done · {MODEL_LABEL} · temperature 0 · judge {judge.JUDGE_VERSION}"
    return out["EN"], out["PT-BR"], flags, status


# ---------------------------------------------------------------- tab 4: background runs

JOB = {"running": False, "kind": "", "run_id": "", "done": 0, "total": 0, "errors": [],
       "records": [], "message": "No run yet."}
_job_lock = threading.Lock()


def _save_job(meta, final=False):
    with _job_lock:
        records = sorted(JOB["records"], key=lambda r: r["id"])
    meta = dict(meta, n_questions=len(records), complete=final)
    storage.save_run(meta["run_id"], records, meta)


def _job_worker(tasks, meta):
    """tasks: list of (qid, callable returning a record)."""
    with ThreadPoolExecutor(BATCH_WORKERS) as pool:
        futures = {pool.submit(fn): qid for qid, fn in tasks}
        for n, fut in enumerate(as_completed(futures), 1):
            qid = futures[fut]
            try:
                rec = fut.result()
                with _job_lock:
                    JOB["records"].append(rec)
            except Exception as e:
                with _job_lock:
                    JOB["errors"].append(f"{qid}: {e}")
            with _job_lock:
                JOB["done"] += 1
            if n % 15 == 0:
                try:
                    _save_job(meta)  # checkpoint, so a restart does not lose everything
                except Exception as e:
                    with _job_lock:
                        JOB["errors"].append(f"checkpoint save failed: {e}")
    try:
        _save_job(meta, final=True)
        msg = f"Saved to {storage.where()} as run {meta['run_id']}."
    except Exception as e:
        msg = f"❌ Could not save the run: {e}"
    with _job_lock:
        JOB.update(running=False, message=msg)


def _start_job(kind, tasks, meta):
    with _job_lock:
        if JOB["running"]:
            return False
        JOB.update(running=True, kind=kind, run_id=meta["run_id"], done=0, total=len(tasks),
                   errors=[], records=[], message="Running…")
    threading.Thread(target=_job_worker, args=(tasks, meta), daemon=True).start()
    return True


def start_full_run(client=None):
    try:
        client = client or get_client()
    except Exception as e:
        return f"❌ {e}"
    run_id = audit.new_run_id("full")
    meta = audit.run_meta(run_id, "full", MODEL, MAX_TOKENS, n=len(QUESTIONS))
    tasks = [(qid, (lambda q=q: answer_and_judge(q, client))) for qid, q in QUESTIONS.items()]
    if not _start_job("full", tasks, meta):
        return job_status()
    return job_status()


def start_rejudge(source_run, client=None):
    """Judge the SAME saved answers again (no new answers): cheap, comparable."""
    if not source_run:
        return "Pick a saved run first."
    try:
        client = client or get_client()
    except Exception as e:
        return f"❌ {e}"
    records, src_meta = storage.load_run(source_run)
    if not records:
        return f"❌ Could not load run {source_run}."
    pairs = audit.legacy_to_answers(records, QUESTIONS)
    run_id = audit.new_run_id("rejudge")
    meta = audit.run_meta(run_id, "rejudge", src_meta.get("model", MODEL), MAX_TOKENS,
                          answers_from=source_run, n=len(pairs))
    tasks = []
    for p in pairs:
        q = dict(QUESTIONS.get(p["id"], {}), **{k: p[k] for k in ("id", "domain", "category", "risk")})
        tasks.append((p["id"], (lambda q=q, p=p: audit.judge_record(
            q, p["answer_en"], p["answer_pt"], judge_chat(client)))))
    _start_job("rejudge", tasks, meta)
    return job_status()


def job_status():
    with _job_lock:
        j = dict(JOB)
        errors = list(j["errors"])
    if not j["total"]:
        return j["message"]
    state = "⏳ Running" if j["running"] else "✅ Finished"
    return (f"### {state} ({j['kind']}): {j['done']} of {j['total']} questions\n"
            f"- Run ID: `{j['run_id']}`\n- {j['message']}\n- Errors: {len(errors)}"
            + ("\n" + "\n".join(f"  - {e}" for e in errors[:10]) if errors else ""))


def import_run(text, name):
    """Save an existing run file (JSON lines) into permanent storage, answers included."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    try:
        records = [json.loads(ln) for ln in lines]
    except Exception as e:
        return f"❌ Not valid JSON lines: {e}"
    if not records or "answer_en" not in records[0]:
        return "❌ This file has no saved answers."
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", name or audit.new_run_id("import"))
    meta = audit.run_meta(run_id, "imported", MODEL, MAX_TOKENS, n=len(records))
    meta["note"] = "Imported run; its judgments use the older judge. Re-judge it for v2 findings."
    storage.save_run(run_id, records, meta)
    return f"✅ Imported {len(records)} questions as run {run_id} into {storage.where()}."


def judged_runs():
    """Runs whose records carry v2 findings (have stable finding IDs)."""
    return [r for r in storage.list_runs() if not r.startswith("import")] or storage.list_runs()


# ---------------------------------------------------------------- tab 2: review

def load_run_state(run_id):
    records, meta = storage.load_run(run_id) if run_id else ([], {})
    meta = dict(meta, run_id=meta.get("run_id", run_id))
    return {"run_id": run_id, "records": records, "meta": meta}


def queue_items(state, mode, status_filter, min_sev):
    reviews = storage.load_reviews()
    by_ev = audit.review_index(state["records"], reviews)
    key = "with_traps" if mode.startswith("With") else "without_traps"
    items = []
    for r in state["records"]:
        for f in r.get(key, {}).get("findings", []):
            if "id" not in f:
                continue
            if not judge.is_issue(f) and status_filter != "All, including not counted":
                continue
            rev = audit.review_for(r, f, reviews, by_ev) or {}
            decision = rev.get("decision", "Not reviewed")
            if status_filter not in ("All", "All, including not counted") and decision != status_filter:
                continue
            if SEV_PRIORITY.get(f["severity"], 9) > SEV_PRIORITY.get(min_sev, 9):
                continue
            items.append((r, f, decision))
    items.sort(key=lambda x: (SEV_PRIORITY.get(x[1]["severity"], 9),
                              CAT_PRIORITY.get(x[1]["category"], 9), x[0]["id"]))
    return items


DECISION_ICON = {"Not reviewed": "⬜", "Confirmed issue": "✅",
                 "False alarm / acceptable variation": "❌", "Needs specialist review": "❓"}


def refresh_queue(state, mode, status_filter, min_sev):
    items = queue_items(state, mode, status_filter, min_sev)
    choices = [(f"{DECISION_ICON.get(d, '')} {r['id']} · {f['severity']} · "
                f"{judge.CATEGORIES.get(f['category'], f['category'])} — {f['explanation'][:70]}",
                f["id"]) for r, f, d in items]
    first = choices[0][1] if choices else None
    return gr.update(choices=choices, value=first), f"{len(choices)} finding(s) in this view"


def find(state, fid):
    for r in state["records"]:
        for key in ("with_traps", "without_traps"):
            for f in r.get(key, {}).get("findings", []):
                if f.get("id") == fid:
                    return r, f
    return None, None


def show_finding(state, fid):
    r, f = find(state, fid)
    empty = ("", "Not reviewed", None, None, "unclear", "", "", "")
    if not r:
        return ("<i>Pick a finding.</i>",) + empty
    meta = state["meta"]
    reviews = storage.load_reviews()
    rev = reviews.get(fid) or audit.review_for(r, f, reviews, audit.review_index(state["records"], reviews)) or {}
    evidence = (
        f"<h3>{esc(r['id'])} · {esc(r['domain'])}</h3>"
        f"{finding_card(f)}"
        "<div style='display:flex;gap:12px;flex-wrap:wrap'>"
        f"<div style='flex:1;min-width:280px'><b>Question (EN)</b><br>{esc(r.get('question_en'))}"
        f"<hr><b>Full answer (EN)</b><div style='max-height:520px;overflow:auto;border:1px solid #8884;"
        f"border-radius:6px;padding:8px;margin-top:4px'>{highlight(r['answer_en'], f['en_quote'])}</div></div>"
        f"<div style='flex:1;min-width:280px'><b>Pergunta (PT-BR)</b><br>{esc(r.get('question_pt'))}"
        f"<hr><b>Resposta completa (PT-BR)</b><div style='max-height:520px;overflow:auto;border:1px solid #8884;"
        f"border-radius:6px;padding:8px;margin-top:4px'>{highlight(r['answer_pt'], f['pt_quote'])}</div></div>"
        "</div>"
        f"<p><small>Run <code>{esc(meta.get('run_id'))}</code> · answers from "
        f"<code>{esc(meta.get('answers_from_run'))}</code> · model {esc(meta.get('model'))} · "
        f"temperature {esc(meta.get('temperature'))} · judge {esc(meta.get('judge_version'))} · "
        f"created {esc(meta.get('created_utc'))} · finding <code>{esc(fid)}</code> · "
        f"quotes: {esc(f.get('support'))}</small></p>"
    )
    return (evidence, rev.get("decision", "Not reviewed"),
            rev.get("category") or f["category"], rev.get("severity") or f["severity"],
            rev.get("problem_language") or f.get("problem_language", "unclear"),
            rev.get("notes", ""), rev.get("source", ""), rev.get("correct_version", ""),
            f"Previously saved: {rev.get('decision')} at {rev.get('reviewed_at')}" if rev else "")


def save_decision(state, fid, decision, category, severity, lang, notes, source, correct):
    if not fid:
        return "Pick a finding first."
    review = {"decision": decision, "category": category, "severity": severity,
              "problem_language": lang, "notes": notes, "source": source,
              "correct_version": correct, "reviewed_at": audit.now_utc(),
              "run_id": state.get("run_id")}
    try:
        storage.save_review(fid, review)
    except Exception as e:
        return f"❌ Not saved: {e}"
    return f"✅ Saved: {decision} · {storage.where()}"


def retest(state, fid, client=None):
    """Ask the same question again now (new answers) and judge: does the issue come back?"""
    r, f = find(state, fid)
    if not r:
        return "Pick a finding first."
    q = QUESTIONS.get(r["id"])
    try:
        rec = answer_and_judge(q, client=client)
    except Exception as e:
        return f"❌ Retest failed: {esc(e)}"
    run_id = audit.new_run_id(f"retest-{r['id']}")
    meta = audit.run_meta(run_id, "retest", MODEL, MAX_TOKENS, n=1)
    meta["retest_of_finding"] = fid
    try:
        storage.save_run(run_id, [rec], meta)
    except Exception:
        pass
    same_cat = [g for g in rec["with_traps"]["findings"]
                if judge.is_issue(g) and g["category"] == f["category"]]
    verdict = (f"Same kind of issue ({judge.CATEGORIES.get(f['category'])}) flagged again: "
               f"**{len(same_cat)}** time(s). Check below whether it is the same substance."
               if same_cat else "This kind of issue was NOT flagged in the new answers.")
    return (f"<p><b>Retest saved as {esc(run_id)}</b> (about 4 model calls). {verdict}</p>"
            + render_judgment(rec["with_traps"]))


# ---------------------------------------------------------------- tab 3: results

def results_view(state):
    if not state["records"]:
        return "Pick a run."
    m = audit.metrics(state["records"], storage.load_reviews())
    return metrics_header(state) + audit.metrics_markdown(m)


def metrics_header(state):
    meta = state["meta"]
    return (f"**Run** `{meta.get('run_id')}` · {meta.get('kind', '')} · answers from "
            f"`{meta.get('answers_from_run')}` · {meta.get('model')} · judge "
            f"{meta.get('judge_version')} · {meta.get('created_utc')}\n\n")


def confirmed_examples(state, limit=3):
    reviews = storage.load_reviews()
    by_ev = audit.review_index(state["records"], reviews)
    items = []
    for r in state["records"]:
        for f in r.get("with_traps", {}).get("findings", []):
            rev = audit.review_for(r, f, reviews, by_ev) or {}
            if rev.get("decision") == "Confirmed issue":
                items.append((r, f, rev))
    items.sort(key=lambda x: SEV_PRIORITY.get(x[2].get("severity") or x[1]["severity"], 9))
    if not items:
        return ("<p><b>No human-confirmed examples yet.</b> Open <i>Review findings</i> and "
                "decide the highest-severity flags first.</p>")
    cards = []
    for r, f, rev in items[:limit]:
        cards.append(f"<h4>✅ {esc(r['id'])} · confirmed by the interpreter</h4>" + finding_card(f)
                     + (f"<p><b>Interpreter's note:</b> {esc(rev.get('notes'))}</p>" if rev.get("notes") else "")
                     + (f"<p><b>Correct version:</b> {esc(rev.get('correct_version'))}</p>"
                        if rev.get("correct_version") else ""))
    return "".join(cards)


def export_all(state):
    if not state["records"]:
        return None, "Pick a run."
    reviews = storage.load_reviews()
    meta = state["meta"]
    out = HERE / "exports"
    out.mkdir(exist_ok=True)
    rid = meta.get("run_id", "run")
    files = []
    sheet = audit.to_csv(audit.review_rows(state["records"], reviews, meta))
    p = out / f"review_sheet_{rid}.csv"
    p.write_text(sheet, encoding="utf-8")
    files.append(p)
    corr = audit.corrections(state["records"], reviews, meta)
    p = out / f"expert_corrections_{rid}.jsonl"
    p.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in corr) + ("\n" if corr else ""),
                 encoding="utf-8")
    files.append(p)
    p = out / f"results_with_reviews_{rid}.json"
    p.write_text(json.dumps({"meta": meta, "metrics": audit.metrics(state["records"], reviews),
                             "reviews": reviews}, ensure_ascii=False, indent=1), encoding="utf-8")
    files.append(p)
    msg = ""
    try:
        import report
        pdf = out / f"LinguaCheck_Report_{rid}.pdf"
        report.build_v2(state["records"], meta, reviews, str(pdf))
        files.append(pdf)
    except Exception as e:
        msg = f" (PDF not made: {e})"
    try:
        for fp in files:
            storage.write_bytes(f"exports/{fp.name}", fp.read_bytes(), f"Export {fp.name}")
        saved = f"Also saved in {storage.where()} under exports/."
    except Exception as e:
        saved = f"Could not copy to permanent storage: {e}"
    return [str(f) for f in files], (f"✅ {len(corr)} confirmed correction pair(s) exported. "
                                    f"{saved}{msg}")


# ---------------------------------------------------------------- confirmed errors (public page)

REVIEW_RUN = os.getenv("REVIEW_RUN", "rejudge_20261030-015756")
FEATURED = ["L07-31a066e75a", "M06-fb23f38106", "F03-933b4e776f", "M21-183f4f9665",
            "L14-9d4ffc5809"]
EXTRA_CATEGORIES = {"legal_overgeneralization": "Legal overgeneralization"}
LANG_NAME = {"PT": "Portuguese (PT-BR)", "EN": "English", "both": "Both", "unclear": "Unclear"}
_cases_cache = {"at": 0, "data": None}


def _cat_label(c):
    return judge.CATEGORIES.get(c) or EXTRA_CATEGORIES.get(c) or str(c).replace("_", " ").capitalize()


def load_expert_added():
    try:
        text = storage.read_text("reviews/expert_added.json")
        return json.loads(text) if text else {}
    except Exception:
        return {}


def load_cases(force=False):
    """Expert-confirmed findings for the reviewed run (cached 5 minutes)."""
    if not force and _cases_cache["data"] and time.time() - _cases_cache["at"] < 300:
        return _cases_cache["data"]
    records, meta = storage.load_run(REVIEW_RUN)
    reviews = storage.load_reviews(refresh=True)
    by_id = {r["id"]: r for r in records}
    confirmed, false_alarms = [], 0
    for r in records:
        for f in r.get("with_traps", {}).get("findings", []):
            rev = reviews.get(f.get("id"))
            if not rev:
                continue
            if rev.get("decision") == "Confirmed issue":
                confirmed.append((r, f, rev))
            elif str(rev.get("decision", "")).startswith("False alarm"):
                false_alarms += 1
    expert = [(k, by_id.get(e.get("question_id"), {}), e)
              for k, e in load_expert_added().items()]
    data = {"records": records, "confirmed": confirmed, "false_alarms": false_alarms,
            "expert": expert, "tr": load_translations()}
    _cases_cache.update(at=time.time(), data=data)
    return data


def _en(text):
    """English translation line shown under Portuguese text (for non-Portuguese readers)."""
    return (f"<div style='margin-top:6px;font-size:0.9em;opacity:0.85'>"
            f"🇺🇸 <i>English translation:</i> {esc(text)}</div>" if text else "")


def load_translations():
    try:
        text = storage.read_text("reviews/translations.json")
        return json.loads(text) if text else {}
    except Exception:
        return {}


def _quote_box(label, quote, wrong, translation=""):
    missing = not quote or quote.strip().lower() in ("(missing)", "missing", "")
    if missing:
        text = ("<i>Not mentioned in this answer.</i>" if wrong
                else "<i>✓ No problem quoted on this point.</i>")
    else:
        text = esc(re.sub(r"\*\*|__|`", "", quote)) + _en(translation)
    border = "2px solid #d03b3b" if wrong else "1px solid #8884"
    tag = " <span style='color:#d03b3b'>✗ problem here</span>" if wrong else ""
    return (f"<div style='flex:1;min-width:240px;border:{border};border-radius:8px;"
            f"padding:10px'><b>{label}</b>{tag}<br>{text}</div>")


def case_card(r, f, rev, featured=False, tr=None):
    tr = tr or {}
    cat = rev.get("category") or f.get("category")
    sev = rev.get("severity") or f.get("severity")
    lang = rev.get("problem_language") or f.get("problem_language", "unclear")
    star = "⭐ " if featured else ""
    correct = rev.get("correct_version") or ""
    return (
        "<div style='border:1px solid #8886;border-radius:10px;padding:14px;margin:12px 0'>"
        f"<div style='font-size:0.85em;opacity:0.8'>{star}{esc(r.get('id', ''))} · "
        f"{esc(r.get('domain', ''))} · {SEV_ICON.get(sev, sev)} · {esc(_cat_label(cat))} · "
        f"wrong language: <b>{esc(LANG_NAME.get(lang, lang))}</b></div>"
        f"<div style='margin:6px 0 10px'><b>Question:</b> {esc(r.get('question_en', ''))}</div>"
        "<div style='display:flex;gap:10px;flex-wrap:wrap'>"
        + _quote_box("Muse in English", f.get("en_quote"), lang in ("EN", "both"))
        + _quote_box("Muse in Portuguese", f.get("pt_quote"), lang in ("PT", "both"),
                     tr.get("pt_quote_en"))
        + "</div>"
        f"<div style='margin-top:10px'><b>Expert verdict:</b> {esc(rev.get('notes', ''))}</div>"
        + (f"<div style='margin-top:6px;background:#22c55e1a;padding:8px;border-radius:6px'>"
           f"<b>Correct version:</b> {esc(correct)}{_en(tr.get('correct_version_en'))}</div>"
           if correct else "")
        + f"<div style='margin-top:6px;font-size:0.85em;opacity:0.8'>Source: "
        f"{esc(rev.get('source', ''))}</div></div>")


def expert_card(r, e, tr=None):
    tr = tr or {}
    lang = e.get("problem_language", "unclear")
    return (
        "<div style='border:2px dashed #eb6834;border-radius:10px;padding:14px;margin:12px 0'>"
        f"<div style='font-size:0.85em;opacity:0.8'>🧠 {esc(e.get('question_id', ''))} · "
        f"{esc(r.get('domain', ''))} · {SEV_ICON.get(e.get('severity'), e.get('severity'))} · "
        f"{esc(_cat_label(e.get('category')))} · missed by the AI judge, caught by the expert</div>"
        f"<div style='margin:6px 0 10px'><b>Question:</b> {esc(r.get('question_en', ''))}</div>"
        "<div style='display:flex;gap:10px;flex-wrap:wrap'>"
        + _quote_box("Muse in English", e.get("evidence_en"), lang in ("EN", "both"))
        + _quote_box("Muse in Portuguese", e.get("evidence_pt"), lang in ("PT", "both"),
                     tr.get("pt_quote_en"))
        + "</div>"
        f"<div style='margin-top:10px'><b>Expert finding:</b> {esc(e.get('notes', ''))}"
        f"{_en(tr.get('notes_en'))}</div>"
        + (f"<div style='margin-top:6px;background:#22c55e1a;padding:8px;border-radius:6px'>"
           f"<b>Correct version:</b> {esc(e.get('correct_version'))}"
           f"{_en(tr.get('correct_version_en'))}</div>"
           if e.get("correct_version") else "")
        + f"<div style='margin-top:6px;font-size:0.85em;opacity:0.8'>Source: "
        f"{esc(e.get('source', ''))}</div></div>")


def errors_summary(data):
    conf = data["confirmed"]
    n_q = len(data["records"]) or 75
    decided = len(conf) + data["false_alarms"]
    from collections import Counter
    sev = Counter((rev.get("severity") or f["severity"]) for _, f, rev in conf)
    lang = Counter((rev.get("problem_language") or "unclear") for _, f, rev in conf)
    dom = Counter(r.get("domain") for r, _, _ in conf)
    qs = len({r["id"] for r, _, _ in conf})
    prec = f"{len(conf) / decided:.0%}" if decided else "n/a"
    return (
        f"### {len(conf)} confirmed cross-language problems in {qs} of {n_q} questions\n"
        f"Every case below was **flagged by LinguaCheck** and **confirmed by a professional "
        f"medical & legal interpreter (EN/PT-BR)** against official sources. The model's answers "
        f"are shown exactly as generated.\n\n"
        f"| Severity | Wrong language | Domain | Judge accuracy |\n|---|---|---|---|\n"
        f"| 🔴 {sev['high']} high · 🟠 {sev['medium']} medium · 🟡 {sev['low']} low "
        f"| Portuguese {lang['PT']} · English {lang['EN']} "
        f"| Finance {dom['Finance']} · Legal {dom['Legal']} · Medical {dom['Medical']} "
        f"| {prec} of {decided} expert-decided flags were real |\n\n"
        f"Plus **{len(data['expert'])} errors the AI judge missed and the expert caught.**\n\n"
        f"_Limits: {n_q} fictional test questions; flags not yet reviewed are not counted; "
        f"answers without flags were not checked for missed errors (recall not measured)._")


def errors_page(domain="All", min_sev="All", force=False):
    try:
        data = load_cases(force)
    except Exception as e:
        return f"⚠️ Could not load the reviewed results: {esc(e)}", ""
    order = {"high": 0, "medium": 1, "low": 2}
    keep = {"All": 3, "high": 0, "medium": 1, "low": 2}[min_sev]
    cases = [c for c in data["confirmed"]
             if (domain == "All" or c[0].get("domain") == domain)
             and order.get(c[2].get("severity") or c[1]["severity"], 2) <= keep]
    feat = {fid: i for i, fid in enumerate(FEATURED)}
    cases.sort(key=lambda c: (feat.get(c[1].get("id"), 99),
                              order.get(c[2].get("severity") or c[1]["severity"], 2),
                              c[0].get("id", "")))
    tr = data.get("tr", {})
    html_out = "".join(case_card(r, f, rev, f.get("id") in feat, tr.get(f.get("id")))
                       for r, f, rev in cases)
    exp = [x for x in data["expert"] if domain == "All" or x[1].get("domain") == domain]
    if exp:
        html_out += ("<h3>🧠 Caught by the expert, missed by the AI judge</h3>"
                     + "".join(expert_card(r, e, tr.get(k)) for k, r, e in exp))
    if not html_out:
        html_out = "<i>No confirmed cases for this filter.</i>"
    return errors_summary(data), html_out


# ---------------------------------------------------------------- page

DECISION_CHOICES = audit.DECISIONS
CATEGORY_CHOICES = [(v, k) for k, v in judge.CATEGORIES.items()]

with gr.Blocks(title="LinguaCheck") as demo:
    gr.Markdown(
        "# LinguaCheck\n"
        "**Does the AI say the same thing in every language?** LinguaCheck asks Meta's model "
        "the same medical, legal or finance question in English and Brazilian Portuguese, flags "
        "meaningful differences with the exact evidence, and lets a professional interpreter "
        "confirm or reject each flag.\n\n"
        f"Model: **{MODEL_LABEL}** · temperature 0 · {DISCLAIMER} All test cases are fictional. "
        f"Storage: {storage.where()}."
    )
    state = gr.State({"run_id": None, "records": [], "meta": {}})

    with gr.Tab("★ Confirmed errors"):
        err_summary = gr.Markdown()
        with gr.Row():
            err_domain = gr.Radio(["All", "Medical", "Legal", "Finance"], value="All",
                                  label="Domain")
            err_sev = gr.Radio(["All", "high", "medium", "low"], value="All",
                               label="Minimum severity")
            err_refresh = gr.Button("Refresh")
        err_cards = gr.HTML()

    with gr.Tab("1 · Ask one question"):
        pick = gr.Dropdown(choices=CHOICES, label=f"Question ({len(CHOICES)} verified)",
                           value=CHOICES[0][1] if CHOICES else None)
        with gr.Row():
            q_en = gr.Textbox(label="Question (EN)", lines=4, interactive=False)
            q_pt = gr.Textbox(label="Pergunta (PT-BR)", lines=4, interactive=False)
        key_box = gr.Markdown()
        with gr.Row():
            use_traps = gr.Checkbox(value=True, label=(
                f"Use the interpreter trap file ({len(judge.TRAPS)} verified traps)"))
            go = gr.Button("Ask Meta's model in both languages and compare", variant="primary")
        flags = gr.HTML()
        with gr.Row():
            a_en = gr.Markdown(label="Answer (EN)")
            a_pt = gr.Markdown(label="Resposta (PT-BR)")
        status = gr.Markdown()

    with gr.Tab("2 · Review findings"):
        with gr.Row():
            run_pick = gr.Dropdown(label="Run", choices=[], allow_custom_value=True)
            reload_btn = gr.Button("Reload runs")
        with gr.Row():
            mode = gr.Radio(["With trap file", "Without trap file"], value="With trap file",
                            label="Judgment")
            status_filter = gr.Dropdown(["Not reviewed", "All"] + DECISION_CHOICES[1:]
                                        + ["All, including not counted"],
                                        value="Not reviewed", label="Show")
            min_sev = gr.Radio(["high", "medium", "low"], value="low",
                               label="Minimum severity")
        queue = gr.Dropdown(label="Findings (most serious first)", choices=[])
        queue_info = gr.Markdown()
        evidence = gr.HTML()
        with gr.Group():
            decision = gr.Radio(DECISION_CHOICES, value="Not reviewed", label="Your decision")
            with gr.Row():
                rev_cat = gr.Dropdown(CATEGORY_CHOICES, label="Category (edit if wrong)")
                rev_sev = gr.Dropdown(["high", "medium", "low"], label="Severity (edit if wrong)")
                rev_lang = gr.Radio(["EN", "PT", "both", "unclear"], value="unclear",
                                    label="Which language is wrong?")
            notes = gr.Textbox(label="Your reasoning", lines=2)
            source = gr.Textbox(label="Supporting source (optional: link or reference)")
            correct = gr.Textbox(label="Correct version (for the expert correction file)", lines=2)
            with gr.Row():
                save_btn = gr.Button("Save decision", variant="primary")
                retest_btn = gr.Button("Retest this question now (≈4 model calls)")
            saved_msg = gr.Markdown()
        retest_out = gr.HTML()

    with gr.Tab("3 · Results"):
        res_refresh = gr.Button("Refresh numbers")
        res_md = gr.Markdown()
        gr.Markdown("### Human-confirmed examples")
        res_examples = gr.HTML()
        export_btn = gr.Button("Export review sheet, expert corrections and PDF report")
        export_files = gr.File(label="Downloads", file_count="multiple")
        export_msg = gr.Markdown()

    with gr.Tab("4 · Runs"):
        gr.Markdown("**Re-judge** reuses saved answers (judge calls only, cheaper, and the trap "
                    "file comparison uses the SAME answers). **Full run** asks for new answers.")
        with gr.Row():
            src_pick = gr.Dropdown(label="Saved run to re-judge", choices=[],
                                   allow_custom_value=True)
            rejudge_btn = gr.Button("Re-judge saved answers (≈150 judge calls)")
        full_btn = gr.Button("Full run: new answers for all questions (≈300 model calls)")
        job_refresh = gr.Button("Refresh progress")
        job_md = gr.Markdown()
        with gr.Accordion("Import a run file", open=False):
            imp_text = gr.Textbox(label="Run file (JSON lines)", lines=3)
            imp_name = gr.Textbox(label="Name", value="import_full_run_20261029-1824")
            imp_btn = gr.Button("Import")
            imp_msg = gr.Markdown()

    # ---- wiring
    demo.load(errors_page, inputs=[err_domain, err_sev], outputs=[err_summary, err_cards])
    for ctl in (err_domain, err_sev):
        ctl.change(errors_page, inputs=[err_domain, err_sev], outputs=[err_summary, err_cards])
    err_refresh.click(lambda d, s: errors_page(d, s, force=True), inputs=[err_domain, err_sev],
                      outputs=[err_summary, err_cards])
    pick.change(show_question, inputs=pick, outputs=[q_en, q_pt, key_box])
    demo.load(show_question, inputs=pick, outputs=[q_en, q_pt, key_box])
    go.click(run_audit, inputs=[pick, use_traps], outputs=[a_en, a_pt, flags, status])

    def _runs():
        runs = storage.list_runs()
        judged = [r for r in runs if not r.startswith("import")]
        return (gr.update(choices=judged, value=judged[0] if judged else None),
                gr.update(choices=runs, value=runs[0] if runs else None))

    demo.load(_runs, outputs=[run_pick, src_pick])
    reload_btn.click(_runs, outputs=[run_pick, src_pick])

    def _pick_run(run_id, mode_v, sf, ms):
        st = load_run_state(run_id)
        q, info = refresh_queue(st, mode_v, sf, ms)
        return st, q, info

    run_pick.change(_pick_run, inputs=[run_pick, mode, status_filter, min_sev],
                    outputs=[state, queue, queue_info])
    for ctl in (mode, status_filter, min_sev):
        ctl.change(refresh_queue, inputs=[state, mode, status_filter, min_sev],
                   outputs=[queue, queue_info])
    queue.change(show_finding, inputs=[state, queue],
                 outputs=[evidence, decision, rev_cat, rev_sev, rev_lang, notes, source, correct,
                          saved_msg])
    save_btn.click(save_decision,
                   inputs=[state, queue, decision, rev_cat, rev_sev, rev_lang, notes, source, correct],
                   outputs=saved_msg)
    retest_btn.click(retest, inputs=[state, queue], outputs=retest_out)

    res_refresh.click(lambda st: (results_view(st), confirmed_examples(st)), inputs=state,
                      outputs=[res_md, res_examples])
    export_btn.click(export_all, inputs=state, outputs=[export_files, export_msg])

    rejudge_btn.click(start_rejudge, inputs=src_pick, outputs=job_md)
    full_btn.click(lambda: start_full_run(), outputs=job_md)
    job_refresh.click(job_status, outputs=job_md)
    imp_btn.click(import_run, inputs=[imp_text, imp_name], outputs=imp_msg, api_name="import_run")


if __name__ == "__main__":
    demo.launch()
