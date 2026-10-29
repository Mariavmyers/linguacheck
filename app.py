"""
LinguaCheck v1 — ask Meta's model the same question in English and
Brazilian Portuguese and show both answers side by side.

Model: Muse Spark Open (Meta open weights) served by Baseten.
The API key is read from the BASETEN_API_KEY environment variable
(local: a .env file; Hugging Face Spaces: a Secret). Never hard-code it.
"""

import csv
import html
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

import judge

load_dotenv()

BASE_URL = os.getenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1")
MODEL = os.getenv("BASETEN_MODEL", "meta-models/muse-spark-open-1026")
MODEL_LABEL = "Muse Spark Open (Meta, open weights) via Baseten"
DISCLAIMER = "⚠️ Model output under audit — not advice."

HERE = Path(__file__).parent
QUESTIONS_FILE = HERE / "data" / "questions.csv"
RESULTS_FILE = HERE / "results" / "answers.jsonl"


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
    # Some servers put the reasoning inside <think>...</think> in the content.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    # If the reasoning block was never closed, drop everything before the tag end.
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
        model=MODEL, messages=messages, temperature=0, max_tokens=4096,
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


def log(record, file=None):
    file = file or RESULTS_FILE
    file.parent.mkdir(exist_ok=True)
    record = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model": MODEL, "temperature": 0, **record}
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_result(qid, lang, question, answer, seconds):
    log({"kind": "answer", "id": qid, "lang": lang, "question": question,
         "answer": answer, "seconds": seconds})


# ---------------------------------------------------------------- UI logic

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


SEVERITY_ICON = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟡 Low"}
TYPE_LABEL = {"fact_drift": "Fact drift", "missing": "Missing in one language",
              "language_trap": "Interpreter trap"}


def esc(s):
    return html.escape(str(s or "")).replace("\n", "<br>")


def render_findings(result):
    """Flags as cards: the EN sentence and the PT sentence side by side."""
    if result.get("error"):
        return f"⚠️ Judge problem: {esc(result['error'])}"
    cc = result["code_checks"]
    traps_note = "with the interpreter trap file" if result["use_traps"] else "WITHOUT the trap file"
    findings = result["findings"]
    head = (f"<h3>{'✅ No drift found' if not findings else f'🚩 {len(findings)} flag(s)'}"
            f" <small>(judge: {esc(MODEL_LABEL)}, {traps_note})</small></h3>")
    cards = []
    order = {"high": 0, "medium": 1, "low": 2}
    for f in sorted(findings, key=lambda f: order.get(f["severity"], 3)):
        trap = f" · trap {esc(f['trap_id'])}" if f.get("trap_id") else ""
        cards.append(
            "<div style='border:1px solid #8884;border-radius:8px;padding:10px;margin:8px 0'>"
            f"<b>{SEVERITY_ICON.get(f['severity'], f['severity'])} · "
            f"{TYPE_LABEL.get(f['type'], esc(f['type']))}{trap}</b><br>{esc(f['explanation'])}"
            "<div style='display:flex;gap:10px;margin-top:8px;flex-wrap:wrap'>"
            f"<div style='flex:1;min-width:220px;background:#3b82f61a;padding:8px;border-radius:6px'>"
            f"<b>EN says</b><br>{esc(f['en_quote'])}</div>"
            f"<div style='flex:1;min-width:220px;background:#22c55e1a;padding:8px;border-radius:6px'>"
            f"<b>PT-BR says</b><br>{esc(f['pt_quote'])}</div></div></div>"
        )
    code_lines = judge.code_checks_text(cc)
    code = f"<details><summary>Code check (numbers and dates)</summary>{esc(code_lines)}</details>"
    return head + "".join(cards) + code


def run_audit(qid, use_traps=True, client=None):
    q = QUESTIONS.get(qid)
    if not q:
        msg = "Pick a question first."
        return msg, msg, "", ""
    out, answers = {}, {}
    for lang, field in (("EN", "question_en"), ("PT-BR", "question_pt")):
        try:
            answer, secs = ask_model(q[field], client=client)
            save_result(qid, lang, q[field], answer, secs)
            answers[lang] = answer
            out[lang] = f"{DISCLAIMER}\n\n{answer}\n\n*({secs} s)*"
        except Exception as e:  # show the problem in plain words, don't crash
            out[lang] = f"❌ Could not get an answer: {e}"
    if len(answers) < 2:
        return out["EN"], out["PT-BR"], "", "Could not compare: one answer is missing."
    try:
        result = judge.judge(q, answers["EN"], answers["PT-BR"],
                             judge_chat(client), use_traps=use_traps)
        log({"kind": "judgment", "id": qid, **result})
        flags = render_findings(result)
    except Exception as e:
        flags = f"❌ The judge could not run: {esc(e)}"
    status = f"Done · {MODEL_LABEL} · temperature 0 · saved to results/answers.jsonl"
    return out["EN"], out["PT-BR"], flags, status


# ---------------------------------------------------------------- full test-set run

def audit_question(q, client=None):
    """EN + PT answers once, then the judge WITH and WITHOUT the trap file."""
    answers = {}
    for lang, field in (("EN", "question_en"), ("PT-BR", "question_pt")):
        answers[lang], _ = ask_model(q[field], client=client)
    out = {"id": q["id"], "domain": q["domain"], "category": q["category"],
           "risk": q["risk"], "answer_en": answers["EN"], "answer_pt": answers["PT-BR"]}
    for use in (True, False):
        out["with_traps" if use else "without_traps"] = judge.judge(
            q, answers["EN"], answers["PT-BR"], judge_chat(client), use_traps=use)
    return out


BATCH = {"running": False, "done": 0, "total": 0, "results": [], "errors": [],
         "started": None, "finished": None, "files": []}
_batch_lock = threading.Lock()
BATCH_WORKERS = 4


def _short(s, n=220):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def compact_result(r):
    def fs(findings):
        return [{"type": f["type"], "sev": f["severity"], "trap": f.get("trap_id"),
                 "en": _short(f["en_quote"]), "pt": _short(f["pt_quote"]),
                 "why": _short(f["explanation"], 300)} for f in findings]
    return {"id": r["id"], "domain": r["domain"], "category": r["category"], "risk": r["risk"],
            "with": fs(r["with_traps"]["findings"]),
            "without": fs(r["without_traps"]["findings"]),
            "judge_error": r["with_traps"].get("error") or r["without_traps"].get("error")}


def _write_batch_files(stamp):
    base = RESULTS_FILE.parent
    base.mkdir(exist_ok=True)
    results = sorted(BATCH["results"], key=lambda r: r["id"])
    full = base / f"full_run_{stamp}.jsonl"
    with open(full, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    review = base / f"review_{stamp}.csv"
    with open(review, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "domain", "trap_file", "type", "severity", "trap_id",
                    "en_quote", "pt_quote", "explanation", "maria_says (real / false alarm)"])
        for r in results:
            for mode, key in (("with", "with_traps"), ("without", "without_traps")):
                for x in r[key]["findings"]:
                    w.writerow([r["id"], r["domain"], mode, x["type"], x["severity"],
                                x.get("trap_id") or "", x["en_quote"], x["pt_quote"],
                                x["explanation"], ""])
    return [str(full), str(review)]


def _batch_worker(ids, client):
    with ThreadPoolExecutor(BATCH_WORKERS) as pool:
        futures = {pool.submit(audit_question, QUESTIONS[i], client): i for i in ids}
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                r = fut.result()
                log({"kind": "batch_result", **r})
                with _batch_lock:
                    BATCH["results"].append(r)
            except Exception as e:
                with _batch_lock:
                    BATCH["errors"].append(f"{qid}: {e}")
            with _batch_lock:
                BATCH["done"] += 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    with _batch_lock:
        BATCH["files"] = _write_batch_files(stamp)
        BATCH["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        BATCH["running"] = False


def start_batch(client=None):
    with _batch_lock:
        if BATCH["running"]:
            return batch_status()
    try:
        client = client or get_client()
    except Exception as e:
        return f"❌ {e}", "", None
    ids = list(QUESTIONS)
    with _batch_lock:
        BATCH.update(running=True, done=0, total=len(ids), results=[], errors=[],
                     started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     finished=None, files=[])
    threading.Thread(target=_batch_worker, args=(ids, client), daemon=True).start()
    return batch_status()


def batch_status():
    with _batch_lock:
        b = dict(BATCH)
        results = sorted(b["results"], key=lambda r: r["id"])
        errors = list(b["errors"])
    if not b["total"]:
        return "No full run yet.", "", None
    drift_with = sum(1 for r in results if r["with_traps"]["findings"])
    drift_without = sum(1 for r in results if r["without_traps"]["findings"])
    high = sum(1 for r in results
               if any(f["severity"] == "high" for f in r["with_traps"]["findings"]))
    state = "⏳ Running" if b["running"] else "✅ Finished"
    md = (f"### {state}: {b['done']} of {b['total']} questions\n"
          f"- Questions with at least one flag **with** the trap file: **{drift_with}**\n"
          f"- Questions with at least one flag **without** the trap file: **{drift_without}**\n"
          f"- Questions with a **high-severity** flag: **{high}**\n"
          f"- Errors: {len(errors)}" + ("\n\n" + "\n".join(f"  - {e}" for e in errors[:10]) if errors else ""))
    summary = json.dumps({"state": "running" if b["running"] else "finished",
                          "done": b["done"], "total": b["total"], "errors": errors,
                          "results": [compact_result(r) for r in results]},
                         ensure_ascii=False)
    files = b["files"] or None
    return md, summary, files


# ---------------------------------------------------------------- page

with gr.Blocks(title="LinguaCheck") as demo:
    gr.Markdown(
        "# LinguaCheck\n"
        "**Does the AI say the same thing in every language?** "
        "Pick a medical, legal or finance question. LinguaCheck asks Meta's model "
        "the same question in English and Brazilian Portuguese and shows both answers "
        "side by side.\n\n"
        f"Model: **{MODEL_LABEL}** · temperature 0 · {DISCLAIMER} "
        "All test cases are fictional."
    )
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

    pick.change(show_question, inputs=pick, outputs=[q_en, q_pt, key_box])
    demo.load(show_question, inputs=pick, outputs=[q_en, q_pt, key_box])
    go.click(run_audit, inputs=[pick, use_traps], outputs=[a_en, a_pt, flags, status])

    with gr.Accordion("Full audit: run all questions (with and without the trap file)",
                      open=False):
        gr.Markdown("Runs every verified question in English and Portuguese, then judges "
                    "each pair twice. Takes a while; press Refresh to see progress.")
        with gr.Row():
            run_all_btn = gr.Button("Run all questions")
            refresh_btn = gr.Button("Refresh progress")
        batch_md = gr.Markdown()
        batch_json = gr.Textbox(label="Results (JSON)", lines=4, max_lines=8,
                                elem_id="batch-json")
        batch_files = gr.File(label="Download full results and review sheet",
                              file_count="multiple")
    run_all_btn.click(start_batch, outputs=[batch_md, batch_json, batch_files])
    refresh_btn.click(batch_status, outputs=[batch_md, batch_json, batch_files])


if __name__ == "__main__":
    demo.launch()
