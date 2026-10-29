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
import time
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


if __name__ == "__main__":
    demo.launch()
