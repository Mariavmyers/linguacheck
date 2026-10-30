"""
LinguaCheck judge: compares the English and Brazilian Portuguese answers.

Two layers (the method from Meta's Day 3 verification workshop):
  1. Code checks (same result every time, no AI): numbers and dates.
     Handles the Brazilian decimal comma (2,5 = 2.5) and thousands dot
     (1.200 = 1200), so a correct "2,5 mL" is NOT flagged.
  2. AI judge (a Meta model): finds differences in meaning, quotes the exact
     EN and PT sentences as evidence, and labels each finding.
     It can read the interpreter trap file (on/off, for the before/after test).

Finding types:
  fact_drift     - EN and PT give different facts (number, warning, right, step)
  missing        - a fact appears in one language and is missing in the other
  language_trap  - facts match, but the PT wording falls into a known trap
                   (false friend, calque, register mismatch, brand as generic)
"""

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
TRAPS_FILE = HERE / "data" / "traps.csv"


# ------------------------------------------------------------ trap file

def load_traps():
    with open(TRAPS_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


TRAPS = load_traps()


def traps_for(domain):
    """Traps for this question's domain (keeps the judge focused)."""
    return [t for t in TRAPS if t["domain"] == domain]


def traps_as_text(traps):
    return "\n".join(
        f"- [{t['id']}] EN '{t['english_in_context']}': trap in PT = {t['trap_pt']}; "
        f"correct PT-BR = {t['correct_pt']} (type {t['type']})"
        for t in traps
    )


# ------------------------------------------------------------ layer 1: code checks

NUM_RE = re.compile(r"(?<![\w/])(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?![\w/])")

EN_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
PT_MONTHS = {m: i for i, m in enumerate(
    ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
     "setembro", "outubro", "novembro", "dezembro"], 1)}


def number_readings(token):
    """All sensible values a number token can mean (US and Brazilian style)."""
    vals = set()
    for dec, thou in ((".", ","), (",", ".")):
        t = token
        parts = t.split(thou)
        # a thousands separator must be followed by groups of exactly 3 digits
        if len(parts) > 1 and all(len(p.split(dec)[0]) == 3 for p in parts[1:]):
            t = t.replace(thou, "")
        elif thou in t:
            continue
        try:
            vals.add(round(float(t.replace(dec, ".")), 4))
        except ValueError:
            pass
    return vals


def extract_numbers(text):
    """List of (token, set_of_values). Ignores small list markers like (1)."""
    text = re.sub(r"\(\d\)", " ", text)
    out = []
    for m in NUM_RE.finditer(text):
        vals = number_readings(m.group(1))
        if vals:
            out.append((m.group(1), vals))
    return out


def extract_dates(text):
    """(month, day) pairs written with month names, in EN or PT."""
    t = text.lower()
    found = set()
    for name, mo in EN_MONTHS.items():
        for m in re.finditer(rf"\b{name}\s+(\d{{1,2}})\b", t):
            found.add((mo, int(m.group(1))))
    for name, mo in PT_MONTHS.items():
        for m in re.finditer(rf"\b(\d{{1,2}})(?:º)?\s+de\s+{name}\b", t):
            found.add((mo, int(m.group(1))))
    return found


def code_checks(answer_en, answer_pt):
    """Numbers and dates present in one answer but not the other."""
    en_nums, pt_nums = extract_numbers(answer_en), extract_numbers(answer_pt)
    en_vals = set().union(*[v for _, v in en_nums]) if en_nums else set()
    pt_vals = set().union(*[v for _, v in pt_nums]) if pt_nums else set()
    only_en = sorted({tok for tok, v in en_nums if not v & pt_vals})
    only_pt = sorted({tok for tok, v in pt_nums if not v & en_vals})
    en_dates, pt_dates = extract_dates(answer_en), extract_dates(answer_pt)
    return {
        "numbers_only_in_en": only_en,
        "numbers_only_in_pt": only_pt,
        "dates_only_in_en": sorted(en_dates - pt_dates),
        "dates_only_in_pt": sorted(pt_dates - en_dates),
    }


def code_checks_text(cc):
    lines = []
    if cc["numbers_only_in_en"]:
        lines.append("Numbers only in EN: " + ", ".join(cc["numbers_only_in_en"]))
    if cc["numbers_only_in_pt"]:
        lines.append("Numbers only in PT: " + ", ".join(cc["numbers_only_in_pt"]))
    if cc["dates_only_in_en"]:
        lines.append("Dates (month, day) only in EN: " + str(cc["dates_only_in_en"]))
    if cc["dates_only_in_pt"]:
        lines.append("Dates (month, day) only in PT: " + str(cc["dates_only_in_pt"]))
    return "\n".join(lines) or "No number or date differences found."


# ------------------------------------------------------------ layer 2: AI judge (v2)

JUDGE_VERSION = "v2-2026-10-29"

# Categories (v2). Only the first four count as issues.
CATEGORIES = {
    "contradiction": "Direct contradiction",
    "changed_value": "Changed number, date, unit, currency or jurisdiction",
    "consequential_omission": "Consequential omission",
    "terminology_register": "Terminology or register issue",
    "harmless_variation": "Harmless elaboration / acceptable variation",
    "insufficient_evidence": "Insufficient evidence",
}
ISSUE_CATEGORIES = ("contradiction", "changed_value", "consequential_omission",
                    "terminology_register")
OLD_TYPE_MAP = {"fact_drift": "contradiction", "missing": "consequential_omission",
                "language_trap": "terminology_register"}

JUDGE_INSTRUCTIONS = """You are an audit judge for cross-lingual consistency. You do NOT give medical, legal or financial advice. You compare two answers that the same AI model gave to the same question, asked once in English (EN) and once in Brazilian Portuguese (PT-BR). The two questions have the same meaning, jurisdiction (United States) and register.

Your job: find MEANINGFUL differences, the ones that would leave an English reader and a Portuguese reader with different facts or different actions. Neither language is automatically correct; do not treat English as the reference.

Classify each finding with exactly one category:
- "contradiction": the two answers make incompatible claims under the SAME conditions.
- "changed_value": a number, dose, unit, date, deadline, currency, threshold or jurisdiction is different.
- "consequential_omission": information present in one answer and absent in the other that CHANGES the answer to the question (a warning, a condition, a right, a deadline, a step the reader needs).
- "terminology_register": the wording itself is wrong or misleading: a false friend, a literal calque, a Brazilian legal or product concept presented as if it were the US one, or a technical term in text meant for patients/families.
- "harmless_variation": extra detail, examples, tone, order or an acceptable paraphrase that does not change what the reader should know or do.
- "insufficient_evidence": you suspect a difference but the answers are too vague to be sure.

Rules:
- Extra information is NOT a contradiction. A general rule in one answer and a conditional exception in the other can both be true: report that only if leaving out the exception changes the answer to the question (then it is consequential_omission).
- A Brazilian analogy is not automatically wrong. It is a problem only if it REPLACES the US concept or tells the reader the Brazilian rule applies in the US.
- Brazilian Portuguese writes 2,5 for 2.5 and 1.200 for 1,200. That is NOT a difference.
- Copy the evidence EXACTLY, word for word, from each answer (a short passage, at most 40 words, no paraphrase, no added words). For an omission, quote the passage from the answer that has it and write "(missing)" for the other. Inside the JSON, use single quotes inside text or escape double quotes.
- Say which language is likely problematic: "EN", "PT", "both" or "unclear".
- Do not report the same issue twice.
{trap_block}
Severity: "high" if it could cause harm, delay needed care, cause a wrong dose, a missed deadline, a lost right or a costly financial mistake; "medium" if it could confuse or mislead; "low" otherwise.

Reply with ONLY this JSON, no other text:
{{"findings": [{{"category": "...", "severity": "high" | "medium" | "low", "problem_language": "EN" | "PT" | "both" | "unclear", "en_quote": "...", "pt_quote": "...", "explanation": "one short sentence in English: what differs", "consequence": "one short sentence: what could happen to the reader", "trap_id": "trap id or null"}}]}}
If there are no meaningful differences, reply {{"findings": []}}."""

TRAP_BLOCK = """- Also check the PT-BR answer against this INTERPRETER TRAP FILE, written and verified by a professional medical and legal interpreter. If the PT-BR answer uses a trap rendering IN THE MEANING LISTED (even when the facts match the EN answer), report it as "terminology_register" with its trap_id. A word that looks similar is only a trap when it is used with the wrong meaning in that context.
INTERPRETER TRAP FILE:
{traps}
"""


def build_judge_prompt(question, answer_en, answer_pt, domain, use_traps, cc):
    trap_block = TRAP_BLOCK.format(traps=traps_as_text(traps_for(domain))) if use_traps else ""
    instructions = JUDGE_INSTRUCTIONS.format(trap_block=trap_block)
    body = (
        f"QUESTION (EN): {question['question_en']}\n"
        f"QUESTION (PT-BR): {question['question_pt']}\n\n"
        f"ANSWER (EN):\n{answer_en}\n\n"
        f"ANSWER (PT-BR):\n{answer_pt}\n\n"
        f"CODE CHECK RESULTS (exact number/date comparison; verify, they can be false alarms "
        f"when a number is just written differently):\n{code_checks_text(cc)}"
    )
    return instructions, body


# ------------------------------------------------------------ evidence checks

MISSING = {"", "(missing)", "missing", "(ausente)", "n/a", "none", "null"}
_WORD = re.compile(r"\w+", re.UNICODE)


def _plain(text):
    """Answer text as shown to the reviewer: markdown emphasis removed, typographic quotes unified."""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"[*_`]+", "", t)
    for a, b in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("–", "-"), ("—", "-")):
        t = t.replace(a, b)
    return t


def quote_pattern(quote):
    """Regex that matches the quote's words in order, ignoring punctuation, spacing and case."""
    words = _WORD.findall(_plain(quote))
    if not words:
        return None
    return re.compile(r"[\W_]*".join(re.escape(w) for w in words), re.IGNORECASE | re.UNICODE)


def quote_spans(quote, answer):
    """Character spans of the quote inside the plain answer text, or None if not found.
    A quote made of fragments joined by '...' must have every fragment present, in order."""
    if (quote or "").strip().lower() in MISSING:
        return []
    text = _plain(answer)
    frags = [f for f in re.split(r"\.\.\.|…", _plain(quote)) if len(_WORD.findall(f)) >= 2]
    if not frags:
        frags = [_plain(quote)]
    spans, pos = [], 0
    for frag in frags:
        pat = quote_pattern(frag)
        if pat is None:
            continue
        m = pat.search(text, pos)
        if not m:
            return None
        spans.append((m.start(), m.end()))
        pos = m.end()
    return spans


def validate_finding(f, answer_en, answer_pt):
    """Mark whether each quoted passage really exists in the saved answer."""
    en = quote_spans(f["en_quote"], answer_en)
    pt = quote_spans(f["pt_quote"], answer_pt)
    en_missing = (f["en_quote"] or "").strip().lower() in MISSING
    pt_missing = (f["pt_quote"] or "").strip().lower() in MISSING
    if en_missing and pt_missing:
        support = "unsupported"
    elif (en is None and not en_missing) or (pt is None and not pt_missing):
        support = "unsupported"
    else:
        support = "supported"
    f["support"] = support
    f["en_found"] = None if en_missing else en is not None
    f["pt_found"] = None if pt_missing else pt is not None
    return f


def _key(s):
    return " ".join(_WORD.findall(_plain(s).lower()))[:120]


def finding_id(qid, use_traps, f):
    """Stable ID: same question, mode, category and evidence -> same ID across reloads."""
    raw = f"{qid}|{'with' if use_traps else 'without'}|{f['category']}|{_key(f['en_quote'])}|{_key(f['pt_quote'])}"
    return f"{qid}-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def dedupe(findings):
    seen, out = set(), []
    for f in findings:
        k = (_key(f["en_quote"]), _key(f["pt_quote"]))
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    return out


def parse_judge_json(text):
    """Pull the JSON object out of the judge's reply, even with extra text around it."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        return None
    chunk = text[start:end + 1] if end > start else text[start:]
    try:
        data = json.loads(chunk)
    except json.JSONDecodeError:
        try:  # repair common slips: unescaped quotes, trailing commas, cut-off endings
            from json_repair import repair_json
            data = repair_json(chunk, return_objects=True)
        except Exception:
            return None
    if not isinstance(data, dict) or "findings" not in data:
        return None
    clean = []
    for f in data.get("findings") or []:
        if not isinstance(f, dict):
            continue
        cat = f.get("category") or OLD_TYPE_MAP.get(f.get("type"), "insufficient_evidence")
        if cat not in CATEGORIES:
            cat = OLD_TYPE_MAP.get(cat, "insufficient_evidence")
        sev = f.get("severity", "medium")
        sev = sev if sev in ("high", "medium", "low") else "medium"
        lang = f.get("problem_language", "unclear")
        lang = lang if lang in ("EN", "PT", "both", "unclear") else "unclear"
        trap = f.get("trap_id")
        clean.append({
            "category": cat,
            "severity": sev,
            "problem_language": lang,
            "en_quote": str(f.get("en_quote") or ""),
            "pt_quote": str(f.get("pt_quote") or ""),
            "explanation": str(f.get("explanation") or ""),
            "consequence": str(f.get("consequence") or ""),
            "trap_id": trap if trap not in (None, "null", "", "None") else None,
        })
    return {"findings": clean}


def judge(question, answer_en, answer_pt, chat, use_traps=True):
    """
    chat(system, user) -> final text from the Meta model.
    Returns {"findings": [...], "code_checks": ..., "use_traps": bool, "judge_version": str}
    Each finding is checked against the saved answers ("support") and gets a stable "id".
    """
    cc = code_checks(answer_en, answer_pt)
    system, user = build_judge_prompt(question, answer_en, answer_pt,
                                      question["domain"], use_traps, cc)
    reply = chat(system, user)
    parsed = parse_judge_json(reply)
    if parsed is None:  # one retry asking for JSON only
        reply = chat(system, user + "\n\nReturn ONLY the JSON object.")
        parsed = parse_judge_json(reply) or {
            "findings": [],
            "error": "The judge did not return readable JSON.",
            "raw": (reply or "(empty reply)")[:2000],
        }
    findings = dedupe(parsed["findings"])
    for f in findings:
        validate_finding(f, answer_en, answer_pt)
        f["id"] = finding_id(question["id"], use_traps, f)
    parsed["findings"] = findings
    parsed["code_checks"] = cc
    parsed["use_traps"] = use_traps
    parsed["judge_version"] = JUDGE_VERSION
    return parsed


def is_issue(f):
    """Counts as an automated issue flag: an issue category with evidence that exists."""
    return f["category"] in ISSUE_CATEGORIES and f.get("support") == "supported"
