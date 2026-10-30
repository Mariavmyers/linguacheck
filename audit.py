"""
Runs, re-judging, honest metrics and exports for LinguaCheck.

Words used on purpose:
  "flag"            = something the automatic judge raised (NOT a proven error)
  "confirmed"       = a flag the interpreter marked "Confirmed issue"
  "false alarm"     = a flag the interpreter marked "False alarm / acceptable variation"
Precision is computed ONLY from flags adjudicated as confirmed or false alarm.
Recall is not reported: unflagged answers have not been reviewed for missed issues.
"""

import csv
import io
import json
from collections import Counter
from datetime import datetime, timezone

import judge

DECISIONS = ["Not reviewed", "Confirmed issue", "False alarm / acceptable variation",
             "Needs specialist review"]
MODES = (("with_traps", "with"), ("without_traps", "without"))


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(kind):
    return f"{kind}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


# ------------------------------------------------------------ building records

def judge_record(q, answer_en, answer_pt, chat):
    """Judge one saved answer pair twice: WITH and WITHOUT the interpreter trap file."""
    rec = {"id": q["id"], "domain": q["domain"], "category": q["category"], "risk": q["risk"],
           "question_en": q["question_en"], "question_pt": q["question_pt"],
           "answer_en": answer_en, "answer_pt": answer_pt}
    for key, _ in MODES:
        rec[key] = judge.judge(q, answer_en, answer_pt, chat, use_traps=(key == "with_traps"))
    return rec


def legacy_to_answers(records, questions):
    """Turn an older run file into answer pairs (answers only; old judgments dropped)."""
    out = []
    for r in records:
        q = questions.get(r["id"], {})
        out.append({"id": r["id"], "domain": r.get("domain", q.get("domain")),
                    "category": r.get("category", q.get("category")),
                    "risk": r.get("risk", q.get("risk")),
                    "question_en": q.get("question_en", ""), "question_pt": q.get("question_pt", ""),
                    "answer_en": r["answer_en"], "answer_pt": r["answer_pt"]})
    return out


# ------------------------------------------------------------ reviews lookup

def evidence_key(f):
    return (judge._key(f.get("en_quote", ""))[:70], judge._key(f.get("pt_quote", ""))[:70])


def review_index(records, reviews):
    """Map evidence -> review, so a decision on a WITH flag also applies to the identical
    WITHOUT flag (same answers, same quotes)."""
    by_evidence = {}
    for r in records:
        for key, _ in MODES:
            for f in r[key]["findings"]:
                if f.get("id") in reviews:
                    by_evidence[(r["id"],) + evidence_key(f)] = reviews[f["id"]]
    return by_evidence


def review_for(r, f, reviews, by_evidence):
    return reviews.get(f.get("id")) or by_evidence.get((r["id"],) + evidence_key(f))


def effective(f, rev):
    """Category/severity after the interpreter's edits."""
    if rev:
        return rev.get("category") or f["category"], rev.get("severity") or f["severity"]
    return f["category"], f["severity"]


# ------------------------------------------------------------ metrics

def metrics(records, reviews):
    by_ev = review_index(records, reviews)
    m = {"questions_tested": len(records), "modes": {}}
    for key, label in MODES:
        flags, q_flagged = [], set()
        cnt = Counter()
        for r in records:
            for f in r[key]["findings"]:
                cnt["all_raw"] += 1
                if f.get("support") != "supported":
                    cnt["unsupported"] += 1
                    continue
                if f["category"] == "harmless_variation":
                    cnt["harmless"] += 1
                    continue
                if f["category"] == "insufficient_evidence":
                    cnt["insufficient"] += 1
                    continue
                flags.append((r, f))
                q_flagged.add(r["id"])
        dec = Counter()
        cat, sev = Counter(), Counter()
        for r, f in flags:
            rev = review_for(r, f, reviews, by_ev)
            d = (rev or {}).get("decision", "Not reviewed")
            dec[d] += 1
            c, s = effective(f, rev)
            cat[c] += 1
            sev[s] += 1
        confirmed = dec["Confirmed issue"]
        false_alarm = dec["False alarm / acceptable variation"]
        adjudicated = confirmed + false_alarm
        m["modes"][label] = {
            "questions_with_flags": len(q_flagged),
            "total_flags": len(flags),
            "unsupported_quotes": cnt["unsupported"],
            "harmless_variation": cnt["harmless"],
            "insufficient_evidence": cnt["insufficient"],
            "by_category": dict(cat), "by_severity": dict(sev),
            "reviewed": len(flags) - dec["Not reviewed"],
            "confirmed": confirmed, "false_alarms": false_alarm,
            "needs_specialist": dec["Needs specialist review"],
            "not_reviewed": dec["Not reviewed"],
            "precision": (confirmed / adjudicated) if adjudicated else None,
            "adjudicated": adjudicated,
            "questions_with_confirmed": len({r["id"] for r, f in flags
                                             if (review_for(r, f, reviews, by_ev) or {})
                                             .get("decision") == "Confirmed issue"}),
        }
    m["guide_effect"] = guide_comparison(records, reviews)
    return m


def guide_comparison(records, reviews):
    """Same saved answers, judged with vs without the trap file: what changed."""
    by_ev = review_index(records, reviews)
    added, removed, reclassified = [], [], []
    for r in records:
        w = {evidence_key(f): f for f in r["with_traps"]["findings"] if judge.is_issue(f)}
        wo = {evidence_key(f): f for f in r["without_traps"]["findings"] if judge.is_issue(f)}
        for k, f in w.items():
            if k not in wo:
                added.append((r, f))
            elif wo[k]["category"] != f["category"]:
                reclassified.append((r, f, wo[k]["category"]))
        for k, f in wo.items():
            if k not in w:
                removed.append((r, f))

    def tally(items):
        d = Counter((review_for(x[0], x[1], reviews, by_ev) or {}).get("decision", "Not reviewed")
                    for x in items)
        return {"count": len(items), "confirmed": d["Confirmed issue"],
                "false_alarms": d["False alarm / acceptable variation"],
                "not_reviewed": d["Not reviewed"]}
    return {"added_by_guide": tally(added), "removed_with_guide": tally(removed),
            "reclassified": len(reclassified),
            "added_ids": [f["id"] for _, f in added]}


def metrics_markdown(m):
    w, wo = m["modes"]["with"], m["modes"]["without"]
    g = m["guide_effect"]

    def pct(a, b):
        return f"{a / b:.0%}" if b else "n/a"
    prec = (f"**{w['precision']:.0%}** ({w['confirmed']} confirmed of {w['adjudicated']} decided)"
            if w["precision"] is not None else "not yet: no flags decided")
    lines = [
        "### Honest numbers (with the interpreter trap file)",
        f"- Questions tested: **{m['questions_tested']}**",
        f"- Questions with at least one automated flag: **{w['questions_with_flags']}** "
        f"({pct(w['questions_with_flags'], m['questions_tested'])} of questions tested)",
        f"- Total automated flags: **{w['total_flags']}** "
        f"(not counted: {w['harmless_variation']} harmless variations, "
        f"{w['insufficient_evidence']} insufficient evidence, "
        f"{w['unsupported_quotes']} with quotes not found in the answers)",
        f"- Reviewed by the interpreter: **{w['reviewed']} of {w['total_flags']}** flags",
        f"- Confirmed issues: **{w['confirmed']}** in {w['questions_with_confirmed']} questions",
        f"- False alarms: **{w['false_alarms']}** · Needs specialist review: "
        f"**{w['needs_specialist']}** · Not reviewed: {w['not_reviewed']}",
        f"- Judge precision on decided flags: {prec}",
        "- Recall: not measured (answers without flags have not been reviewed for missed issues).",
        "",
        "### Does the trap file help? (same saved answers, judged twice)",
        f"- Without the trap file: {wo['total_flags']} flags in {wo['questions_with_flags']} questions",
        f"- With the trap file: {w['total_flags']} flags in {w['questions_with_flags']} questions",
        f"- Flags only found WITH the trap file: **{g['added_by_guide']['count']}** "
        f"(confirmed {g['added_by_guide']['confirmed']}, false alarms "
        f"{g['added_by_guide']['false_alarms']}, not reviewed {g['added_by_guide']['not_reviewed']})",
        f"- Flags only found WITHOUT it: {g['removed_with_guide']['count']} · "
        f"same evidence, different category: {g['reclassified']}",
        "- More flags alone is not an improvement: judge the trap file by how many of its "
        "extra flags you confirm.",
        ("- ⚠️ Many flags appear only in one of the two judgments in BOTH directions: much of this "
         "difference is the judge varying between calls, not the trap file. Only reviewed "
         "decisions can show the trap file's real effect."
         if g["removed_with_guide"]["count"] > 0.5 * max(1, g["added_by_guide"]["count"]) else ""),
        "",
        "_No flag does not mean both answers are correct. Unreviewed flags are possible "
        "differences, not proven errors._",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------ exports

def review_rows(records, reviews, meta):
    by_ev = review_index(records, reviews)
    rows = []
    for r in records:
        for key, label in MODES:
            for f in r[key]["findings"]:
                rev = review_for(r, f, reviews, by_ev) or {}
                rows.append({
                    "run_id": meta.get("run_id", ""), "finding_id": f.get("id", ""),
                    "question_id": r["id"], "domain": r["domain"], "trap_file": label,
                    "judge_category": f["category"], "judge_severity": f["severity"],
                    "problem_language": f.get("problem_language", ""),
                    "quotes_found": f.get("support", ""),
                    "en_quote": f["en_quote"], "pt_quote": f["pt_quote"],
                    "explanation": f["explanation"], "consequence": f.get("consequence", ""),
                    "trap_id": f.get("trap_id") or "",
                    "decision": rev.get("decision", "Not reviewed"),
                    "reviewed_category": rev.get("category", ""),
                    "reviewed_severity": rev.get("severity", ""),
                    "reviewer_problem_language": rev.get("problem_language", ""),
                    "reviewer_notes": rev.get("notes", ""), "source": rev.get("source", ""),
                    "correct_version": rev.get("correct_version", ""),
                    "reviewed_at": rev.get("reviewed_at", ""),
                })
    return rows


def to_csv(rows):
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return "﻿" + buf.getvalue()


def corrections(records, reviews, meta):
    """Expert correction pairs: every flag the interpreter confirmed (WITH-trap judgments)."""
    by_ev = review_index(records, reviews)
    out = []
    for r in records:
        for f in r["with_traps"]["findings"]:
            rev = review_for(r, f, reviews, by_ev) or {}
            if rev.get("decision") != "Confirmed issue":
                continue
            cat, sev = effective(f, rev)
            out.append({
                "finding_id": f.get("id"), "question_id": r["id"], "domain": r["domain"],
                "question_en": r["question_en"], "question_pt": r["question_pt"],
                "model": meta.get("model"), "run_id": meta.get("run_id"),
                "model_answer_en_excerpt": f["en_quote"], "model_answer_pt_excerpt": f["pt_quote"],
                "category": cat, "severity": sev,
                "problem_language": rev.get("problem_language") or f.get("problem_language"),
                "what_is_wrong": rev.get("notes") or f["explanation"],
                "expert_correct_version": rev.get("correct_version", ""),
                "source": rev.get("source", ""),
                "reviewed_by": "Professional medical/legal interpreter (EN/PT-BR)",
                "reviewed_at": rev.get("reviewed_at"),
            })
    return out


def run_meta(run_id, kind, model, max_tokens, answers_from=None, n=0):
    return {"run_id": run_id, "kind": kind, "created_utc": now_utc(), "model": model,
            "temperature": 0, "max_tokens": max_tokens, "judge_version": judge.JUDGE_VERSION,
            "answers_from_run": answers_from or run_id, "n_questions": n,
            "trap_file_size": len(judge.TRAPS),
            "note": "temperature 0 reduces but does not guarantee identical outputs"}
