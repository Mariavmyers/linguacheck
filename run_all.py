"""
Run the whole test set: every verified question, EN + PT-BR answers once,
then the judge twice (WITH and WITHOUT the interpreter trap file).

Usage:  python run_all.py            (all 75 questions)
        python run_all.py M01 L03    (only these IDs)

Writes (in results/, never uploaded to GitHub):
  run_<time>.jsonl   every answer and judgment, for the record
  run_<time>.csv     one row per question: flags with vs without the trap file
  review_<time>.csv  one row per flag, with a column for Maria to mark
                     "real" or "false alarm" (this gives the accuracy number)
"""

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import app
import judge

WORKERS = 4  # questions handled at the same time


def run_one(q, client):
    answers = {}
    for lang, field in (("EN", "question_en"), ("PT-BR", "question_pt")):
        answers[lang], _ = app.ask_model(q[field], client=client)
    out = {"id": q["id"], "domain": q["domain"], "risk": q["risk"],
           "answer_en": answers["EN"], "answer_pt": answers["PT-BR"]}
    for use in (True, False):
        out["with_traps" if use else "without_traps"] = judge.judge(
            q, answers["EN"], answers["PT-BR"], app.judge_chat(client), use_traps=use)
    return out


def main(ids):
    client = app.get_client()
    qs = [q for qid, q in app.QUESTIONS.items() if not ids or qid in ids]
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    app.RESULTS_FILE.parent.mkdir(exist_ok=True)
    base = app.RESULTS_FILE.parent
    rows, review = [], []
    with ThreadPoolExecutor(WORKERS) as pool, \
            open(base / f"run_{stamp}.jsonl", "w", encoding="utf-8") as raw:
        futures = {pool.submit(run_one, q, client): q["id"] for q in qs}
        for n, fut in enumerate(as_completed(futures), 1):
            qid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"[{n}/{len(qs)}] {qid}: ERROR {e}")
                continue
            raw.write(json.dumps(r, ensure_ascii=False) + "\n")
            w, wo = r["with_traps"]["findings"], r["without_traps"]["findings"]
            rows.append({
                "id": qid, "domain": r["domain"], "risk": r["risk"],
                "flags_with_traps": len(w),
                "high_with_traps": sum(f["severity"] == "high" for f in w),
                "trap_flags": sum(f["type"] == "language_trap" for f in w),
                "flags_without_traps": len(wo),
                "high_without_traps": sum(f["severity"] == "high" for f in wo),
            })
            for mode, fs in (("with", w), ("without", wo)):
                for f in fs:
                    review.append({"id": qid, "trap_file": mode, **f,
                                   "maria_says (real / false alarm)": ""})
            print(f"[{n}/{len(qs)}] {qid}: {len(w)} flags with traps, {len(wo)} without")

    rows.sort(key=lambda r: r["id"])
    review.sort(key=lambda r: (r["id"], r["trap_file"]))
    for name, data in ((f"run_{stamp}.csv", rows), (f"review_{stamp}.csv", review)):
        if data:
            with open(base / name, "w", newline="", encoding="utf-8-sig") as f:
                wr = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                wr.writeheader()
                wr.writerows(data)
    drift = sum(1 for r in rows if r["flags_with_traps"])
    print(f"\nDone. {len(rows)} questions. Drift flagged on {drift} with the trap file, "
          f"{sum(1 for r in rows if r['flags_without_traps'])} without.")
    print(f"Files: results/run_{stamp}.csv and results/review_{stamp}.csv")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
