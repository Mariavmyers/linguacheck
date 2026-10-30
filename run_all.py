"""
Command-line full run (same as the app's "Full run" button).

Usage:  python run_all.py            (all verified questions)
        python run_all.py M01 L03    (only these IDs)

Saves the run (both full answers + judgments with and without the trap file)
to permanent storage: the private HF dataset if HF_TOKEN is set, else local_data/.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import app
import audit
import storage

WORKERS = 4


def main(ids):
    client = app.get_client()
    qs = [q for qid, q in app.QUESTIONS.items() if not ids or qid in ids]
    run_id = audit.new_run_id("full")
    records = []
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(app.answer_and_judge, q, client): q["id"] for q in qs}
        for n, fut in enumerate(as_completed(futures), 1):
            qid = futures[fut]
            try:
                records.append(fut.result())
                print(f"[{n}/{len(qs)}] {qid} done")
            except Exception as e:
                print(f"[{n}/{len(qs)}] {qid}: ERROR {e}")
    records.sort(key=lambda r: r["id"])
    meta = audit.run_meta(run_id, "full", app.MODEL, app.MAX_TOKENS, n=len(records))
    storage.save_run(run_id, records, meta)
    print(f"Saved run {run_id} to {storage.where()}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
