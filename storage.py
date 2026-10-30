"""
Permanent storage for LinguaCheck: a PRIVATE Hugging Face dataset
(default hugmarie/linguacheck-data), written with the HF_TOKEN secret.

Layout inside the dataset:
  runs/<run_id>.jsonl         one line per question: both full answers + judgments
  runs/<run_id>.meta.json     model, settings, time, judge version, source run
  reviews/reviews.json        the interpreter's decisions, keyed by finding id
  exports/...                 review sheet, corrections, PDF report

Without HF_TOKEN (e.g. on a laptop) everything goes to the local folder
"local_data/" instead, so the app still works.
"""

import json
import os
import threading
from pathlib import Path

DATA_REPO = os.getenv("DATA_REPO", "hugmarie/linguacheck-data")
HERE = Path(__file__).parent
LOCAL = HERE / "local_data"
_lock = threading.Lock()


def _token():
    return os.getenv("HF_TOKEN", "").strip()


def remote_enabled():
    return bool(_token())


def where():
    return (f"private Hugging Face dataset {DATA_REPO}" if remote_enabled()
            else "local folder local_data/ (no HF_TOKEN set: not permanent on the Space)")


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=_token())


# ------------------------------------------------------------ low level

def write_text(path, text, message="Update"):
    local = LOCAL / path
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text, encoding="utf-8")
    if remote_enabled():
        _api().upload_file(path_or_fileobj=text.encode("utf-8"), path_in_repo=path,
                           repo_id=DATA_REPO, repo_type="dataset", commit_message=message)


def write_bytes(path, data, message="Update"):
    local = LOCAL / path
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    if remote_enabled():
        _api().upload_file(path_or_fileobj=data, path_in_repo=path,
                           repo_id=DATA_REPO, repo_type="dataset", commit_message=message)


def read_text(path):
    if remote_enabled():
        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(DATA_REPO, path, repo_type="dataset", token=_token(),
                                force_download=True)
            return Path(p).read_text(encoding="utf-8")
        except Exception:
            pass
    local = LOCAL / path
    return local.read_text(encoding="utf-8") if local.exists() else None


def list_files(prefix):
    names = set()
    if remote_enabled():
        try:
            names |= {f for f in _api().list_repo_files(DATA_REPO, repo_type="dataset")
                      if f.startswith(prefix)}
        except Exception:
            pass
    base = LOCAL / prefix
    if base.exists():
        names |= {str(p.relative_to(LOCAL)).replace(os.sep, "/") for p in base.rglob("*")
                  if p.is_file()}
    return sorted(names)


# ------------------------------------------------------------ runs

def save_run(run_id, records, meta):
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    write_text(f"runs/{run_id}.jsonl", body, f"Save run {run_id}")
    write_text(f"runs/{run_id}.meta.json", json.dumps(meta, ensure_ascii=False, indent=1),
               f"Save run {run_id} metadata")


def list_runs():
    return sorted({f[len("runs/"):-len(".jsonl")] for f in list_files("runs/")
                   if f.endswith(".jsonl")}, reverse=True)


def load_run(run_id):
    text = read_text(f"runs/{run_id}.jsonl")
    if not text:
        return [], {}
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    meta_text = read_text(f"runs/{run_id}.meta.json")
    return records, (json.loads(meta_text) if meta_text else {})


# ------------------------------------------------------------ reviews

REVIEWS_PATH = "reviews/reviews.json"
_reviews_cache = None


def load_reviews(refresh=False):
    global _reviews_cache
    with _lock:
        if _reviews_cache is None or refresh:
            text = read_text(REVIEWS_PATH)
            _reviews_cache = json.loads(text) if text else {}
        return dict(_reviews_cache)


def save_review(finding_id, review):
    """review: decision, category, severity, problem_language, notes, source,
    correct_version, reviewed_at. Saved permanently (one commit per save)."""
    global _reviews_cache
    with _lock:
        if _reviews_cache is None:
            text = read_text(REVIEWS_PATH)
            _reviews_cache = json.loads(text) if text else {}
        _reviews_cache[finding_id] = review
        body = json.dumps(_reviews_cache, ensure_ascii=False, indent=1)
    write_text(REVIEWS_PATH, body, f"Review {finding_id}: {review.get('decision')}")
    return review
