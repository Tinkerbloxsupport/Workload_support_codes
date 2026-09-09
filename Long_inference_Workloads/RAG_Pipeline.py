#!/usr/bin/env python3
"""
RAG benchmark - averaged throughput, with per-question detail AND answers.
Each "inference" runs all 10 questions separately, prints each question's
answer + eval_count + duration, then reports the AVERAGE eval_count, AVERAGE
duration, and the resulting throughput for that batch. Repeats for N inferences.

  average eval_count = (sum of 10 eval_counts) / 10
  average duration   = (sum of 10 eval_durations in seconds) / 10
  throughput         = average eval_count / average duration

Files needed: rag.py, helpdesk.txt, questions.txt
Usage:
  python3 rag.py        # 30 inferences (default)
  python3 rag.py 1      # one inference (recommended first, to check answers)
"""

import math, re, sys, time, json, urllib.request, urllib.error
from datetime import datetime, timezone

# ---- Config -----------------------------------------------------------------
OLLAMA      = "http://127.0.0.1:8083"
DOCS_FILE   = "helpdesk.txt"
QENS_FILE   = "questions.txt"
MODEL       = "llama3:70b"
EMBED_MODEL = "nomic-embed-text"
THRESHOLD   = 0.55
RUNS        = 30
RETRIES     = 4
BACKOFF     = 1.5
TIMEOUT     = 1800

# ---- Ollama helpers ---------------------------------------------------------
def post(path, payload):
    data = json.dumps(payload).encode()
    url = OLLAMA + path
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read())
        except Exception as e:
            last_err = e
            print(f"    [attempt {attempt}/{RETRIES}] call to {path} failed: {type(e).__name__}: {e}")
            if attempt < RETRIES:
                wait = BACKOFF * (2 ** (attempt - 1))
                print(f"    retrying in {wait:.0f}s ...")
                time.sleep(wait)
    raise RuntimeError(f"Ollama request to {url} failed after {RETRIES} attempts: {last_err}")

def embed(text):
    return post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})["embedding"]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

# ---- File loading -----------------------------------------------------------
def load_docs(path):
    blocks = re.split(r"\n\s*\n", open(path).read())
    docs   = []
    for b in blocks:
        b = b.strip()
        m = re.match(r"^\d+\.\s+(.*)", b)
        if m:
            lines = b.split("\n")
            docs.append({"title": m.group(1).strip(),
                         "text":  " ".join(l.strip() for l in lines[1:]).strip()})
    return docs

def load_questions(path):
    return [ln.strip() for ln in open(path) if ln.strip()]

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---- Prepare each question's prompt once (RAG match) ------------------------
def prepare(questions, docs):
    prepared = []
    for i, q in enumerate(questions, 1):
        qvec  = embed("search_query: " + q)
        best  = max(docs, key=lambda d: cosine(qvec, d["vec"]))
        score = cosine(qvec, best["vec"])
        if score >= THRESHOLD:
            prompt = f"Answer using only the context.\n\nContext: {best['text']}\n\nQuestion: {q}"
        else:
            prompt = q
        prepared.append({"q": q, "prompt": prompt, "note": best["title"], "score": score})
        print(f"  Q{i} matched: {best['title']:34} (score {score:.3f})")
    return prepared

# ---- Main -------------------------------------------------------------------
def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else RUNS

    docs      = load_docs(DOCS_FILE)
    questions = load_questions(QENS_FILE)
    print(f"Loaded {len(docs)} documents from {DOCS_FILE}")
    print(f"Loaded {len(questions)} questions from {QENS_FILE}")

    print("Embedding documents once with nomic-embed-text ...")
    for d in docs:
        d["vec"] = embed("search_document: " + d["title"] + ". " + d["text"])

    print("\nMatching each question to its note (done once) ...")
    prepared = prepare(questions, docs)
    nq = len(prepared)

    print(f"\nRunning {runs} inferences on {MODEL} "
          f"(each = avg of {nq} questions run separately) ...\n")

    for r in range(1, runs + 1):
        start_iso = utc_now()
        sum_tokens = 0
        sum_ev_sec = 0.0

        print(f"========== inference #{r}  (avg of {nq} questions) ==========")
        print(f"  start time : {start_iso}\n")

        for i, item in enumerate(prepared, 1):
            resp   = post("/api/generate",
                          {"model": MODEL, "prompt": item["prompt"], "stream": False})
            answer = resp.get("response", "").strip()
            ev_n   = resp.get("eval_count", 0)
            ev_s   = resp.get("eval_duration", 0) / 1e9      # this question's gen time (sec)
            tps    = ev_n / ev_s if ev_s else 0
            sum_tokens += ev_n
            sum_ev_sec += ev_s

            print(f"  Q{i} [{item['note']}]  tokens {ev_n}  duration {ev_s:.2f}s  {tps:.2f} tok/s")
            print(f"    Q: {item['q']}")
            print(f"    A: {answer}\n")

        end_iso = utc_now()

        avg_tokens   = sum_tokens / nq
        avg_duration = sum_ev_sec / nq
        throughput   = avg_tokens / avg_duration if avg_duration else 0

        print(f"  end time   : {end_iso}")
        print(f"  -- averages over {nq} questions --")
        print(f"  average eval_count      : {avg_tokens:.1f}")
        print(f"  average duration (sec)  : {avg_duration:.3f}")
        print(f"  throughput (tok/s)      : {throughput:.2f}")
        print()

if __name__ == "__main__":
    main()
