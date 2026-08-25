"""
streaming_bench.py — Streaming inference with llama3:70b
10 questions × 30 iterations = 300 streamed answers
CSV saved incrementally after every single answer.
"""

import csv
import os
import time
from datetime import datetime

import ollama

MODEL            = "llama3:70b"
TOTAL_ITERATIONS = 30
CSV_FILE         = "streaming_bench_results.csv"

QUESTIONS = [
    "What is photosynthesis?",
    "Explain gravity in simple terms.",
    "What causes thunder and lightning?",
    "How does the internet work?",
    "What is machine learning?",
    "Why is the sky blue?",
    "How do vaccines work?",
    "What is the speed of light?",
    "Explain the water cycle.",
    "What is DNA?",
]

CSV_FIELDS = [
    "timestamp",
    "iteration",
    "question_num",
    "question",
    "first_token_s",
    "total_time_s",
    "token_count",
    "tokens_per_sec",
]


def init_csv():
    """Write header row once at startup (overwrites any previous run)."""
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    print(f"  [CSV] Logging to {CSV_FILE}\n")


def append_csv(row: dict):
    """Append one result row — called immediately after each answer."""
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def stream_question(iteration: int, q_num: int, question: str):
    print(f"\n  [Q{q_num}] {question}")
    print("  ", end="", flush=True)

    t_start     = time.time()
    first_token = None
    token_count = 0

    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            if first_token is None:
                first_token = time.time() - t_start
            print(token, end="", flush=True)
            token_count += 1

    total_time = time.time() - t_start
    tok_per_s  = round(token_count / total_time, 1) if total_time > 0 else 0

    print(f"\n  First token: {first_token:.2f}s | "
          f"Total: {total_time:.1f}s | "
          f"{token_count} tokens | "
          f"{tok_per_s} tok/s")

    # ── Save to CSV immediately after this answer ──────────────────────
    row = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iteration":      iteration,
        "question_num":   q_num,
        "question":       question,
        "first_token_s":  round(first_token, 3) if first_token else 0,
        "total_time_s":   round(total_time, 3),
        "token_count":    token_count,
        "tokens_per_sec": tok_per_s,
    }
    append_csv(row)

    return first_token, total_time, token_count, tok_per_s


def main():
    print(f"\n{'='*60}")
    print(f"  Streaming Benchmark — {MODEL}")
    print(f"  {len(QUESTIONS)} questions × {TOTAL_ITERATIONS} iterations "
          f"= {len(QUESTIONS) * TOTAL_ITERATIONS} total answers")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    init_csv()

    for iteration in range(1, TOTAL_ITERATIONS + 1):
        print(f"\n{'─'*60}")
        print(f"  Iteration {iteration}/{TOTAL_ITERATIONS}")
        print(f"{'─'*60}")

        for i, question in enumerate(QUESTIONS, start=1):
            stream_question(iteration, i, question)

    print(f"\n{'='*60}")
    print(f"  Done — {TOTAL_ITERATIONS} iterations complete")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results saved to: {CSV_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
