import requests
import json
import csv
import os
import re

# Fixed __file__ variable name with double underscores
DATA_FILE = os.path.join(os.path.dirname(__file__), "data.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "batch_scores.csv")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

def load_data(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def score_line(text):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "think": False,
            "messages": [
                {
                    "role": "user",
                    "content": f'''You are a strict and literal relevance scorer. 

STRICT RULES — follow exactly: 

Score 0-2 : sentence has NO mention of AI, ML, deep learning, neural networks, NLP, data science, or any AI technology 
Score 3-5 : sentence INDIRECTLY mentions something that could relate to AI (e.g. data, computers, automation) 
Score 6-8 : sentence DIRECTLY mentions one AI concept or application 
Score 9-10 : sentence is fully and specifically about AI with technical depth 

IMPORTANT: 

A sentence about corona virus = 0-2 (not AI) 
A sentence about a country = 0-2 (not AI) 
A sentence about someone's ambition = 0-2 (not AI) 
A sentence about cats, food, weather = 0-2 (not AI) 
Do NOT find indirect AI connections. Judge ONLY what is LITERALLY written. 

Reply ONLY as JSON, nothing else: {{"score": 5, "reason": "one line explanation"}} 

Sentence: {text}'''
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 150
            }
        },
        timeout=300
    )
    
    data = resp.json()
    raw = data.get("message", {}).get("content", "").strip()
    
    # Extract JSON from potential model chatter
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    return {"score": "N/A", "reason": raw}

def run():
    print("=" * 55)
    print("PIPELINE 2 — Batch Scoring (Qwen3)")
    print("=" * 55)

    if not os.path.exists(DATA_FILE):
        print(f"[!] Error: {DATA_FILE} not found.")
        return

    lines = load_data(DATA_FILE)
    print(f"Loaded {len(lines)} lines\n")

    rows = []

    for i, line in enumerate(lines, 1):
        print(f"[{i}/{len(lines)}] {line}")

        r = score_line(line)

        print(f"           Score : {r.get('score')}/10")
        print(f"           Reason: {r.get('reason')}\n")

        rows.append({
            "line": i,
            "text": line,
            "score": r.get("score"),
            "reason": r.get("reason")
        })

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line", "text", "score", "reason"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done → {OUTPUT_FILE}")

if __name__ == "__main__":
    run()
