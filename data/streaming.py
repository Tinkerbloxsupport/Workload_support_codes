import requests
import json
import os
import sys
import numpy as np

# Fixed __file__ variable name
DATA_FILE = os.path.join(os.path.dirname(__file__), "data.txt")

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3:8b"

def load_data(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def get_embedding(text):
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )
    return r.json()["embedding"]

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def find_context(question, lines):
    print("Embedding with nomic-embed-text...")
    q_emb = get_embedding(question)
    
    # Calculate scores and sort
    scored = sorted(
        [(cosine(q_emb, get_embedding(l)), l) for l in lines],
        key=lambda x: x[0],
        reverse=True
    )
    
    top = [l for _, l in scored[:3]]
    
    print("Top 3 context lines found\n")
    return "\n".join(top)

def stream_answer(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    print(f"Streaming answer ({GEN_MODEL}):\n" + "-" * 50)
    
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": True
        },
        stream=True,
        timeout=120
    ) as resp:
        for chunk in resp.iter_lines():
            if chunk:
                data = json.loads(chunk)
                print(data.get("response", ""), end="", flush=True)
                
                if data.get("done"):
                    break
    print("\n" + "-" * 50)

def run():
    print("=" * 55)
    print("PIPELINE 3 — Streaming (Nomic + LLaMA3)")
    print("=" * 55)

    if not os.path.exists(DATA_FILE):
        print(f"[!] Error: {DATA_FILE} not found.")
        return

    lines = load_data(DATA_FILE)
    
    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "How is AI used in healthcare?"
    )
    
    print(f"Loaded {len(lines)} lines")
    print(f"Question: {question}\n")
    
    context = find_context(question, lines)
    stream_answer(question, context)
    
    print("\nStreaming pipeline complete.")

if __name__ == "__main__":
    run()
