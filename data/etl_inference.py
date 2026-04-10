import argparse
import re
import requests
from search import search

OLLAMA_URL = 'http://localhost:11434'
RERANK_MODEL = 'qwen3:latest'

def rerank_score(query, passage):
    prompt = f'''Score the relevance between query and passage. Return ONLY a number between 0 and 1.

Query: {query} 
Passage: {passage} 
Score:'''
    try:
        r = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={
                'model': RERANK_MODEL,
                'prompt': prompt,
                'stream': False,
                'think': False,
                'options': {'temperature': 0, 'num_predict': 10}
            },
            timeout=60
        )
        r.raise_for_status()
        raw = r.json().get('response', '').strip()
        
        # Remove any thinking tokens if the model generates them
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        
        # Extract the numeric score
        match = re.search(r'0?\.\d+|1\.0|[01]', raw)
        if match:
            return min(max(float(match.group()), 0.0), 1.0)
    except Exception as e:
        print(f'[!] Rerank error: {e}')
        return 0.3
    return 0.3

def rerank(query, results):
    print(f'[->] Reranking {len(results)} results...')
    for r in results:
        r['rerank_score'] = rerank_score(query, r['text'])
        print(f' Score: {r["rerank_score"]:.3f} | {r["source"]} | chunk {r["chunk_index"]}')
    return sorted(results, key=lambda x: x['rerank_score'], reverse=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query')
    parser.add_argument('--collection', default='my_docs')
    parser.add_argument('--top', default=5, type=int)
    args = parser.parse_args()

    print(f'[1/2] Vector search for: "{args.query}"')
    results = search(args.query, args.collection, args.top)

    if not results:
        print('[!] No results. Run embed.py first.')
        exit()

    print(f'[2/2] Reranking with {RERANK_MODEL}...')
    reranked = rerank(args.query, results)

    print('\n-- Final Reranked Results --')
    for i, r in enumerate(reranked, 1):
        print(f' #{i} | vector: {r["score"]} -> rerank: {r["rerank_score"]:.3f}')
        print(f'       Source: {r["source"]} [chunk {r["chunk_index"]}]')
        print(f'       Text: {r["text"][:300]}...')
