import argparse
import requests
from qdrant_client import QdrantClient

OLLAMA_URL  = 'http://localhost:11434'
QDRANT_URL  = 'http://localhost:6333'
EMBED_MODEL = 'nomic-embed-text'

def get_embedding(text):
    r = requests.post(
        f'{OLLAMA_URL}/api/embeddings',
        json={'model': EMBED_MODEL, 'prompt': text}, 
        timeout=60
    )
    r.raise_for_status()
    return r.json()['embedding']

def search(query, collection, top_k):
    client = QdrantClient(url=QDRANT_URL)
    results = client.query_points(
        collection_name=collection,
        query=get_embedding(query),
        limit=top_k,
        with_payload=True
    ).points
    
    seen, unique = set(), []
    for h in results:
        text = h.payload.get('text', '')
        if text not in seen:
            seen.add(text)
            unique.append({
                'score': round(h.score, 4),
                'text': text,
                'source': h.payload.get('source', ''),
                'chunk_index': h.payload.get('chunk_index', 0)
            })
    return unique

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query')
    parser.add_argument('--collection', default='my_docs')
    parser.add_argument('--top', default=5, type=int)
    args = parser.parse_args()
    
    print(f'[->] Searching: "{args.query}"')
    results = search(args.query, args.collection, args.top)
    
    if not results:
        print('[!] No results. Run embed.py first.')
    
    for i, r in enumerate(results, 1):
        print(f'-- Result {i} (score: {r["score"]}) --')
        print(f'   Source: {r["source"]} [chunk {r["chunk_index"]}]')
        print(f'   Text: {r["text"][:300]}')
