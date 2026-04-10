import argparse
import uuid
from pathlib import Path
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

OLLAMA_URL = 'http://localhost:11434'
QDRANT_URL = 'http://localhost:6333'
EMBED_MODEL = 'nomic-embed-text'
VECTOR_SIZE = 768
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def read_file(path):
    if path.suffix.lower() == '.pdf':
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return '\n'.join(page.extract_text() or '' for page in reader.pages)
    return path.read_text(encoding='utf-8', errors='ignore')

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+size].strip())
        start += size - overlap
    return [c for c in chunks if c]

def get_embedding(text):
    r = requests.post(
        f'{OLLAMA_URL}/api/embeddings',
        json={'model': EMBED_MODEL, 'prompt': text}, 
        timeout=60
    )
    r.raise_for_status()
    return r.json()['embedding']

def ensure_collection(client, collection):
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f'  [+] Created collection: {collection}')
    else:
        print(f'  [~] Using existing collection: {collection}')

def embed_documents(docs_dir, collection):
    docs_path = Path(docs_dir)
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, collection)
    
    files = list(docs_path.glob('**/*.txt')) + list(docs_path.glob('**/*.pdf'))
    print(f'[->] Found {len(files)} file(s). Starting embedding...')
    
    total_chunks = 0
    for file in files:
        print(f'  Processing: {file.name}')
        text = read_file(file)
        if not text.strip():
            continue
            
        chunks = chunk_text(text)
        print(f'    -> {len(chunks)} chunks')
        
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=get_embedding(chunk),
                payload={'text': chunk, 'source': file.name, 'chunk_index': i}
            ) for i, chunk in enumerate(chunks)
        ]
        
        client.upsert(collection_name=collection, points=points)
        total_chunks += len(chunks)
        print(f'    Stored {len(chunks)} chunks')
        
    print(f'[OK] Done! {total_chunks} total chunks embedded into {collection}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', default='./documents')
    parser.add_argument('--collection', default='my_docs')
    args = parser.parse_args()
    embed_documents(args.docs, args.collection)
