import os
import glob

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from dotenv import load_dotenv

from embed_client import embed

load_dotenv()

qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

def chunk_text(text, size=450, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

records = []
paths = glob.glob("data/**/*.md", recursive=True) + glob.glob("data/**/*.txt", recursive=True)
for path in paths:
    with open(path) as f:
        text = f.read()
    for chunk in chunk_text(text):
        records.append((path, chunk))

texts = [r[1] for r in records]

vectors = embed(texts)

if qdrant.collection_exists("practice_docs"):
    qdrant.delete_collection("practice_docs")
qdrant.create_collection(
    collection_name="practice_docs",
    vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
)

points = [
    PointStruct(id=i, vector=vec, payload={"text": text, "source": source})
    for i, ((source, text), vec) in enumerate(zip(records, vectors))
]

def upsert_all(points, batch_size=100):
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name="practice_docs", points=points[i:i + batch_size])

upsert_all(points)
print(f"Ingested {len(points)} chunks from ./data/")
