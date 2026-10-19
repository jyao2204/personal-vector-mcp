import os
import sys

from qdrant_client import QdrantClient
import numpy as np

from dotenv import load_dotenv

from embed_client import embed

load_dotenv()

qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

if len(sys.argv) < 2:
    sys.exit('usage: uv run phase1_query.py "your question"')
query = " ".join(sys.argv[1:])

query_vec = embed([query], input_type="query")[0]

results = qdrant.query_points(collection_name="practice_docs", query=query_vec, limit=5)
for r in results.points:
    print(r.score, r.payload["source"], r.payload["text"][:100])
