import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from embed_client import embed

load_dotenv()

qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

if len(sys.argv) < 2:
    sys.exit('usage: uv run query.py "your question"')
query = " ".join(sys.argv[1:])

query_vec = embed([query], input_type="query")[0]

results = qdrant.query_points(collection_name=os.environ.get("COLLECTION", "notes"), query=query_vec, limit=5)
for r in results.points:
    print(r.score, r.payload["source"], " ".join(r.payload["text"].split())[:100])
