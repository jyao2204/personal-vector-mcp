import argparse
import glob
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from embed_client import dim, embed

load_dotenv()

qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
COLLECTION = os.environ.get("COLLECTION", "notes")
DATA = Path("data")
INGESTED = DATA / "ingested"
# Splits at paragraph breaks first, then line breaks, then spaces, so chunks rarely end mid-sentence.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def upsert_all(points, batch_size=100):
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=COLLECTION, points=points[i:i + batch_size])

def ingest_file(path):
    # The source is where the file ends up, so a note dropped into data/ again replaces its old chunks.
    dest = path if path.is_relative_to(INGESTED) else INGESTED / path.relative_to(DATA)
    source = str(dest)
    chunks = splitter.split_text(path.read_text())
    vectors = embed(chunks)
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
    )
    upsert_all([
        PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{i}")), vector=vec, payload={"text": text, "source": source})
        for i, (text, vec) in enumerate(zip(chunks, vectors))
    ])
    # Moved only once its chunks are stored, so a failed run leaves the rest in data/ for the next one.
    dest.parent.mkdir(parents=True, exist_ok=True)
    path.replace(dest)
    return len(chunks)

parser = argparse.ArgumentParser(description="Embed new notes under data/ into Qdrant and move them into data/ingested/.")
parser.add_argument("--rebuild", action="store_true", help="drop the collection and re-ingest every note, data/ingested/ included")
args = parser.parse_args()

paths = sorted(Path(p) for ext in ("md", "txt") for p in glob.glob(f"{DATA}/**/*.{ext}", recursive=True))
if args.rebuild:
    qdrant.delete_collection(COLLECTION)
else:
    paths = [p for p in paths if not p.is_relative_to(INGESTED)]
if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim(), distance=Distance.COSINE),
    )

chunks = sum(ingest_file(p) for p in paths)
print(f"Ingested {chunks} chunks from {len(paths)} files into {COLLECTION!r}")
