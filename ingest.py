import argparse
import glob
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import StatusCode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from embed_client import dim, embed

load_dotenv()

# The same provider setup as mcp/server.py's, written again because that image builds from mcp/ alone. The provider's
# own exit hook flushes the last spans, so no explicit shutdown is needed.
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006")
provider = TracerProvider(resource=Resource.create({"service.name": "ingest"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(timeout=2), schedule_delay_millis=1000))
trace.set_tracer_provider(provider)
# Phoenix being down must cost nothing, log noise included.
logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)
tracer = trace.get_tracer("ingest")

qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
COLLECTION = os.environ.get("COLLECTION", "notes")
DATA = Path("data")
INGESTED = DATA / "ingested"
# Splits at paragraph breaks first, then line breaks, then spaces, so chunks rarely end mid-sentence. 1500 characters
# keeps a recipe or a section with its heading in one chunk; the embedder truncates at 2048 tokens, about 6000.
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def normalize_tags(tags):
    """Lowercase, single-spaced, non-empty, deduplicated in first-seen order. Written again in mcp/server.py,
    which builds from mcp/ alone, so the search filter normalises the same way."""
    tags = [tags] if isinstance(tags, str) else tags or []
    return list(dict.fromkeys(t for t in (" ".join(str(t).split()).lower() for t in tags) if t))


def split_frontmatter(path):
    """Returns (metadata, body). Metadata is the YAML between --- fences at the top of the file, {} without one."""
    raw = path.read_text()
    m = FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    try:
        return yaml.safe_load(m.group(1)) or {}, raw[m.end():]
    except yaml.YAMLError as e:
        raise SystemExit(f"{path}: malformed frontmatter: {e}") from e


def upsert_all(points, batch_size=100):
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=COLLECTION, points=points[i:i + batch_size])

def ingest_file(path):
    with tracer.start_as_current_span(
        f"ingest {path.name}", attributes={"openinference.span.kind": "CHAIN", "input.value": str(path)}
    ) as span:
        try:
            return _ingest_file(path, span)
        except BaseException as e:
            # SystemExit from malformed frontmatter is not an Exception, so OpenTelemetry would otherwise leave
            # the failed file unmarked.
            span.set_status(StatusCode.ERROR, str(e))
            raise


def _ingest_file(path, span):
    # The source is where the file ends up, so a note dropped into data/ again replaces its old chunks.
    dest = path if path.is_relative_to(INGESTED) else INGESTED / path.relative_to(DATA)
    source = str(dest)
    meta, body = split_frontmatter(path)
    topic = meta.get("topic")
    document_date = meta.get("document_date")
    chunks = splitter.split_text(body)
    span.set_attribute("ingest.chunks", len(chunks))
    # The topic rides along in each chunk's vector, so a chunk is found by what its document is about too.
    vectors = embed([f"{topic}\n\n{text}" if topic else text for text in chunks])
    payload = {
        "source": source,
        "tags": normalize_tags(meta.get("tags")),
        "topic": topic,
        "document_date": None if document_date is None else str(document_date),
        "date_ingested": datetime.now(UTC).isoformat(),
    }
    with tracer.start_as_current_span("store in qdrant"):
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
        )
        upsert_all([
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{i}")), vector=vec, payload={"text": text, **payload})
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
# Every run, not only on creation, so a collection built before tags existed gets the index without a rebuild.
qdrant.create_payload_index(COLLECTION, "tags", PayloadSchemaType.KEYWORD)

chunks = sum(ingest_file(p) for p in paths)
print(f"Ingested {chunks} chunks from {len(paths)} files into {COLLECTION!r}")
