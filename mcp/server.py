import os
from typing import Annotated

import httpx
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:8080")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION", "notes")
MODEL = "Qwen/Qwen3-Embedding-0.6B"

mcp = MCPServer("notes")
qdrant = QdrantClient(url=QDRANT_URL)


class Hit(BaseModel):
    text: str
    source: str
    score: float
    tags: list[str]
    topic: str | None
    document_date: str | None
    date_ingested: str | None


class Tag(BaseModel):
    tag: str
    count: int


def normalize_tags(tags):
    """Lowercase, single-spaced, non-empty, deduplicated in first-seen order. The same rule as ingest.py's,
    written again because this image builds from mcp/ alone."""
    return list(dict.fromkeys(t for t in (" ".join(str(t).split()).lower() for t in tags) if t))


@mcp.tool()
def list_tags() -> list[Tag]:
    """Every tag on the user's notes with the number of chunks carrying it, most common first. Call it before passing tags to search_notes, so the filter only names tags that exist."""
    try:
        hits = qdrant.facet(collection_name=COLLECTION, key="tags", limit=1000).hits
    except Exception as e:
        raise ToolError(f"qdrant at {QDRANT_URL}, collection {COLLECTION!r} failed: {e}") from e
    return [Tag(tag=h.value, count=h.count) for h in hits]


@mcp.tool()
def search_notes(
    query: Annotated[str, Field(description="What to look for, as a natural-language question or keywords.")],
    k: Annotated[int, Field(description="How many chunks to return, 1 to 20; values outside that range are clamped.")] = 5,
    tags: Annotated[list[str] | None, Field(description="Only search chunks carrying at least one of these tags, as listed by list_tags; case and spacing do not matter. Leave unset for a first search, and set it only when the user names a domain or unfiltered results are off-topic.")] = None,
) -> list[Hit]:
    """Semantic search over the user's notes and decision records. Use it for any question about the user's projects, decisions, integrations, or past work. Returns the k most relevant chunks, each with its source filename, its document's tags and topic, and a cosine similarity score, highest first. Cite the source filenames in the answer."""
    k = max(1, min(k, 20))
    query_filter = Filter(must=[FieldCondition(key="tags", match=MatchAny(any=normalize_tags(tags)))]) if tags else None
    # ToolError text reaches the caller; any other exception is masked as a bare "Error executing tool".
    try:
        r = httpx.post(
            f"{EMBED_URL}/v1/embeddings",
            json={"input": [query], "model": MODEL, "input_type": "query"},
            timeout=60,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise ToolError(f"embedder at {EMBED_URL} failed: {e}") from e
    vec = r.json()["data"][0]["embedding"]
    try:
        points = qdrant.query_points(collection_name=COLLECTION, query=vec, query_filter=query_filter, limit=k).points
    except Exception as e:
        raise ToolError(f"qdrant at {QDRANT_URL}, collection {COLLECTION!r} failed: {e}") from e
    # .get() for the metadata: chunks stored before frontmatter support carry only text and source until a rebuild.
    return [
        Hit(
            text=p.payload["text"],
            source=p.payload["source"],
            score=p.score,
            tags=p.payload.get("tags", []),
            topic=p.payload.get("topic"),
            document_date=p.payload.get("document_date"),
            date_ingested=p.payload.get("date_ingested"),
        )
        for p in points
    ]


if __name__ == "__main__":
    # Binding 0.0.0.0 inside the container turns off the SDK's automatic localhost Host check, so set it
    # explicitly: a page that rebinds its DNS to 127.0.0.1 must not reach the tool. A cloud deploy adds its
    # public hostname here alongside auth.
    app = mcp.streamable_http_app(
        host="0.0.0.0",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
        ),
    )
    # Not mcp.run(): it gives uvicorn no shutdown timeout, and a client's subscriptions/listen stream never ends,
    # so SIGTERM closed the port but never exited and the restart policy never fired. 5s is under Docker's 10s kill.
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=mcp.settings.log_level.lower(), timeout_graceful_shutdown=5)
