import logging
import os
from typing import Annotated

import httpx
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
)
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:8080")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION", "notes")
MODEL = "Qwen/Qwen3-Embedding-0.6B"

mcp = MCPServer("notes")
qdrant = QdrantClient(url=QDRANT_URL)
tracer = trace.get_tracer("notes-mcp")


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


class ToolCallsOnly(Sampler):
    """Keeps the SDK's housekeeping spans (initialize, tools/list, notifications) out of Phoenix's trace list.
    Everything else follows the caller's sampling decision, so a caller that is not sampled records nothing."""

    def __init__(self):
        self._parent_based = ParentBased(ALWAYS_ON)

    def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None, trace_state=None):
        method = (attributes or {}).get("mcp.method.name")
        if method is not None and method != "tools/call":
            return SamplingResult(Decision.DROP)
        return self._parent_based.should_sample(parent_context, trace_id, name, kind, attributes, links, trace_state)

    def get_description(self):
        return "ToolCallsOnly"


class JoinCallerTrace:
    """Claude Code sends its trace as an HTTP traceparent header, while the SDK only reads the request's _meta and
    falls back to the ambient context, which this sets. Records no spans itself."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = context.attach(extract({k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}))
        try:
            await self.app(scope, receive, send)
        finally:
            context.detach(token)


@mcp.tool()
def list_tags() -> list[Tag]:
    """Every tag on the user's notes with the number of chunks carrying it, most common first. Call it before passing tags to search_notes, so the filter only names tags that exist."""
    with tracer.start_as_current_span("qdrant facet"):
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
    wanted = normalize_tags(tags) if tags else None
    query_filter = Filter(must=[FieldCondition(key="tags", match=MatchAny(any=wanted))]) if tags else None
    attributes = {"openinference.span.kind": "RETRIEVER", "input.value": query, "search_notes.k": k}
    if wanted is not None:
        attributes["search_notes.tags"] = wanted
    # The SDK's tools/call span is the trace's root without make claude, and Phoenix's trace list shows its input.value.
    trace.get_current_span().set_attribute("input.value", query)
    with tracer.start_as_current_span("retrieve", attributes=attributes) as span:
        # ToolError text reaches the caller; any other exception is masked as a bare "Error executing tool".
        with tracer.start_as_current_span("embed query", attributes={"openinference.span.kind": "EMBEDDING", "embedding.model_name": MODEL}):
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
        with tracer.start_as_current_span("qdrant query_points"):
            try:
                points = qdrant.query_points(collection_name=COLLECTION, query=vec, query_filter=query_filter, limit=k).points
            except Exception as e:
                raise ToolError(f"qdrant at {QDRANT_URL}, collection {COLLECTION!r} failed: {e}") from e
        # .get() for the metadata: chunks stored before frontmatter support carry only text and source until a rebuild.
        hits = [
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
        for i, h in enumerate(hits):
            span.set_attributes({
                f"retrieval.documents.{i}.document.id": h.source,
                f"retrieval.documents.{i}.document.score": h.score,
                f"retrieval.documents.{i}.document.content": h.text,
            })
        return hits


if __name__ == "__main__":
    # The same provider setup as ingest.py's, written again because this image builds from mcp/ alone.
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006")
    provider = TracerProvider(resource=Resource.create({"service.name": "notes-mcp"}), sampler=ToolCallsOnly())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(timeout=2), schedule_delay_millis=1000))
    trace.set_tracer_provider(provider)
    # Phoenix being down must cost nothing, log noise included.
    logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)
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
    uvicorn.run(JoinCallerTrace(app), host="0.0.0.0", port=8000, log_level=mcp.settings.log_level.lower(), timeout_graceful_shutdown=5)
