import os
from typing import Annotated

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

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


@mcp.tool()
def search_notes(
    query: Annotated[str, Field(description="What to look for, as a natural-language question or keywords.")],
    k: Annotated[int, Field(description="How many chunks to return, 1 to 20; values outside that range are clamped.")] = 5,
) -> list[Hit]:
    """Semantic search over the user's notes and decision records. Use it for any question about the user's projects, decisions, integrations, or past work. Returns the k most relevant chunks, each with its source filename and a cosine similarity score, highest first. Cite the source filenames in the answer."""
    k = max(1, min(k, 20))
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
        points = qdrant.query_points(collection_name=COLLECTION, query=vec, limit=k).points
    except Exception as e:
        raise ToolError(f"qdrant at {QDRANT_URL}, collection {COLLECTION!r} failed: {e}") from e
    return [Hit(text=p.payload["text"], source=p.payload["source"], score=p.score) for p in points]


if __name__ == "__main__":
    # Binding 0.0.0.0 inside the container turns off the SDK's automatic localhost Host check, so set it
    # explicitly: a page that rebinds its DNS to 127.0.0.1 must not reach the tool. A cloud deploy adds its
    # public hostname here alongside auth.
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
        ),
    )
