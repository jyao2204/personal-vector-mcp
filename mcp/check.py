# /// script
# dependencies = ["mcp>=2.3", "httpx", "opentelemetry-sdk"]
# ///
import asyncio
import os

import httpx
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from mcp import Client

# Every call runs inside an unsampled trace, so the server records none of them and Phoenix stays untouched.
trace.set_tracer_provider(TracerProvider(sampler=ALWAYS_OFF))

URL = os.environ.get("MCP_URL", "http://localhost:8000/mcp")
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "check", "version": "0"}},
}


def hits(result):
    assert not result.is_error, result.content
    return result.structured_content["result"]


async def main():
    # mode="legacy" forces the initialize handshake, which is what Claude Code speaks.
    with trace.get_tracer("mcp-check").start_as_current_span("mcp check"):
        async with Client(URL, mode="legacy") as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert tools.keys() == {"search_notes", "list_tags"}, tools.keys()
            search = tools["search_notes"]
            assert "source" in search.description and "the user" in search.description, search.description
            props = search.input_schema["properties"]
            assert "description" in props["query"] and "20" in props["k"]["description"], props
            assert "list_tags" in props["tags"]["description"], props["tags"]

            # Works on anyone's notes: nothing below assumes what the collection holds or how big it is.
            top = hits(await c.call_tool("search_notes", {"query": "notes", "k": 20}))
            assert top, "collection is empty, run uv run ingest.py first"
            assert {"text", "source", "score", "tags", "topic", "document_date", "date_ingested"} <= top[0].keys(), top[0]
            scores = [h["score"] for h in top]
            assert scores == sorted(scores, reverse=True), scores

            # A tag filter only returns chunks carrying that tag, whatever its case, and an unknown tag returns nothing.
            tags = hits(await c.call_tool("list_tags", {}))
            assert all({"tag", "count"} <= t.keys() for t in tags), tags
            for t in tags[:1]:
                scoped = hits(await c.call_tool("search_notes", {"query": "notes", "k": 20, "tags": [t["tag"].upper()]}))
                assert scoped and all(t["tag"] in h["tags"] for h in scoped), (t, scoped)
            assert hits(await c.call_tool("search_notes", {"query": "notes", "tags": ["no such tag"]})) == []

            # Searching for a stored chunk's own text must bring back its file.
            probe = top[len(top) // 2]
            found = hits(await c.call_tool("search_notes", {"query": probe["text"], "k": 3}))
            assert probe["source"] in [h["source"] for h in found], (probe["source"], found)

            n = len(top)
            assert len(hits(await c.call_tool("search_notes", {"query": "x", "k": 100}))) == n
            assert len(hits(await c.call_tool("search_notes", {"query": "x", "k": 0}))) == 1
            assert len(hits(await c.call_tool("search_notes", {"query": "x", "k": -5}))) == 1
            assert len(hits(await c.call_tool("search_notes", {"query": "x"}))) == min(5, n)

            e = await c.call_tool("search_notes", {"query": ""})
            assert e.is_error and "embedder" in e.content[0].text, e.content


asyncio.run(main())

# A page on another origin that rebinds its DNS to 127.0.0.1 must not reach the tool.
forged = httpx.post(URL, json=INIT, headers={"Host": "evil.example:8000", "Accept": "application/json, text/event-stream"})
assert 400 <= forged.status_code < 500, forged.status_code
print("mcp ok")
