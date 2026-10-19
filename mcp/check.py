# /// script
# dependencies = ["mcp>=2.3", "httpx"]
# ///
import asyncio
import os

import httpx

from mcp import Client

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
    async with Client(URL, mode="legacy") as c:
        tools = (await c.list_tools()).tools
        assert [t.name for t in tools] == ["search_notes"], tools
        assert "source" in tools[0].description and "the user" in tools[0].description, tools[0].description
        props = tools[0].input_schema["properties"]
        assert "description" in props["query"] and "20" in props["k"]["description"], props

        # Works on anyone's notes: nothing below assumes what the collection holds or how big it is.
        top = hits(await c.call_tool("search_notes", {"query": "notes", "k": 20}))
        assert top, "collection is empty, run uv run ingest.py first"
        assert {"text", "source", "score"} <= top[0].keys(), top[0]
        scores = [h["score"] for h in top]
        assert scores == sorted(scores, reverse=True), scores

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
