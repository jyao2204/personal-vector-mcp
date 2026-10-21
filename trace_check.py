# /// script
# dependencies = ["mcp>=2.3", "httpx", "opentelemetry-sdk"]
# ///
import asyncio
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
from opentelemetry.sdk.trace import TracerProvider

from mcp import Client

# Checks that searches and ingests reach Phoenix in the expected shape, then deletes every trace it made.
# Needs the compose stack up, Phoenix included, and a notes collection with at least one chunk.
PHOENIX_URL = os.environ.get("PHOENIX_URL", "http://localhost:6006")
MCP_URL = os.environ.get("MCP_URL", "http://localhost:8000/mcp")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
ROOT = Path(__file__).resolve().parent
NONCE = uuid.uuid4().hex[:8]
STARTED = datetime.now(UTC).isoformat()
# No exporter, and deliberately not the global provider: the MCP client opens its own span per request through the
# global tracer and sends that span's id, so with a global SDK provider the server's spans would hang off an
# unexported client span and an "unparented" search would still carry a trace. With only the API's no-op global
# provider the client passes the current span's context through unchanged, and sends nothing when there is none,
# like Claude Code without telemetry.
tracer = TracerProvider().get_tracer("trace_check")


def spans(**params):
    r = httpx.get(f"{PHOENIX_URL}/v1/projects/default/spans", params={"limit": 1000, **params})
    r.raise_for_status()
    return r.json()["data"]


def wait_for(done, timeout=30, **params):
    """Polls Phoenix until done(spans) holds, since senders export in batches."""
    deadline = time.monotonic() + timeout
    while not done(found := spans(**params)):
        assert time.monotonic() < deadline, (params, sorted(s["name"] for s in found))
        time.sleep(0.5)
    return found


def delete_trace(trace_id):
    httpx.delete(f"{PHOENIX_URL}/v1/traces/{trace_id}").raise_for_status()


def kids(found, parent):
    return {s["name"]: s for s in found if s["parent_id"] == parent["context"]["span_id"]}


async def check_search():
    with tracer.start_as_current_span("trace_check") as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        root_id = format(root.get_span_context().span_id, "016x")
        async with Client(MCP_URL, mode="legacy") as c:
            hits = (await c.call_tool("search_notes", {"query": "notes", "k": 20})).structured_content["result"]
            await c.call_tool("search_notes", {"query": "notes by tag", "tags": ["  No Such   TAG "]})
            assert (await c.call_tool("search_notes", {"query": ""})).is_error
            await c.call_tool("list_tags", {})
    found = wait_for(lambda ss: sum(s["name"].startswith("tools/call") for s in ss) == 4, trace_id=trace_id)
    try:
        # Tool calls and their steps only: no initialize, tools/list or HTTP-level spans.
        assert {s["name"] for s in found} == {
            "tools/call search_notes", "tools/call list_tags", "retrieve", "embed query", "qdrant query_points", "qdrant facet",
        }, sorted(s["name"] for s in found)
        # The calls joined this script's trace through the MCP request's _meta.
        assert all(s["parent_id"] == root_id for s in found if s["name"].startswith("tools/call")), found
        retrieves = {s["attributes"].get("input.value", ""): s for s in found if s["name"] == "retrieve"}
        ok = retrieves["notes"]
        a = ok["attributes"]
        assert ok["span_kind"] == "RETRIEVER" and a["search_notes.k"] == 20, ok
        for i, h in enumerate(hits):
            assert a[f"retrieval.documents.{i}.document.id"] == h["source"], (i, a)
            assert abs(a[f"retrieval.documents.{i}.document.score"] - h["score"]) < 1e-6, (i, a)
            assert a[f"retrieval.documents.{i}.document.content"] == h["text"], i
        assert f"retrieval.documents.{len(hits)}.document.id" not in a
        steps = kids(found, ok)
        assert steps.keys() == {"embed query", "qdrant query_points"}, steps.keys()
        assert steps["embed query"]["span_kind"] == "EMBEDDING", steps["embed query"]
        assert retrieves["notes by tag"]["attributes"]["search_notes.tags"] == ["no such tag"], retrieves["notes by tag"]
        assert retrieves[""]["status_code"] == "ERROR", retrieves[""]
        [tags_call] = [s for s in found if s["name"] == "tools/call list_tags"]
        assert kids(found, tags_call).keys() == {"qdrant facet"}
    finally:
        delete_trace(trace_id)


async def check_unparented_search():
    # Outside any span, like a Claude Code session started without make claude: the search is a trace of its own.
    query = f"trace check {NONCE}"
    async with Client(MCP_URL, mode="legacy") as c:
        await c.call_tool("search_notes", {"query": query, "k": 1})
    found = wait_for(lambda ss: any(s["attributes"].get("input.value") == query for s in ss), name="retrieve", start_time=STARTED)
    [r] = [s for s in found if s["attributes"].get("input.value") == query]
    trace_id = r["context"]["trace_id"]
    try:
        [top] = [s for s in spans(trace_id=trace_id) if s["parent_id"] is None]
        assert top["name"] == "tools/call search_notes", top
        # Both sessions above opened with initialize, tools/list and a notification; none may reach Phoenix.
        assert not spans(start_time=STARTED, name=["initialize", "tools/list", "notifications/initialized"])
    finally:
        delete_trace(trace_id)


asyncio.run(check_search())
asyncio.run(check_unparented_search())
print("trace ok")
