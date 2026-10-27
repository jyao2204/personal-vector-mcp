# /// script
# dependencies = ["mcp>=2.3", "httpx", "httpx2", "opentelemetry-sdk"]
# ///
import asyncio
import contextlib
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import httpx2
from mcp.client.streamable_http import streamable_http_client
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
COLLECTION = "trace_check"
TMP = tempfile.TemporaryDirectory()  # removed at exit, with every note written under it
# Traces whose ids the checks know before making them; every other trace the run makes carries NONCE in a span name or query.
made_traces = set()


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


def delete_made_traces():
    """Deletes every trace this run made, found or not by the check that made it, so a failed run leaves none behind."""
    ours = (s for s in spans(start_time=STARTED) if NONCE in s["name"] or NONCE in str(s["attributes"].get("input.value", "")))
    for trace_id in made_traces | {s["context"]["trace_id"] for s in ours}:
        r = httpx.delete(f"{PHOENIX_URL}/v1/traces/{trace_id}")
        if r.status_code != 404:
            r.raise_for_status()


def kids(found, parent):
    return {s["name"]: s for s in found if s["parent_id"] == parent["context"]["span_id"]}


async def check_search():
    with tracer.start_as_current_span("trace_check") as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        made_traces.add(trace_id)
        root_id = format(root.get_span_context().span_id, "016x")
        async with Client(MCP_URL, mode="legacy") as c:
            hits = (await c.call_tool("search_notes", {"query": "notes", "k": 20})).structured_content["result"]
            await c.call_tool("search_notes", {"query": "notes by tag", "tags": ["  No Such   TAG "]})
            assert (await c.call_tool("search_notes", {"query": ""})).is_error
            await c.call_tool("list_tags", {})
    found = wait_for(lambda ss: sum(s["name"].startswith("tools/call") for s in ss) == 4, trace_id=trace_id)
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


async def check_unparented_search():
    # Outside any span, like a Claude Code session started without make claude: the search is a trace of its own.
    query = f"trace check {NONCE}"
    async with Client(MCP_URL, mode="legacy") as c:
        await c.call_tool("search_notes", {"query": query, "k": 1})
    # Found by its input, since the trace list shows the root span's input, so the query must be on it.
    found = wait_for(lambda ss: any(s["attributes"].get("input.value") == query for s in ss),
                     name="tools/call search_notes", start_time=STARTED)
    [top] = [s for s in found if s["attributes"].get("input.value") == query]
    assert top["parent_id"] is None, top
    # Both sessions above opened with initialize, tools/list and a notification; none may reach Phoenix.
    assert not spans(start_time=STARTED, name=["initialize", "tools/list", "notifications/initialized"])


async def check_caller_header():
    # The Python client only sends trace context in _meta; Claude Code sends it as a traceparent header, which the
    # server's JoinCallerTrace must adopt. The ids are invented here, so only that header can put the span in this trace.
    trace_id, span_id = uuid.uuid4().hex, uuid.uuid4().hex[:16]
    made_traces.add(trace_id)
    async with (
        httpx2.AsyncClient(headers={"traceparent": f"00-{trace_id}-{span_id}-01"}) as http,
        Client(streamable_http_client(MCP_URL, http_client=http), mode="legacy") as c,
    ):
        await c.call_tool("search_notes", {"query": "notes", "k": 1})
    [call] = [s for s in wait_for(lambda ss: any(s["name"] == "tools/call search_notes" for s in ss), trace_id=trace_id)
              if s["name"] == "tools/call search_notes"]
    assert call["parent_id"] == span_id, call


def note(text, prefix):
    """Writes a throwaway note under a fresh data/ folder and returns (folder, file name)."""
    data = Path(tempfile.mkdtemp(dir=TMP.name), "data")
    data.mkdir()
    name = f"{prefix}-{NONCE}.md"
    (data / name).write_text(text)
    return data, name


def ingest(data, **env):
    # Runs ingest.py the way make ingest does, from the note's folder, into a throwaway collection.
    return subprocess.Popen(
        ["uv", "run", "--project", str(ROOT), "python", str(ROOT / "ingest.py")],
        cwd=data.parent, env={**os.environ, "COLLECTION": COLLECTION, **env},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def file_trace(name):
    [f] = wait_for(lambda ss: len(ss) == 1, name=f"ingest {name}")
    return f


def check_ingest():
    # Thirty-two 1,400-character chunks make two full batches; the first must show up while the second is still embedding.
    data, name = note("\n\n".join(f"paragraph {i} " + "deep modules hide complexity " * 50 for i in range(32)), "trace-check")
    run = ingest(data)
    try:
        # A 16-chunk batch takes about ten seconds on CPU.
        wait_for(lambda ss: len(ss) == 1, timeout=180, name="embed batch 1/2", start_time=STARTED)
        assert run.poll() is None, "batch 1 only reached Phoenix after ingest finished"
    except BaseException:
        run.kill()
        raise
    _, err = run.communicate()
    assert run.returncode == 0, err
    f = file_trace(name)
    found = wait_for(lambda ss: len(ss) == 4, trace_id=f["context"]["trace_id"])
    assert f["parent_id"] is None and f["span_kind"] == "CHAIN" and f["attributes"]["ingest.chunks"] == 32, f
    steps = kids(found, f)
    assert steps.keys() == {"embed batch 1/2", "embed batch 2/2", "store in qdrant"}, steps.keys()
    for key, position in (("embed batch 1/2", (1, 2, 16)), ("embed batch 2/2", (2, 2, 16))):
        b = steps[key]
        assert b["span_kind"] == "EMBEDDING", b
        assert (b["attributes"]["embed.batch"], b["attributes"]["embed.batches"], b["attributes"]["embed.chunks"]) == position, b
        assert b["attributes"]["llm.token_count.prompt"] > 0, b


def check_failed_ingest():
    data, name = note("---\ntags: [unclosed\n---\nbody\n", "trace-check-bad")
    run = ingest(data)
    _, err = run.communicate()
    assert run.returncode != 0 and "malformed frontmatter" in err, "malformed frontmatter was accepted"
    f = file_trace(name)
    assert f["status_code"] == "ERROR", f


def timed_ingest(name_prefix, **env):
    data, name = note("A short note.", name_prefix)
    start = time.monotonic()
    run = ingest(data, **env)
    _, err = run.communicate()
    assert run.returncode == 0, err
    return name, time.monotonic() - start, err


def check_ingest_without_phoenix():
    name, with_phoenix, _ = timed_ingest("trace-check-up")
    file_trace(name)  # its export must land before the final cleanup
    # Port 1 refuses connections, standing in for a stopped Phoenix.
    _, without, err = timed_ingest("trace-check-down", OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:1")
    assert "export" not in err.lower(), err
    assert without < with_phoenix + 3, (without, with_phoenix)


try:
    asyncio.run(check_search())
    asyncio.run(check_unparented_search())
    asyncio.run(check_caller_header())
    check_ingest()
    check_failed_ingest()
    check_ingest_without_phoenix()
finally:
    # First, so a Phoenix that is down cannot leave the collection behind. A Qdrant that is down has none to drop,
    # and must not skip the trace cleanup either.
    with contextlib.suppress(httpx.HTTPError):
        httpx.delete(f"{QDRANT_URL}/collections/{COLLECTION}")
    if sys.exc_info()[0]:
        time.sleep(2)  # a failed run may stop before the senders' batched exports (every second) have landed
    delete_made_traces()
print("trace ok")
