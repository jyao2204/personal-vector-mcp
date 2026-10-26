# personal-vector-mcp

Semantic search over your own markdown notes, run entirely on your machine and exposed to Claude Code as an MCP tool.
A local Qwen3-Embedding-0.6B service turns notes into vectors, Qdrant stores them, and an MCP server lets Claude search them and answer with sources.
No API key is needed, and no embedding or chat model is paid for: Claude Code is the step that writes the answer.

## What's here

- `embedder/` is a FastAPI service that serves Qwen3-Embedding-0.6B (1024 dimensions) over an OpenAI-style `/v1/embeddings` endpoint.
- `mcp/` is an MCP server over streamable HTTP with two tools, `search_notes` and `list_tags`.
- `docker-compose.yml` runs Qdrant, the embedder, the MCP server, and Phoenix together.
- `ingest.py` splits notes into chunks, embeds them, and stores them in Qdrant.
- `query.py` searches the stored chunks from the command line.
- `embed_client.py` is the shared embedding client the host scripts use.
- Phoenix, run by compose, shows traces of searches and ingests in a web UI.
- `smoke.py`, `ingest_check.py`, `mcp/check.py`, and `trace_check.py` are the runnable checks.

## Requirements

- Docker with at least 6 GB of memory available to containers.
  The embedder alone uses about 3 GB once the model is loaded and more while it embeds a batch.
  Phoenix adds about 0.5 GB.
- [uv](https://docs.astral.sh/uv/) to run the host scripts.
- [Claude Code](https://claude.com/claude-code) to ask questions through the MCP tool.

Everything runs on CPU.
The embedder image is about 4 GB because the model is baked in at build time.

## Add your notes

Put your notes in a `data/` folder at the repo root.
The ingest script picks up every `*.md` and `*.txt` file in it, including subfolders, and splits each into chunks of at most 1500 characters at paragraph, line, or word breaks.
A frontmatter block at the top of a note is read for tags and a topic, as described under "Tag your notes", and is not chunked.
Each search result names the file it came from, so descriptive filenames make better citations.

`uv run ingest.py` embeds each note under `data/` that is not in `data/ingested/`, then moves it into `data/ingested/`, keeping its subfolder.
So each run embeds only the notes added since the last one.
To update a note, drop the new version into `data/` again, and the next run replaces its old chunks and its old file.
Edits made in place under `data/ingested/`, and files deleted from it, are only picked up by `uv run ingest.py --rebuild`.
`--rebuild` drops the collection and re-ingests every note, `data/ingested/` included.
`make ingest` and `make ingest-rebuild` run the same two commands.
If a run fails partway, the notes it finished are already in `data/ingested/` and the next run picks up the rest.

## Tag your notes

A note can start with a YAML frontmatter block, and the ingest script reads three optional fields from it:

```yaml
---
tags: [cooking, food, korean cuisine]
topic: Author James Jones shares fusion Asian American recipes he formed after his trip through Southeast Asia
document_date: 2019-05-01
---
```

`tags` is a list of keywords that search results can be narrowed to.
They are normalised on the way in: lowercased, trimmed, inner whitespace collapsed to one space, and duplicates dropped, so `Korean  Cuisine` and `korean cuisine` are the same tag.
`topic` is a one-sentence description of the whole note.
It is embedded together with every chunk of the note, so a chunk is also found by what its document is about, while the stored text stays the chunk alone.
`document_date` is when the note or its source was written, in any form YAML accepts.
Other fields are ignored, and a note without frontmatter ingests as before.
A frontmatter block that is not valid YAML stops the run with the file named, and the note stays in `data/` for a fix.

Each stored chunk carries `tags`, `topic`, `document_date`, and `date_ingested`, the UTC time the ingest run stored it.
Chunks stored before this existed carry none of these until `uv run ingest.py --rebuild`; searches still return them, with empty tags and null fields.

## Run

Start Qdrant, the embedder, the MCP server, and Phoenix with `docker compose up --build -d`.
The first build downloads the 1.2 GB embedding model once and bakes it into the image.
Check the embedder with `uv run smoke.py`, which prints `smoke ok` and `client ok`.
Ingest your notes with `uv run ingest.py`.
Embedding takes minutes on CPU, about ten for a thousand chunks, so the first run is the slow one.
Check the ingest itself with `uv run ingest_check.py`, which prints `ingest ok` and uses its own collection and temporary notes.
Query the collection with `uv run query.py "your question"`, which prints the five closest chunks with their scores and files.
Check the MCP server with `uv run mcp/check.py`, which prints `mcp ok`.
It searches your notes without assuming what they contain, so run it after `ingest.py`.
Check the tracing with `uv run trace_check.py`, which prints `trace ok`.
It needs Phoenix up, takes about a minute, and deletes the traces it makes.
The Qdrant dashboard is at http://localhost:6333/dashboard.

Each service restarts by itself when it crashes or Docker restarts, unless you stopped it with `docker compose stop`.
After a reboot the stack comes back once Docker Desktop is running, so turn on its "Start Docker Desktop when you sign in" setting to make that automatic.
While the embedder loads its model, for about a minute after Docker starts, `search_notes` reports that the embedder failed.

## Ask Claude Code

The `notes` MCP server is registered in `.mcp.json`.
The first time you start `claude` in this directory, it asks whether to use that server; approve it.
If you declined by mistake, run `claude mcp reset-project-choices` and start `claude` again.
`claude mcp list` shows `notes` as connected once the stack is up and the server is approved.
Then ask Claude Code anything your notes cover, and it calls `search_notes` and cites the files it found.

`search_notes` takes a `query`, an optional `k`, the number of chunks to return, and an optional `tags` list.
`k` defaults to 5 and is clamped to 1 to 20.
With `tags`, only chunks carrying at least one of those tags are searched, with the same normalisation as ingest, so case and spacing do not matter.
Each result carries the chunk's text, source, score, tags, topic, document date, and ingest time.
`list_tags` returns every tag in the collection with the number of chunks carrying it, so Claude only filters on tags that exist.
Its description tells Claude to search unfiltered first and narrow by tag only when you name a domain or the first results are off-topic.
Both tools search the `notes` collection that `ingest.py` builds.

## Watch it in Phoenix

Phoenix runs with the rest of the stack and shows what the tools did.
Open http://localhost:6006 and look in the `default` project.

A search records the query, `k`, the tags, and the ranked chunks with their score, source, and text.
A `list_tags` call is recorded too.
An ingest records one trace per file, a step for each embedding batch with its position and token count, and the step that stores the chunks in Qdrant.

`make claude` starts Claude Code with its own tracing turned on and sent to Phoenix.
Your question then appears at the top of the trace, with each search beneath it.
A session started any other way still records its searches, each as its own trace, without the question.
`make claude ARGS=--continue` passes options through to `claude`.
Claude Code's tracing is a beta feature, so its step names may change.

To watch an ingest live, open the Spans tab and replace its default filter `parent_id is None` with `span_kind == 'EMBEDDING'`.
The default filter hides the batches until the whole file has finished.

Traces are kept in `phoenix_storage/` as SQLite, forever by default.
Set `PHOENIX_DEFAULT_RETENTION_POLICY_DAYS` on the `phoenix` service to cap that.
They hold copies of your note text, and the traces from `make claude` also hold the email address and account IDs that Claude Code attaches.

## Configuration

Every service URL comes from the environment, and only the defaults mention localhost.

| Variable | Default | Read by |
|---|---|---|
| `EMBED_URL` | `http://localhost:8080` | host scripts, MCP server |
| `QDRANT_URL` | `http://localhost:6333` | host scripts, MCP server |
| `COLLECTION` | `notes` | `ingest.py`, `query.py`, MCP server |
| `MCP_URL` | `http://localhost:8000/mcp` | `mcp/check.py` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:6006` | `ingest.py`, MCP server, `make claude` |
| `PHOENIX_URL` | `http://localhost:6006` | `trace_check.py` |

The ingest and query scripts also load these from a `.env` file at the repo root, but a value set in the shell wins over it.
A shell-wide `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, as some people export for work, would send note text and prompts from `ingest.py` and `make claude` to that collector instead of Phoenix, so unset it before running them.
Inside compose, the MCP server gets `http://embedder:8080`, `http://qdrant:6333`, and `http://phoenix:6006` from `docker-compose.yml`.

## Embedding API

`POST /v1/embeddings` takes `{"input": "text" or ["text", ...], "input_type": "document" or "query"}` and returns OpenAI-shaped embeddings.
`input_type` defaults to `"document"`; `"query"` applies the model's query prompt, which improves retrieval for questions.
Empty input returns 422, and inputs longer than 2048 tokens are truncated.
Clients should send at most 16 texts of up to 1500 characters per request; larger requests can exhaust the container's memory.
`GET /health` reports the model and its dimension once it is loaded.

For example: `curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' -d '{"input": "hello", "input_type": "query"}'`.

## Offline and security

No script needs an API key or a network connection once the images are built and the Python dependencies are installed.
Only Claude Code itself talks to Anthropic.

Qdrant, the embedder, the MCP server, and Phoenix have no authentication.
Their ports (6333, 6334, 8080, 8000, 6006) are bound to `127.0.0.1`, so only this machine can reach them.
The MCP server also rejects requests whose `Host` header is not localhost, which blocks DNS-rebinding attacks from web pages.
Phoenix does not check the `Host` header, so a web page using DNS rebinding could read your traces, as it already could read Qdrant.
Phoenix's API also accepts new spans and trace deletions without authentication, so such a page could add or delete traces as well.
Phoenix makes no outside requests on its own, and its agent assistant and bash tool are turned off in `docker-compose.yml`.
Add authentication before exposing any of them on a network.
