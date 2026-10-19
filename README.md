# private-mcp

Semantic search over your own markdown notes, run entirely on your machine and exposed to Claude Code as an MCP tool.
A local Qwen3-Embedding-0.6B service turns notes into vectors, Qdrant stores them, and an MCP server lets Claude search them and answer with sources.
No API key is needed, and no embedding or chat model is paid for: Claude Code is the step that writes the answer.

## What's here

- `embedder/` is a FastAPI service that serves Qwen3-Embedding-0.6B (1024 dimensions) over an OpenAI-style `/v1/embeddings` endpoint.
- `mcp/` is an MCP server over streamable HTTP with one tool, `search_notes`.
- `docker-compose.yml` runs Qdrant, the embedder, and the MCP server together.
- `phase1_ingest.py` and `phase1_query.py` chunk, embed, store, and search notes with plain HTTP and `qdrant-client`.
- `phase2_ingest.py` does the same ingest through LangChain.
- `embed_client.py` is the shared embedding client the host scripts use.
- `smoke.py` and `mcp/check.py` are the runnable checks.

## Requirements

- Docker with at least 4 GB of memory available to containers.
  The embedder alone uses about 3 GB once the model is loaded.
- [uv](https://docs.astral.sh/uv/) to run the host scripts.
- [Claude Code](https://claude.com/claude-code) to ask questions through the MCP tool.

Everything runs on CPU.
The embedder image is about 4 GB because the model is baked in at build time.

## Add your notes

Put your notes in a `data/` folder at the repo root.
The ingest scripts pick up every `*.md` and `*.txt` file in it, including subfolders.
Each search result names the file it came from, so descriptive filenames make better citations.

## Run

Start Qdrant, the embedder, and the MCP server with `docker compose up --build -d`.
The first build downloads the 1.2 GB embedding model once and bakes it into the image.
Check the embedder with `uv run smoke.py`, which prints `smoke ok` and `client ok`.
Build the collections with `uv run phase1_ingest.py` and `uv run phase2_ingest.py`.
Each ingest rebuilds its collection from scratch and takes minutes on CPU, about ten for a thousand chunks.
Query the raw collection with `uv run phase1_query.py "your question"`, which prints the five closest chunks with their scores and files.
Check the MCP server with `uv run mcp/check.py`, which prints `mcp ok`.
It searches your notes without assuming what they contain, so run it after `phase1_ingest.py`.
The Qdrant dashboard is at http://localhost:6333/dashboard.

After a reboot, start the stack again with `docker compose up -d`.

## Ask Claude Code

The `notes` MCP server is registered in `.mcp.json`.
The first time you start `claude` in this directory, it asks whether to use that server; approve it.
If you declined by mistake, run `claude mcp reset-project-choices` and start `claude` again.
`claude mcp list` shows `notes` as connected once the stack is up and the server is approved.
Then ask Claude Code anything your notes cover, and it calls `search_notes` and cites the files it found.

`search_notes` takes a `query` and an optional `k`, the number of chunks to return.
`k` defaults to 5 and is clamped to 1 to 20.
It searches the `practice_docs` collection that `phase1_ingest.py` builds.

## Configuration

Every service URL comes from the environment, and only the defaults mention localhost.

| Variable | Default | Read by |
|---|---|---|
| `EMBED_URL` | `http://localhost:8080` | host scripts, MCP server |
| `QDRANT_URL` | `http://localhost:6333` | host scripts, MCP server |
| `COLLECTION` | `practice_docs` | MCP server |
| `MCP_URL` | `http://localhost:8000/mcp` | `mcp/check.py` |

The ingest and query scripts also load these from a `.env` file at the repo root.
Inside compose, the MCP server gets `http://embedder:8080` and `http://qdrant:6333` from `docker-compose.yml`.

## Embedding API

`POST /v1/embeddings` takes `{"input": "text" or ["text", ...], "input_type": "document" or "query"}` and returns OpenAI-shaped embeddings.
`input_type` defaults to `"document"`; `"query"` applies the model's query prompt, which improves retrieval for questions.
Empty input returns 422, and inputs longer than 2048 tokens are truncated.
Clients should send at most 64 texts per request.
`GET /health` reports the model and its dimension once it is loaded.

For example: `curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' -d '{"input": "hello", "input_type": "query"}'`.

## Offline and security

No script needs an API key or a network connection once the images are built and the Python dependencies are installed.
Only Claude Code itself talks to Anthropic.

Qdrant, the embedder, and the MCP server have no authentication.
Their ports (6333, 6334, 8080, 8000) are bound to `127.0.0.1`, so only this machine can reach them.
The MCP server also rejects requests whose `Host` header is not localhost, which blocks DNS-rebinding attacks from web pages.
Add authentication before exposing any of them on a network.
