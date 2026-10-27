import os
from typing import Literal

import httpx
from opentelemetry import trace

MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Texts per request. Attention memory grows with batch times sequence length squared, and 64 chunks of 1500
# characters pushed the CPU embedder past Docker's memory limit and restarted it mid-ingest.
BATCH = 16
# API only, so callers that install no provider (query.py, smoke.py) pay nothing.
tracer = trace.get_tracer("embed_client")


def _url(path: str) -> str:
    return f"{os.environ.get('EMBED_URL', 'http://localhost:8080')}{path}"


def dim() -> int:
    r = httpx.get(_url("/health"))
    r.raise_for_status()
    return r.json()["dim"]


def embed(texts: list[str], input_type: Literal["document", "query"] = "document") -> list[list[float]]:
    vectors = []
    starts = range(0, len(texts), BATCH)
    for n, i in enumerate(starts, 1):
        batch = texts[i:i + BATCH]
        with tracer.start_as_current_span(f"embed batch {n}/{len(starts)}", attributes={
            "openinference.span.kind": "EMBEDDING",
            "embedding.model_name": MODEL,
            "embed.batch": n,
            "embed.batches": len(starts),
            "embed.chunks": len(batch),
        }) as span:
            r = httpx.post(_url("/v1/embeddings"), json={"input": batch, "model": MODEL, "input_type": input_type}, timeout=300)
            r.raise_for_status()
            body = r.json()
            # usage is optional in the OpenAI shape, and a missing count must not fail the embedding.
            if usage := body.get("usage"):
                span.set_attributes({"llm.token_count.prompt": usage["prompt_tokens"], "llm.token_count.total": usage["prompt_tokens"]})
        vectors.extend(e["embedding"] for e in sorted(body["data"], key=lambda e: e["index"]))
    return vectors
