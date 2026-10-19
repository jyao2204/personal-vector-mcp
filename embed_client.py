import os
from typing import Literal

import httpx

MODEL = "Qwen/Qwen3-Embedding-0.6B"
BATCH = 64


def _url(path: str) -> str:
    return f"{os.environ.get('EMBED_URL', 'http://localhost:8080')}{path}"


def dim() -> int:
    r = httpx.get(_url("/health"))
    r.raise_for_status()
    return r.json()["dim"]


def embed(texts: list[str], input_type: Literal["document", "query"] = "document") -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), BATCH):
        r = httpx.post(_url("/v1/embeddings"), json={"input": texts[i:i + BATCH], "model": MODEL, "input_type": input_type}, timeout=300)
        r.raise_for_status()
        vectors.extend(e["embedding"] for e in sorted(r.json()["data"], key=lambda e: e["index"]))
    return vectors
