import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
# ponytail: 2048-token cap keeps one CPU request bounded; raise it (model allows 32k) on a CUDA base image.
MAX_SEQ_LENGTH = 2048


@asynccontextmanager
async def lifespan(app: FastAPI):
    # transformers 5 defaults to the checkpoint's bf16, which leaves normalised vectors off unit length by ~0.4%.
    model = SentenceTransformer(MODEL_ID, model_kwargs={"dtype": "float32"})
    model.max_seq_length = MAX_SEQ_LENGTH
    app.state.model = model
    yield


app = FastAPI(lifespan=lifespan)


NonEmpty = Annotated[str, Field(min_length=1)]


class EmbedRequest(BaseModel):
    input: NonEmpty | list[NonEmpty] = Field(min_length=1)
    model: str = MODEL_ID
    input_type: Literal["document", "query"] = "document"


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "dim": app.state.model.get_embedding_dimension()}


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest):
    model = app.state.model
    texts = [req.input] if isinstance(req.input, str) else req.input
    vectors = model.encode(
        texts,
        prompt_name="query" if req.input_type == "query" else None,
        normalize_embeddings=True,
    )
    tokens = sum(len(ids) for ids in model.tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"])
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vectors)],
        "model": req.model,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }
