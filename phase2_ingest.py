import os
import glob

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from embed_client import embed

load_dotenv()

paths = glob.glob("data/**/*.md", recursive=True) + glob.glob("data/**/*.txt", recursive=True)
docs = []
for path in paths:
    with open(path) as f:
        text = f.read()
    docs.append(Document(page_content=text, metadata={"source": path}))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)


class QwenEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return embed([text], "query")[0]


# from_documents batches both the embed call and the upsert at 64 per call internally,
# so embed_client's own batching is a no-op here and only matters for phase1_ingest.py.
QdrantVectorStore.from_documents(
    chunks,
    QwenEmbeddings(),
    url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    collection_name="langchain_docs",
    force_recreate=True,
)
print(f"Ingested {len(chunks)} chunks from {len(paths)} files into 'langchain_docs'")
