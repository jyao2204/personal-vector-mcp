import os
import subprocess
import sys
import tempfile
from pathlib import Path

from qdrant_client import QdrantClient

# Runs ingest.py on throwaway notes in a throwaway collection; your data/ and notes collection are untouched.
COLLECTION = "ingest_check"
INGEST = Path(__file__).resolve().parent / "ingest.py"
qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


def ingest(cwd, *args):
    subprocess.run([sys.executable, INGEST, *args], cwd=cwd, env={**os.environ, "COLLECTION": COLLECTION}, check=True)


def stored():
    points, _ = qdrant.scroll(COLLECTION, limit=100)
    return sorted((p.payload["source"], p.payload["text"]) for p in points)


def files(data):
    return sorted(str(p.relative_to(data)) for p in data.rglob("*") if p.is_file())


qdrant.delete_collection(COLLECTION)
with tempfile.TemporaryDirectory() as tmp:
    data = Path(tmp, "data")
    (data / "sub").mkdir(parents=True)
    (data / "a.md").write_text("a" * 300 + "\n\n" + "b" * 300)
    (data / "sub" / "b.txt").write_text("bee")
    try:
        # A first run with no collection creates it, splits at the paragraph break, and moves each note under ingested/.
        ingest(tmp)
        assert files(data) == ["ingested/a.md", "ingested/sub/b.txt"], files(data)
        assert stored() == sorted([
            ("data/ingested/a.md", "a" * 300),
            ("data/ingested/a.md", "b" * 300),
            ("data/ingested/sub/b.txt", "bee"),
        ]), stored()

        # Notes already under ingested/ are skipped, and a note dropped in again replaces all its old chunks.
        (data / "ingested" / "sub" / "b.txt").write_text("edited in place")
        (data / "a.md").write_text("short")
        ingest(tmp)
        assert files(data) == ["ingested/a.md", "ingested/sub/b.txt"], files(data)
        assert stored() == [("data/ingested/a.md", "short"), ("data/ingested/sub/b.txt", "bee")], stored()

        # --rebuild re-reads ingested/ too.
        ingest(tmp, "--rebuild")
        assert stored() == [("data/ingested/a.md", "short"), ("data/ingested/sub/b.txt", "edited in place")], stored()
    finally:
        qdrant.delete_collection(COLLECTION)
print("ingest ok")
