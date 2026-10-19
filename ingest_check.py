import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from embed_client import embed

# Runs ingest.py on throwaway notes in a throwaway collection; your data/ and notes collection are untouched.
COLLECTION = "ingest_check"
INGEST = Path(__file__).resolve().parent / "ingest.py"
qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


def ingest(cwd, *args):
    subprocess.run([sys.executable, INGEST, *args], cwd=cwd, env={**os.environ, "COLLECTION": COLLECTION}, check=True)


def stored():
    points, _ = qdrant.scroll(COLLECTION, limit=100)
    return sorted((p.payload["source"], p.payload["text"]) for p in points)


def point(source):
    points, _ = qdrant.scroll(COLLECTION, limit=100, with_vectors=True)
    [p] = [p for p in points if p.payload["source"] == source]
    return p


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

        # Frontmatter is stripped from the stored text and lands in the payload, with tags normalised.
        (data / "c.md").write_text(
            "---\ntags: [Cooking, ' korean   cuisine ', cooking, 2019]\ntopic: Fusion recipes\ndocument_date: 2019-05-01\n---\nbody text\n"
        )
        before = datetime.now(UTC)
        ingest(tmp)
        c = point("data/ingested/c.md").payload
        assert c["text"] == "body text", c
        assert c["tags"] == ["cooking", "korean cuisine", "2019"], c
        assert c["topic"] == "Fusion recipes" and c["document_date"] == "2019-05-01", c
        assert datetime.fromisoformat(c["date_ingested"]) >= before, c
        plain = point("data/ingested/a.md").payload
        assert plain["tags"] == [] and plain["topic"] is None and plain["document_date"] is None, plain
        # The topic is embedded with each chunk, so the vector is that of topic plus chunk, not the chunk alone.
        [with_topic] = embed(["Fusion recipes\n\nbody text"])
        assert max(abs(x - y) for x, y in zip(point("data/ingested/c.md").vector, with_topic)) < 1e-4
        assert qdrant.get_collection(COLLECTION).payload_schema["tags"].data_type == "keyword"

        # Malformed frontmatter fails the run and leaves the note in data/ for a fix.
        (data / "bad.md").write_text("---\ntags: [unclosed\n---\nbody\n")
        try:
            ingest(tmp)
            raise AssertionError("ingest accepted malformed frontmatter")
        except subprocess.CalledProcessError:
            pass
        assert "bad.md" in files(data) and "ingested/bad.md" not in files(data), files(data)
    finally:
        qdrant.delete_collection(COLLECTION)
print("ingest ok")
