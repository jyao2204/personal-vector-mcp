.PHONY: ingest ingest-rebuild

ingest:
	uv run ingest.py

ingest-rebuild:
	uv run ingest.py --rebuild
