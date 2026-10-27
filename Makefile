.PHONY: ingest ingest-rebuild claude

ingest:
	uv run ingest.py

ingest-rebuild:
	uv run ingest.py --rebuild

# Claude Code reads OTEL_* only from its own environment (project settings files are ignored), so its tracing is on
# just for sessions started here. Pass claude's own options through ARGS, e.g. make claude ARGS=--continue.
claude:
	CLAUDE_CODE_ENABLE_TELEMETRY=1 CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1 \
	OTEL_TRACES_EXPORTER=otlp OTEL_METRICS_EXPORTER=none OTEL_LOGS_EXPORTER=none \
	OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf \
	OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=$${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:6006}/v1/traces \
	OTEL_LOG_USER_PROMPTS=1 OTEL_LOG_TOOL_DETAILS=1 \
	claude $(ARGS)
