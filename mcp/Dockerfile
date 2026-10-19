FROM ghcr.io/astral-sh/uv:0.11.24-python3.13-trixie-slim
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
COPY server.py .
EXPOSE 8000
CMD ["python", "server.py"]
