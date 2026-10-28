FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src src
COPY sources.txt questions.jsonl ./
COPY data/pages.jsonl data/pages.jsonl
RUN uv sync --frozen --no-dev

ENV HF_HOME=/cache/huggingface
VOLUME ["/cache", "/app/data/chroma"]

EXPOSE 8000
ENTRYPOINT ["uv", "run", "--no-sync", "ptqa"]
CMD ["serve", "--host", "0.0.0.0"]
