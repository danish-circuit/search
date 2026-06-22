# One image, three targets -- the standard Dagster topology:
#   evaluator-code      : gRPC code location serving the job definitions
#   evaluator-webserver : the Dagster UI
#   evaluator-daemon    : schedules/sensors/run-queue
# All three run the same code; only the entrypoint differs.
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    DAGSTER_HOME=/app

WORKDIR /app

# Install dependencies first (layer-cached).
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev

# App code + Dagster instance/workspace config.
COPY evaluator ./evaluator
COPY config ./config
COPY dagster.yaml workspace.yaml ./
COPY evaluator-code-start.sh evaluator-webserver-start.sh evaluator-daemon-start.sh ./
RUN uv sync --no-dev && chmod +x ./*.sh

# Put the venv on PATH so the start scripts can call dagster directly.
ENV PATH="/app/.venv/bin:${PATH}"

FROM base AS evaluator-code
EXPOSE 4000
CMD ["./evaluator-code-start.sh"]

FROM base AS evaluator-webserver
EXPOSE 3000
CMD ["./evaluator-webserver-start.sh"]

FROM base AS evaluator-daemon
CMD ["./evaluator-daemon-start.sh"]