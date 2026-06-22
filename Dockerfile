# Single image for the FastAPI app. Uses uv for fast, reproducible installs.
FROM python:3.12-slim

# uv: copy the static binary from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (layer-cached) using only the lockfiles.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev

# Now copy the app source.
COPY app ./app
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]