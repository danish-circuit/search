# Search Agent

A small, readable codebase for learning how to build a **search agent** with
[Pydantic AI](https://ai.pydantic.dev/), **Postgres**, and **pgvector**.

It demonstrates four retrieval strategies side by side:

| Method | What it does | Powered by |
|---|---|---|
| **Lexical** | Keyword / full-text search | Postgres FTS (`tsvector`, `websearch_to_tsquery`, `ts_rank_cd`) |
| **Semantic** | Meaning-based search | pgvector cosine similarity (`<=>`) over OpenAI embeddings |
| **Hybrid** | Blends lexical + semantic | Reciprocal Rank Fusion (RRF) |
| **HyDE** | Hypothetical Document Embeddings | LLM writes a fake answer, we embed *that*, then search |

A Claude (Opus) agent is given all four as tools and decides which to use.

## Architecture

```
  Streamlit frontend  (http://localhost:8501)
        │  HTTP
        ▼
  FastAPI backend     (http://localhost:8000)
  POST /ingest  ── upload a PDF, index it
  POST /chat    ── streaming agent answer (SSE)
  GET  /search  ── run one search method directly (great for demos)
  GET  /config  ── feature flags the frontend reads
        │
        ▼
  Postgres 16 + pgvector
   • documents / chunks tables
   • HNSW index  -> semantic search
   • GIN tsvector index -> lexical search
```

Three Docker services (`db`, `api`, `frontend`) built from one image.

Everything runs in Docker. The only things you provide are two API keys.

## Quickstart

```bash
# 1. Configure your keys
cp .env.example .env
#    then edit .env and set ANTHROPIC_API_KEY and OPENAI_API_KEY

# 2. Bring it up
docker compose up --build

# 3. Open the Streamlit UI
open http://localhost:8501/
#    (the FastAPI backend is at http://localhost:8000, e.g. /docs)
```

On first boot, `db/init.sql` runs automatically and creates the schema + indexes.

## Using it

**Ingest a PDF** (or just use the upload box in the UI):

```bash
curl -F "file=@your.pdf" http://localhost:8000/ingest
```

**Ask the agent** (streamed over Server-Sent Events):

```bash
curl -N -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What does the document say about X?"}'
```

**Compare search methods directly** (no agent in the loop):

```bash
curl "http://localhost:8000/search?q=neural%20networks&method=lexical"
curl "http://localhost:8000/search?q=neural%20networks&method=semantic"
curl "http://localhost:8000/search?q=neural%20networks&method=hybrid"
curl "http://localhost:8000/search?q=neural%20networks&method=hyde"
```

## Where to read the code

Files are small and heavily commented. A good reading order for the talk:

1. **`app/search.py`** — the four search strategies (the main event).
2. **`app/hyde.py`** — how the hypothetical document is generated.
3. **`app/ingest.py`** — PDF → text → chunks → embeddings → Postgres.
4. **`app/agent.py`** — wiring the search methods up as Pydantic AI tools.
5. **`app/main.py`** — the FastAPI endpoints and SSE streaming.
6. **`db/init.sql`** — the schema and the two indexes that make search fast.
7. **`frontend/streamlit_app.py`** — the Streamlit UI (a thin client over the API).

## Configuration

All settings live in `.env` (see `.env.example`). Defaults are sensible; you
really only need the two API keys. If you change the embedding model, update
`EMBEDDING_DIM` to match and recreate the database:

```bash
docker compose down -v && docker compose up --build
```

## Evaluating the agent

The [`evaluator/`](evaluator/) directory adds a **Dagster** harness that scores
the agent with LLM-as-judge metrics (RAGAS-style) and records everything in
**Opik** for inspection. It runs as part of the same `docker compose` stack.

```bash
make opik                 # start a local Opik instance (UI at :5173)
docker compose up --build  # brings up the evaluator services too
open http://localhost:3001 # the Dagster UI -- run the `index`, then `evaluate` job
```

The `index` job pulls a small ViDoRe v3 subset (≤10 PDFs), the `evaluate` job
asks ~15 questions and judges the answers + retrieved context. Agent runs are
traced to Opik automatically (one trace per `/chat` or `/ask`, with a span per
search-tool call). See [`evaluator/README.md`](evaluator/README.md) for details.


## Local development (without Docker)

```bash
uv sync
# point DATABASE_URL at a local pgvector instance, then start the backend:
uv run uvicorn app.main:app --reload

# in another shell, start the frontend (API_URL defaults to localhost:8000):
uv run streamlit run frontend/streamlit_app.py
```

## Notes & caveats (it's a teaching repo!)

- Chunking is naive fixed-size windows with overlap. Real systems are smarter.
- No auth, no rate limiting, no multi-tenancy.
- Scanned/image-only PDFs won't extract text (no OCR here).
