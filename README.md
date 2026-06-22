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
                 FastAPI (one service)
  POST /ingest  ── upload a PDF, index it
  POST /chat    ── streaming agent answer (SSE)
  GET  /search  ── run one search method directly (great for demos)
  GET  /        ── tiny built-in web UI
        │
        ▼
  Postgres 16 + pgvector
   • documents / chunks tables
   • HNSW index  -> semantic search
   • GIN tsvector index -> lexical search
```

Everything runs in Docker. The only things you provide are two API keys.

## Quickstart

```bash
# 1. Configure your keys
cp .env.example .env
#    then edit .env and set ANTHROPIC_API_KEY and OPENAI_API_KEY

# 2. Bring it up
docker compose up --build

# 3. Open the UI
open http://localhost:8000/
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

## Configuration

All settings live in `.env` (see `.env.example`). Defaults are sensible; you
really only need the two API keys. If you change the embedding model, update
`EMBEDDING_DIM` to match and recreate the database:

```bash
docker compose down -v && docker compose up --build
```

## Local development (without Docker)

```bash
uv sync
# point DATABASE_URL at a local pgvector instance, then:
uv run uvicorn app.main:app --reload
```

## Notes & caveats (it's a teaching repo!)

- Chunking is naive fixed-size windows with overlap. Real systems are smarter.
- No auth, no rate limiting, no multi-tenancy.
- Scanned/image-only PDFs won't extract text (no OCR here).
