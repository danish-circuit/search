-- This script runs once, automatically, when the Postgres data volume is first created.
-- It sets up everything the app needs: the pgvector extension, the documents table,
-- the chunks table (with both an embedding column and a full-text-search column),
-- and the indexes that make semantic + lexical search fast.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded PDF.
CREATE TABLE IF NOT EXISTS documents (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title       TEXT NOT NULL,
    n_chunks    INT  NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk of text extracted from a document.
-- NOTE: the embedding dimension (1536) matches OpenAI's text-embedding-3-small.
-- If you switch embedding models, change this number AND EMBEDDING_DIM in .env,
-- then recreate the volume:  docker compose down -v && docker compose up
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    embedding     vector(1536),
    -- A generated tsvector column: Postgres keeps this in sync with `content`
    -- automatically. This is what powers lexical / full-text search.
    fts           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Index for SEMANTIC search: approximate nearest neighbour over the embeddings.
-- HNSW with cosine distance (vector_cosine_ops) is a good general-purpose default.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Index for LEXICAL search: GIN over the full-text-search column.
CREATE INDEX IF NOT EXISTS chunks_fts_idx
    ON chunks USING gin (fts);