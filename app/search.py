"""The four search strategies this talk is about.

Each function takes a query string and returns a ranked list of SearchResult.
They're written to be read side-by-side:

  1. lexical_search   -- Postgres full-text search (keyword / BM25-like).
  2. semantic_search  -- pgvector cosine similarity over embeddings.
  3. hybrid_search    -- Reciprocal Rank Fusion of lexical + semantic.
  4. hyde_search      -- ask the LLM for a fake answer, embed THAT, then search.
"""

from dataclasses import dataclass

from app import db
from app.embeddings import embed_one


@dataclass
class SearchResult:
    chunk_id: int
    document_id: int
    title: str
    content: str
    score: float
    method: str


# --------------------------------------------------------------------------- #
# 1. LEXICAL  -- Postgres Full Text Search                                      #
# --------------------------------------------------------------------------- #
async def lexical_search(query: str, limit: int = 5) -> list[SearchResult]:
    """Keyword search using Postgres FTS.

    - `websearch_to_tsquery` parses a Google-style query (quotes, OR, -negation).
    - `@@` is the match operator against the generated `fts` tsvector column.
    - `ts_rank_cd` scores matches by term frequency / proximity.
    """
    sql = """
        SELECT c.id, c.document_id, d.title, c.content,
               ts_rank_cd(c.fts, websearch_to_tsquery('english', %(q)s)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.fts @@ websearch_to_tsquery('english', %(q)s)
        ORDER BY score DESC
        LIMIT %(limit)s
    """
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, {"q": query, "limit": limit})
            rows = await cur.fetchall()
    return [
        SearchResult(r[0], r[1], r[2], r[3], float(r[4]), "lexical")
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# 2. SEMANTIC  -- pgvector cosine similarity                                    #
# --------------------------------------------------------------------------- #
async def semantic_search(query: str, limit: int = 5) -> list[SearchResult]:
    """Meaning-based search. Embed the query, then find the nearest chunk vectors.

    `<=>` is pgvector's cosine-distance operator (0 = identical, 2 = opposite).
    We convert distance to a 0..1 similarity score for readability.
    """
    return await _semantic_from_vector(await embed_one(query), limit, method="semantic")


async def _semantic_from_vector(
    vector: list[float], limit: int, method: str
) -> list[SearchResult]:
    """Shared core: nearest-neighbour search given an already-computed vector."""
    sql = """
        SELECT c.id, c.document_id, d.title, c.content,
               1 - (c.embedding <=> %(v)s::vector) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> %(v)s::vector
        LIMIT %(limit)s
    """
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, {"v": vector, "limit": limit})
            rows = await cur.fetchall()
    return [
        SearchResult(r[0], r[1], r[2], r[3], float(r[4]), method)
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# 3. HYBRID  -- Reciprocal Rank Fusion (RRF)                                     #
# --------------------------------------------------------------------------- #
def _rrf_fuse(
    result_lists: list[list[SearchResult]], k: int = 60, limit: int = 5
) -> list[SearchResult]:
    """Combine several ranked lists into one using Reciprocal Rank Fusion.

    RRF ignores the raw (incomparable) scores and uses only RANK position:
        score(doc) = sum over lists of  1 / (k + rank)
    A doc that ranks highly in multiple lists wins. `k` damps the influence of
    very low ranks; 60 is the value from the original RRF paper.
    """
    fused: dict[int, float] = {}
    by_id: dict[int, SearchResult] = {}
    for results in result_lists:
        for rank, res in enumerate(results):
            fused[res.chunk_id] = fused.get(res.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id[res.chunk_id] = res

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out: list[SearchResult] = []
    for chunk_id, score in ranked:
        res = by_id[chunk_id]
        out.append(
            SearchResult(res.chunk_id, res.document_id, res.title, res.content, score, "hybrid")
        )
    return out


async def hybrid_search(query: str, limit: int = 5) -> list[SearchResult]:
    """Run lexical + semantic search and fuse the rankings with RRF.

    This gets you the best of both worlds: exact keyword hits AND conceptual
    matches, without having to normalise two very different score scales.
    """
    # Pull a few extra candidates from each side before fusing.
    pool = max(limit * 2, 10)
    lexical = await lexical_search(query, limit=pool)
    semantic = await semantic_search(query, limit=pool)
    return _rrf_fuse([lexical, semantic], limit=limit)


# --------------------------------------------------------------------------- #
# 4. HyDE  -- Hypothetical Document Embeddings                                   #
# --------------------------------------------------------------------------- #
async def hyde_search(query: str, limit: int = 5) -> list[SearchResult]:
    """Hypothetical Document Embeddings.

    Short queries make poor embeddings -- they look nothing like the documents.
    HyDE asks the LLM to *write a plausible answer* to the query, then embeds
    that hypothetical answer and runs semantic search with it. The fake answer
    lives in the same 'space' as real document text, so retrieval improves.

    The generated text does NOT need to be factually correct -- it just needs to
    look like the kind of document we hope to find.
    """
    # Imported here to avoid a circular import (agent imports search).
    from app.hyde import generate_hypothetical_document

    hypothetical = await generate_hypothetical_document(query)
    vector = await embed_one(hypothetical)
    return await _semantic_from_vector(vector, limit, method="hyde")