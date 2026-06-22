"""Thin wrapper around OpenAI's embeddings API.

We keep this isolated so the rest of the app never talks to OpenAI directly --
if you wanted to swap in a local model (e.g. fastembed), this is the only file
you'd touch.
"""

from openai import AsyncOpenAI

from app.config import settings

_client = AsyncOpenAI(api_key=settings.openai_api_key)


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings into vectors."""
    if not texts:
        return []
    resp = await _client.embeddings.create(model=settings.embedding_model, input=texts)
    # The API preserves input order, so we can return embeddings as-is.
    return [item.embedding for item in resp.data]


async def embed_one(text: str) -> list[float]:
    """Embed a single string. Convenience wrapper around embed()."""
    (vector,) = await embed([text])
    return vector