"""Database access: a shared async connection pool, plus startup/shutdown helpers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from app.config import settings

_pool: AsyncConnectionPool | None = None


async def _configure(conn: AsyncConnection) -> None:
    """Run for every new pooled connection: teach psycopg about the vector type."""
    await register_vector_async(conn)


async def open_pool() -> None:
    """Open the global pool. Called once on app startup."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=10,
            configure=_configure,
            open=False,
        )
        await _pool.open(wait=True)


async def close_pool() -> None:
    """Close the global pool. Called once on app shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def connection() -> AsyncIterator[AsyncConnection]:
    """Borrow a connection from the pool."""
    if _pool is None:
        raise RuntimeError("Connection pool is not open")
    async with _pool.connection() as conn:
        yield conn