"""The PDF ingest pipeline.

Flow:  PDF bytes -> extracted text -> overlapping chunks -> embeddings -> Postgres.

This is deliberately simple so it's easy to follow in a talk. A production system
would do smarter chunking (sentence/heading aware), dedup, OCR for scanned PDFs, etc.
"""

import io

from pypdf import PdfReader

from app import db
from app.config import settings
from app.embeddings import embed


def extract_text(pdf_bytes: bytes) -> str:
    """Pull plain text out of a PDF, one page after another."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping, fixed-size windows.

    Overlap keeps ideas that straddle a boundary retrievable from either chunk.
    """
    text = " ".join(text.split())  # normalise whitespace
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


async def ingest_pdf(title: str, pdf_bytes: bytes) -> dict:
    """Run the full pipeline for one PDF and persist it. Returns a small summary."""
    text = extract_text(pdf_bytes)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No extractable text found in PDF (is it a scanned image?)")

    vectors = await embed(chunks)

    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO documents (title, n_chunks) VALUES (%s, %s) RETURNING id",
                (title, len(chunks)),
            )
            row = await cur.fetchone()
            document_id = row[0]

            # Insert every chunk with its embedding.
            for i, (content, vector) in enumerate(zip(chunks, vectors, strict=True)):
                await cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (document_id, i, content, vector),
                )
        await conn.commit()

    return {"document_id": document_id, "title": title, "n_chunks": len(chunks)}