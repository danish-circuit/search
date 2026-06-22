"""HTTP client for the search-agent backend (the `api` service)."""

from __future__ import annotations

import httpx

from evaluator.config import settings
from evaluator.types import GoldenPage

_INGEST_TIMEOUT = 120.0
_ASK_TIMEOUT = 300.0



def ingest_pdf(page: GoldenPage, pdf_bytes: bytes) -> dict:
    """POST one PDF to /ingest. Returns the backend's summary."""
    files = {"file": (page.pdf_name, pdf_bytes, "application/pdf")}
    resp = httpx.post(f"{settings.api_url}/ingest", files=files, timeout=_INGEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()



def ask(question: str) -> dict:
    """POST a question to /ask. Returns answer + trace_id + retrieved context."""
    resp = httpx.post(
        f"{settings.api_url}/ask",
        json={"question": question},
        timeout=_ASK_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()



def healthz() -> bool:
    try:
        resp = httpx.get(f"{settings.api_url}/healthz", timeout=10)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False