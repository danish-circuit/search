"""FastAPI application: ingest PDFs, run the streaming agent, and demo raw search."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from sse_starlette.sse import EventSourceResponse

from app import db, search
from app.agent import agent
from app.config import settings
from app.ingest import ingest_pdf


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Open the DB pool on startup, close it on shutdown.
    await db.open_pool()
    yield
    await db.close_pool()


app = FastAPI(title="Search Agent", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Ingest                                                                       #
# --------------------------------------------------------------------------- #
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Upload a PDF; it gets extracted, chunked, embedded, and stored."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a .pdf file")
    data = await file.read()
    try:
        return await ingest_pdf(title=file.filename, pdf_bytes=data)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


# --------------------------------------------------------------------------- #
# Raw search (for the talk: compare methods directly, no agent in the loop)    #
# --------------------------------------------------------------------------- #
_METHODS = {
    "lexical": search.lexical_search,
    "semantic": search.semantic_search,
    "hybrid": search.hybrid_search,
}
# HyDE is only offered when enabled (see HYDE_ENABLED).
if settings.hyde_enabled:
    _METHODS["hyde"] = search.hyde_search


@app.get("/search")
async def raw_search(
    q: str = Query(..., description="Search query"),
    method: str = Query("hybrid", description="lexical | semantic | hybrid | hyde"),
    limit: int = Query(5, ge=1, le=20),
) -> dict:
    """Run a single search method and return ranked chunks. Great for demos."""
    fn = _METHODS.get(method)
    if fn is None:
        raise HTTPException(400, f"Unknown method '{method}'. Use one of {list(_METHODS)}.")
    results = await fn(q, limit=limit)
    return {"method": method, "query": q, "results": [asdict(r) for r in results]}


# --------------------------------------------------------------------------- #
# Streaming chat with the agent                                                #
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """Stream the agent's run over Server-Sent Events.

    We use `agent.iter()` to walk the agent graph node-by-node. This lets us:
      - stream text deltas from EVERY model turn (not just the first), and
      - emit a 'tool' event whenever the agent calls a search tool, so the UI
        can show which retrieval strategy Claude chose -- handy for the talk.
    """

    async def event_stream() -> AsyncIterator[dict]:
        async with agent.iter(req.message) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    # A model turn: stream its text deltas as they arrive.
                    turn_had_text = False
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, PartStartEvent) and isinstance(
                                event.part, TextPart
                            ):
                                if event.part.content:
                                    turn_had_text = True
                                    yield {"event": "token", "data": event.part.content}
                            elif isinstance(event, PartDeltaEvent) and isinstance(
                                event.delta, TextPartDelta
                            ):
                                if event.delta.content_delta:
                                    turn_had_text = True
                                    yield {"event": "token", "data": event.delta.content_delta}
                    # Separate this turn's text from whatever comes next (the next
                    # turn or the final answer), so markdown headings/lists that
                    # begin a turn render correctly instead of gluing on.
                    if turn_had_text:
                        yield {"event": "token", "data": "\n\n"}
                elif Agent.is_call_tools_node(node):
                    # The agent decided to call tools: announce which ones.
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, FunctionToolCallEvent):
                                # Include the search query the agent passed in.
                                args = event.part.args_as_dict()
                                query = args.get("query", "")
                                label = event.part.tool_name
                                if query:
                                    label = f"{label}({query})"
                                yield {"event": "tool", "data": label}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())


# --------------------------------------------------------------------------- #
# Config (the Streamlit frontend reads this to know which features are on)      #
# --------------------------------------------------------------------------- #
@app.get("/config")
async def config() -> dict:
    """Expose feature flags so the frontend can adapt (e.g. hide HyDE)."""
    methods = ["lexical", "semantic", "hybrid"]
    if settings.hyde_enabled:
        methods.append("hyde")
    return {"hyde_enabled": settings.hyde_enabled, "methods": methods}