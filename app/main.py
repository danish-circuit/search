"""FastAPI application: ingest PDFs, run the streaming agent, and demo raw search."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
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
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, PartStartEvent) and isinstance(
                                event.part, TextPart
                            ):
                                if event.part.content:
                                    yield {"event": "token", "data": event.part.content}
                            elif isinstance(event, PartDeltaEvent) and isinstance(
                                event.delta, TextPartDelta
                            ):
                                if event.delta.content_delta:
                                    yield {"event": "token", "data": event.delta.content_delta}
                elif Agent.is_call_tools_node(node):
                    # The agent decided to call tools: announce which ones.
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, FunctionToolCallEvent):
                                yield {
                                    "event": "tool",
                                    "data": event.part.tool_name,
                                }
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())


# --------------------------------------------------------------------------- #
# A tiny built-in UI so students can poke at it without writing a client.      #
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    # Show the HyDE option in the method dropdown only when it's enabled.
    hyde_option = "<option>hyde</option>" if settings.hyde_enabled else ""
    return _INDEX_HTML.replace("{{HYDE_OPTION}}", hyde_option)


_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Search Agent</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}
 textarea,input,select,button{font:inherit} h2{margin-top:2rem}
 #answer{white-space:pre-wrap;background:#f5f5f5;padding:1rem;border-radius:8px;min-height:3rem}
 .row{display:flex;gap:.5rem;margin:.5rem 0} .row>*{flex:1}
 pre{background:#f5f5f5;padding:1rem;border-radius:8px;overflow:auto}
</style></head><body>
<h1>\U0001F50D Search Agent</h1>

<h2>1. Ingest a PDF</h2>
<input type="file" id="pdf" accept="application/pdf">
<button onclick="ingest()">Upload</button>
<div id="ingestOut"></div>

<h2>2. Ask the agent (streaming)</h2>
<div class="row"><input id="q" placeholder="Ask a question..."></div>
<button onclick="ask()">Ask</button>
<div id="answer"></div>

<h2>3. Compare raw search methods</h2>
<div class="row">
  <input id="sq" placeholder="Search query...">
  <select id="method">
    <option>hybrid</option><option>semantic</option><option>lexical</option>{{HYDE_OPTION}}
  </select>
  <button onclick="doSearch()">Search</button>
</div>
<pre id="searchOut"></pre>

<script>
async function ingest(){
  const f=document.getElementById('pdf').files[0];
  if(!f){return}
  const fd=new FormData(); fd.append('file',f);
  document.getElementById('ingestOut').textContent='Uploading...';
  const r=await fetch('/ingest',{method:'POST',body:fd});
  document.getElementById('ingestOut').textContent=JSON.stringify(await r.json());
}
async function ask(){
  const msg=document.getElementById('q').value;
  const out=document.getElementById('answer'); out.textContent='';
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:msg})});
  const reader=r.body.getReader(); const dec=new TextDecoder(); let buf='';
  while(true){
    const {value,done}=await reader.read(); if(done)break;
    buf+=dec.decode(value,{stream:true});
    const parts=buf.split('\\n\\n'); buf=parts.pop();
    for(const p of parts){
      const ev={}; p.split('\\n').forEach(l=>{
        const i=l.indexOf(': '); if(i>0) ev[l.slice(0,i)]=l.slice(i+2);
      });
      if(ev.event==='tool' && ev.data) out.textContent+=`\U0001F527 [${ev.data}]\n`;
      if(ev.event==='token' && ev.data) out.textContent+=ev.data;
    }
  }
}
async function doSearch(){
  const q=document.getElementById('sq').value;
  const m=document.getElementById('method').value;
  const r=await fetch(`/search?q=${encodeURIComponent(q)}&method=${m}`);
  document.getElementById('searchOut').textContent=JSON.stringify(await r.json(),null,2);
}
</script>
</body></html>"""