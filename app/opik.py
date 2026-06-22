"""Opik tracing for the search agent.

This is the observability layer for the talk's "evaluating agents" session. It
does two jobs:

  1. Trace every agent run to a local Opik instance: one trace per question,
     with a child span per search-tool call (recording the query the agent
     chose AND the chunks that came back). This is what makes the agent's
     retrieval visible in the Opik UI.
  2. Expose small helpers the *evaluator* service uses to post LLM-judge
     feedback scores back onto those same traces, and to group a benchmark
     run as an Opik dataset + experiment.

Tracing is opt-in: if ``OPIK_URL_OVERRIDE`` is unset we no-op, so the base demo
(Session 1) keeps running with zero Opik dependency.

The agent tools (in ``app/agent.py``) call :func:`record_tool_call` as they run.
We stash the structured ``SearchResult`` objects in a context variable so the
trace writer -- and the evaluator response -- can report exactly which
documents/chunks were retrieved, without re-parsing the model-facing strings.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.search import SearchResult


# --------------------------------------------------------------------------- #
# Per-run capture of the agent's tool calls                                    #
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """One search-tool invocation made by the agent during a run."""

    name: str
    query: str
    results: list[SearchResult] = field(default_factory=list)


@dataclass
class AgentRun:
    """The full record of one traced agent run."""

    trace_id: str
    question: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def contexts(self) -> list[str]:
        """Flat list of retrieved chunk texts, in call/rank order (deduped)."""
        seen: set[int] = set()
        out: list[str] = []
        for call in self.tool_calls:
            for r in call.results:
                if r.chunk_id in seen:
                    continue
                seen.add(r.chunk_id)
                out.append(r.content)
        return out

    def retrieved_titles(self) -> list[str]:
        """Document titles retrieved, in first-seen rank order (deduped)."""
        seen: set[str] = set()
        out: list[str] = []
        for call in self.tool_calls:
            for r in call.results:
                if r.title in seen:
                    continue
                seen.add(r.title)
                out.append(r.title)
        return out


# Holds the in-progress tool-call list for the current agent run. ``None`` means
# "not inside a traced run", so tools cheaply skip recording.
_current_calls: ContextVar[list[ToolCall] | None] = ContextVar(
    "opik_current_calls", default=None
)


def start_capture() -> object:
    """Begin capturing tool calls for one run. Returns a token for :func:`stop`."""
    return _current_calls.set([])


def stop_capture(token: object) -> list[ToolCall]:
    """Finish capturing; return the recorded calls and restore prior state."""
    calls = _current_calls.get() or []
    _current_calls.reset(token)  # type: ignore[arg-type]
    return calls


def record_tool_call(name: str, query: str, results: list[SearchResult]) -> None:
    """Called by agent tools. No-op outside a capture scope."""
    calls = _current_calls.get()
    if calls is not None:
        calls.append(ToolCall(name=name, query=query, results=list(results)))


# --------------------------------------------------------------------------- #
# Opik client (lazy, optional)                                                  #
# --------------------------------------------------------------------------- #
PROJECT_NAME = os.environ.get("OPIK_PROJECT_NAME", "search-agent")

_client: Any = None
_client_ready = False


def enabled() -> bool:
    """True when an Opik instance is configured (``OPIK_URL_OVERRIDE`` set)."""
    return bool(os.environ.get("OPIK_URL_OVERRIDE"))


def _get_client() -> Any:
    """Lazily build the high-level Opik client. Returns ``None`` if disabled."""
    global _client, _client_ready
    if _client_ready:
        return _client
    _client_ready = True
    if not enabled():
        _client = None
        return None
    try:
        import opik

        _client = opik.Opik(project_name=PROJECT_NAME)
    except Exception:  # noqa: BLE001 -- tracing must never break the app
        _client = None
    return _client


def _new_id() -> str:
    """A UUIDv7 string -- Opik requires v7 ids for traces/spans."""
    from uuid6 import uuid7

    return str(uuid7())


def write_trace(question: str, answer: str, tool_calls: list[ToolCall]) -> str | None:
    """Write one trace (+ a span per tool call) to Opik. Returns the trace id.

    Returns ``None`` when tracing is disabled or the write fails -- the caller
    treats the run as untraced and carries on.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        trace_id = _new_id()
        now = datetime.now(UTC)
        trace = client.trace(
            id=trace_id,
            name="agent-run",
            input={"question": question},
            output={"answer": answer},
            start_time=now,
            end_time=datetime.now(UTC),
            project_name=PROJECT_NAME,
        )
        for call in tool_calls:
            references = [
                {
                    "title": r.title,
                    "document_id": r.document_id,
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                }
                for r in call.results
            ]
            span_start = datetime.now(UTC)
            trace.span(
                name=f"tool-{call.name}",
                type="tool",
                input={"query": call.query},
                output={
                    "chunks": [r.content for r in call.results],
                    "references": references,
                },
                metadata={"method": call.name, "references": references},
                start_time=span_start,
                end_time=datetime.now(UTC),
            )
        client.flush()
        return trace_id
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Evaluator-facing helpers (feedback scores, datasets, experiments)            #
# --------------------------------------------------------------------------- #
def post_feedback_scores(trace_id: str, scores: list[dict[str, Any]]) -> None:
    """Attach LLM-judge scores to an existing trace.

    ``scores`` is a list of ``{name, value, reason}`` dicts. Used by the
    evaluate job after it judges an agent answer.
    """
    client = _get_client()
    if client is None or not scores:
        return
    batch = [
        {
            "id": trace_id,
            "name": s["name"],
            "value": s["value"],
            "reason": s.get("reason"),
        }
        for s in scores
    ]
    client.log_traces_feedback_scores(batch, project_name=PROJECT_NAME)
    client.flush()
