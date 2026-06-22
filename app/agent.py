"""The search agent, built with Pydantic AI.

The agent is given four tools -- one per search strategy -- and a system prompt
that explains when each is appropriate. Claude decides which tool(s) to call to
answer the user's question, reads the retrieved chunks, and writes an answer with
citations. This is a compact illustration of agentic RAG.
"""

from pydantic_ai import Agent, RunContext

from app import opik, search
from app.config import settings
from app.search import SearchResult

_SYSTEM_PROMPT = """\
You are a research assistant that answers questions using a private document
library. You cannot see the documents directly -- you must retrieve relevant
passages using your search tools, then answer ONLY from what you retrieve.

You have four search tools. Choose deliberately:
  - lexical_search: best for exact terms, names, codes, acronyms, quoted phrases.
  - semantic_search: best for conceptual / paraphrased questions.
  - hybrid_search: a safe default that blends both; use when unsure.
  - hyde_search: best for short or vague questions where the user's wording is
    unlikely to match the document wording.

Guidelines:
  - Call one or more tools before answering. Prefer hybrid_search if unsure.
  - If the passages don't contain the answer, say so honestly. Do not invent facts.
  - Cite the document titles you used.
"""

agent = Agent(settings.agent_model, system_prompt=_SYSTEM_PROMPT)


def _format(results: list[SearchResult]) -> str:
    """Render search results into a compact, model-friendly string."""
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"[{r.method} | score={r.score:.3f} | doc='{r.title}']\n{r.content}")
    return "\n\n---\n\n".join(lines)


@agent.tool
async def lexical_search(ctx: RunContext, query: str) -> str:
    """Keyword / full-text search. Best for exact terms, names, codes, phrases."""
    results = await search.lexical_search(query)
    opik.record_tool_call("lexical_search", query, results)
    return _format(results)


@agent.tool
async def semantic_search(ctx: RunContext, query: str) -> str:
    """Meaning-based vector search. Best for conceptual or paraphrased questions."""
    results = await search.semantic_search(query)
    opik.record_tool_call("semantic_search", query, results)
    return _format(results)


@agent.tool
async def hybrid_search(ctx: RunContext, query: str) -> str:
    """Blended keyword + vector search (RRF). A safe default when unsure."""
    results = await search.hybrid_search(query)
    opik.record_tool_call("hybrid_search", query, results)
    return _format(results)


# HyDE is opt-out via HYDE_ENABLED. When disabled we don't register the tool at
# all, so the agent never sees it and won't try to use it.
if settings.hyde_enabled:

    @agent.tool
    async def hyde_search(ctx: RunContext, query: str) -> str:
        """HyDE: generate a hypothetical answer, embed it, then vector search.

        Best for short or vague queries.
        """
        results = await search.hyde_search(query)
        opik.record_tool_call("hyde_search", query, results)
        return _format(results)