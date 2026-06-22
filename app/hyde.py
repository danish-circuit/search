"""HyDE helper: generate a hypothetical answer document for a query."""

from pydantic_ai import Agent

from app.config import settings

_HYDE_PROMPT = (
    "You are helping a search system. Given a user's question, write a short, "
    "factual-sounding passage (2-4 sentences) that would plausibly appear in a "
    "document that answers it. Write it as an excerpt from such a document -- do "
    "NOT address the user, ask questions, or add caveats. It does not need to be "
    "true; it only needs to read like the target document."
)

_hyde_agent = Agent(settings.agent_model, system_prompt=_HYDE_PROMPT)


async def generate_hypothetical_document(query: str) -> str:
    """Return an LLM-written hypothetical document for the given query."""
    result = await _hyde_agent.run(query)
    return result.output