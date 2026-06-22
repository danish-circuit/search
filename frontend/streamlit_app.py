"""Streamlit frontend for the search agent.

This is a thin client over the FastAPI backend. It does three things:
  1. Upload PDFs to /ingest
  2. Chat with the agent via the streaming /chat endpoint (Server-Sent Events)
  3. Compare the raw search methods via /search

All heavy lifting lives in the backend; this file is just UI + HTTP calls.
"""

import os

import httpx
import streamlit as st

# The backend URL. In docker-compose the service is reachable as 'api'.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Search Agent", page_icon="\U0001F50D", layout="centered")
st.title("\U0001F50D Search Agent")


@st.cache_data(ttl=30)
def get_config() -> dict:
    """Fetch backend feature flags (e.g. which search methods are enabled)."""
    try:
        return httpx.get(f"{API_URL}/config", timeout=10).json()
    except httpx.HTTPError:
        return {"methods": ["lexical", "semantic", "hybrid"]}


def stream_chat(message: str):
    """Yield markdown chunks from the backend's SSE /chat endpoint.

    Designed for st.write_stream: every yielded string is appended to the
    rendered output (and to the value write_stream returns). Tool calls are
    yielded as inline badges so students see which search strategy was used.
    """
    with httpx.stream(
        "POST",
        f"{API_URL}/chat",
        json={"message": message},
        timeout=None,
    ) as resp:
        event = None
        data_lines: list[str] = []

        def harden_newlines(text: str) -> str:
            # Streamlit's st.markdown follows CommonMark, where a SINGLE newline
            # is a soft break that renders as a space -- collapsing the model's
            # line breaks into one paragraph. Convert single newlines into hard
            # breaks (two trailing spaces) while leaving blank-line paragraph
            # breaks intact, so the answer renders the way the model wrote it.
            text = text.replace("\n\n", "\x00")
            text = text.replace("\n", "  \n")
            return text.replace("\x00", "\n\n")

        def flush(event_name, lines):
            # Per the SSE spec, multiple `data:` lines in one event are joined
            # with newlines. Rejoining here preserves markdown (lists, headings).
            text = "\n".join(lines)
            if event_name == "tool":
                # Put each tool call on its own line so it stands apart.
                return f"\n\n`\U0001F527 {text}`\n\n"
            if event_name == "token":
                return harden_newlines(text)
            return None

        for line in resp.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
            elif line == "":
                # Blank line terminates an event: emit the accumulated data.
                chunk = flush(event, data_lines)
                if chunk is not None:
                    yield chunk
                event = None
                data_lines = []


cfg = get_config()

tab_chat, tab_ingest, tab_search = st.tabs(
    ["\U0001F4AC Chat", "\U0001F4C4 Ingest", "\U0001F50E Search"]
)


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
with tab_chat:
    st.caption("Ask a question. The agent picks search tools and answers from your PDFs.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer = st.write_stream(stream_chat(prompt))
        st.session_state.messages.append({"role": "assistant", "content": answer})


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
with tab_ingest:
    st.caption("Upload a PDF. It is extracted, chunked, embedded, and stored.")
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    if uploaded is not None and st.button("Ingest", type="primary"):
        with st.spinner("Indexing..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            resp = httpx.post(f"{API_URL}/ingest", files=files, timeout=None)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Indexed '{data['title']}' into {data['n_chunks']} chunks.")
        else:
            st.error(f"{resp.status_code}: {resp.text}")

# --------------------------------------------------------------------------- #
# Search (manual testing of each retrieval method, no agent involved)
# --------------------------------------------------------------------------- #
with tab_search:
    st.caption("Run a single search method directly to compare how each behaves.")
    col1, col2 = st.columns([3, 1])
    query = col1.text_input("Query", key="search_q")
    method = col2.selectbox("Method", cfg["methods"])
    if st.button("Search") and query:
        with st.spinner(f"Running {method} search..."):
            resp = httpx.get(
                f"{API_URL}/search",
                params={"q": query, "method": method, "limit": 5},
                timeout=None,
            )
        if resp.status_code != 200:
            st.error(f"{resp.status_code}: {resp.text}")
        else:
            results = resp.json()["results"]
            if not results:
                st.info("No results.")
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**{r['title']}** \u00b7 score `{r['score']:.3f}`")
                    st.write(" ".join(r["content"].split()))

with st.sidebar:
    st.subheader("Backend")
    st.code(API_URL)
    st.json(cfg)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()